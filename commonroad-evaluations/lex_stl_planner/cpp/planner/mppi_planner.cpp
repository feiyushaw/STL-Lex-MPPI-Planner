#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <atomic>
#include <algorithm>
#include <numeric>
#include "mppi_planner.hpp"
#include "profiler.hpp"
#include "static_config.hpp"

// Constructor
MPPIPlanner::MPPIPlanner(const Config &cfg,
                         const std::vector<std::vector<double>> &reference_path,
                         double corridor_width,
                         const std::vector<std::vector<double>> &speed_limits,
                         const std::vector<std::vector<double>> &stop_signs,
                         const std::vector<std::vector<double>> &intersections,
                         const std::vector<std::vector<double>> &bus_stops)
    : BasePlanner(cfg, reference_path, corridor_width, speed_limits, stop_signs, intersections, bus_stops),
      seed_(cfg.planner.mppi.seed),
      n_samples_initial_(cfg.planner.mppi.n_samples_initial),
      num_iterations_(cfg.planner.mppi.num_iterations),
      initial_covariance_(cfg.planner.mppi.initial_covariance),
      initial_lambda_(cfg.planner.mppi.initial_lambda),
      gamma_(cfg.planner.mppi.gamma),
      beta_decay_method_(cfg.planner.mppi.beta_decay_method),
      beta_min_(cfg.planner.mppi.beta_min),
      sample_count_decay_method_(cfg.planner.mppi.sample_count_decay_method),
      n_samples_final_(cfg.planner.mppi.n_samples_final),
      cfg_ptr_(&cfg),
      sampling_method_(cfg.planner.mppi.sampling_method),
      return_best_sample_(cfg.planner.mppi.return_best_sample)
{
    assert(initial_covariance_.isDiagonal() && "The covariance matrix must be diagonal!");

    if (NUM_THREADS > 0)
    {
        max_threads_ = NUM_THREADS;
    }
    else
    {
        max_threads_ = omp_get_num_procs();
    }
    omp_set_num_threads(max_threads_);

    block_slices_ = FOMIGOBlockUpdater::buildBlockSlices(K_, block_steps_);

    // Compute bit weight of highest-priority rule (OPERATIONAL_LIMITS) for Corollary 1 prefix pruning
    int total_bits = 0;
    std::vector<int> rule_bits;
    for (const auto &r : cfg.rules.rulebook_order)
    {
        int n_int = cfg.rules.rule_discretization.at(r).n_intervals;
        int b = static_cast<int>(std::ceil(std::log2(n_int + 1)));
        rule_bits.push_back(b);
        total_bits += b;
    }
    if (!rule_bits.empty() && !cfg.rules.rulebook_order.empty() && cfg.rules.rulebook_order[0] == OPERATIONAL_LIMITS)
    {
        op_limits_weight_ = static_cast<uint64_t>(1) << (total_bits - rule_bits[0]);
    }

    initializeParameterValues();
    precomputeSamples();
    initializeCostFunction(cfg);

    if (verbose_)
    {
        printPlannerInfo();
    }
}

std::vector<Eigen::MatrixXd> MPPIPlanner::initializeBlockFactors() const
{
    std::vector<Eigen::MatrixXd> A_blocks;
    A_blocks.reserve(block_slices_.size());
    Eigen::VectorXd sigma_diag = initial_covariance_.diagonal().cwiseSqrt();

    for (const auto &[start_col, num_cols] : block_slices_)
    {
        const int dim_b = u_dim_ * num_cols;
        Eigen::MatrixXd A_b = Eigen::MatrixXd::Zero(dim_b, dim_b);
        for (int c = 0; c < num_cols; ++c)
        {
            for (int d = 0; d < u_dim_; ++d)
            {
                int idx = c * u_dim_ + d;
                A_b(idx, idx) = sigma_diag(d);
            }
        }
        A_blocks.push_back(std::move(A_b));
    }
    return A_blocks;
}

std::tuple<bool, Eigen::MatrixXd, Eigen::MatrixXd, uint64_t> MPPIPlanner::forwardRolloutWithCertificatePruning(
    const Eigen::MatrixXd &control_input,
    uint64_t incumbent_best_cost) const
{
    Eigen::MatrixXd x(x_dim_, K_);
    Eigen::MatrixXd y(y_dim_, K_);

    x.col(0) = data_.x_0;
    const bool check_op_limits = enable_certificate_pruning_ && (op_limits_weight_ > 0) && (incumbent_best_cost < op_limits_weight_);
    const double v_min = cfg_ptr_->ego.v_limits[0];
    const double v_max = cfg_ptr_->ego.v_limits[1];
    const double delta_min = cfg_ptr_->ego.steering_angle_limits[0];
    const double delta_max = cfg_ptr_->ego.steering_angle_limits[1];

    for (int k = 0; k < K_ - 1; ++k)
    {
        x.col(k + 1) = system_.f(x.col(k), control_input.col(k));
        y.col(k) = system_.g(x.col(k), control_input.col(k));

        // Corollary 1 & Proposition 4 Prefix Certificate Pruning:
        // If operational limits are violated at step k+1 and an incumbent already satisfies operational limits,
        // the trajectory lower bound is at least op_limits_weight_ > incumbent_best_cost.
        if (check_op_limits)
        {
            const double delta_next = x(2, k + 1);
            const double v_next = x(3, k + 1);
            if (v_next < v_min - 1e-4 || v_next > v_max + 1e-4 || delta_next < delta_min - 1e-4 || delta_next > delta_max + 1e-4)
            {
                return {true, x, y, op_limits_weight_ + static_cast<uint64_t>(K_ - k)};
            }
        }
    }

    y.col(K_ - 1) = system_.g(x.col(K_ - 1), control_input.col(K_ - 1));
    return {false, x, y, 0};
}

// Planning implementation
PlannerResult MPPIPlanner::planImpl(const Eigen::MatrixXd &u_init)
{
    Eigen::MatrixXd u = initializeControlInput(u_init);
    std::vector<Eigen::MatrixXd> A_blocks = initializeBlockFactors();
    std::vector<std::vector<Eigen::MatrixXd>> trajectory_samples;
    trajectory_samples.reserve(num_iterations_);

    Eigen::MatrixXd best_overall_sample = u;
    uint64_t best_overall_cost = std::numeric_limits<uint64_t>::max();

    const auto start_time = std::chrono::high_resolution_clock::now();

    for (int iteration = 0; iteration < num_iterations_; ++iteration)
    {
        auto [u_new, samples, u_best, cost_best] = mppiStep(u, iteration, A_blocks, best_overall_cost);

        u = u_new;
        trajectory_samples.push_back(samples);

        if (cost_best < best_overall_cost)
        {
            best_overall_cost = cost_best;
            best_overall_sample = u_best;
        }
    }

    const auto end_time = std::chrono::high_resolution_clock::now();
    const double solve_time = std::chrono::duration<double>(end_time - start_time).count();

    auto [best_overall_x, best_overall_y] = forwardRollout(best_overall_sample);

    Eigen::MatrixXd final_u;
    Eigen::MatrixXd final_x;
    Eigen::MatrixXd final_y;

    if (return_best_sample_)
    {
        final_u = best_overall_sample;
        final_x = best_overall_x;
        final_y = best_overall_y;
    }
    else
    {
        final_u = u;
        auto [pi_x, pi_y] = forwardRollout(u);
        final_x = pi_x;
        final_y = pi_y;
    }

    BaseCostFunction *cost_func = thread_local_cost_functions_[0].get();
    const uint64_t cost = cost_func->evaluate(final_y);

    const std::vector<double> continuous_costs = cost_func->evaluateRulebook(final_y);
    const std::vector<double> discrete_costs = cost_func->evaluateRulebook(final_y, true);
    const double remaining_input_cost = calculateRemainingInputCost(final_u);

    return PlannerResult{final_x, final_u, cost, continuous_costs, discrete_costs, remaining_input_cost, solve_time, trajectory_samples, best_overall_x, true};
}

// Perform a single FOM-IGO (Block-xNES + Certificate Pruning) improvement step
std::tuple<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>, Eigen::MatrixXd, uint64_t> MPPIPlanner::mppiStep(
    Eigen::MatrixXd control_input,
    int iteration,
    std::vector<Eigen::MatrixXd> &A_blocks,
    uint64_t incumbent_best_cost)
{
    if (verbose_)
    {
        std::cout << "Processing FOM-IGO iteration: " << iteration + 1 << "/" << num_iterations_ << std::endl;
    }

    const Eigen::MatrixXd &current_covariance = covariance_schedule_[iteration];
    const double current_lambda = lambda_schedule_[iteration];
    const int n_samples = sample_count_schedule_[iteration];

    std::vector<Eigen::MatrixXd> eps(n_samples);
    std::vector<Eigen::MatrixXd> u_samples(n_samples);
    std::vector<Eigen::MatrixXd> x(n_samples);
    std::vector<Eigen::MatrixXd> y(n_samples);
    Eigen::VectorXd total_costs(n_samples);
    std::vector<uint64_t> scalarized_costs(n_samples);

    std::atomic<bool> projection_error_occurred{false};

    Eigen::MatrixXd cov_inv = current_covariance.inverse();
    Eigen::VectorXd inv_sqrt_cov = current_covariance.diagonal().cwiseSqrt().cwiseInverse();
    std::vector<Eigen::RowVectorXd> correction_weights(K_);
    for (int k = 0; k < K_; ++k)
    {
        correction_weights[k] = current_lambda * control_input.col(k).transpose() * cov_inv;
    }

    const std::vector<Eigen::MatrixXd> &eps_samples = precomputed_eps_[iteration];
    const std::vector<std::pair<int, int>> &thread_ranges = thread_sample_ranges_[iteration];
    const bool use_block_xnes = (num_iterations_ > 1);

#pragma omp parallel if (max_threads_ > 1)
    {
        int thread_id = omp_get_thread_num();
        auto [start_idx, end_idx] = thread_ranges[thread_id];
        BaseCostFunction *cost_func = thread_local_cost_functions_[thread_id].get();

        for (int n = start_idx; n < end_idx; ++n)
        {
            try
            {
                Eigen::MatrixXd eps_sample(u_dim_, K_);
                if (use_block_xnes)
                {
                    // Whiten precomputed sample and map through learned block-diagonal factor A_blocks
                    Eigen::MatrixXd s_white = inv_sqrt_cov.asDiagonal() * eps_samples[n];
                    for (size_t b = 0; b < block_slices_.size(); ++b)
                    {
                        const int s_col = block_slices_[b].first;
                        const int n_cols = block_slices_[b].second;
                        const int dim_b = u_dim_ * n_cols;
                        Eigen::MatrixXd s_blk = s_white.middleCols(s_col, n_cols);
                        Eigen::VectorXd s_vec = Eigen::Map<const Eigen::VectorXd>(s_blk.data(), dim_b);
                        Eigen::VectorXd d_vec = A_blocks[b] * s_vec;
                        eps_sample.middleCols(s_col, n_cols) =
                            Eigen::Map<const Eigen::MatrixXd>(d_vec.data(), u_dim_, n_cols);
                    }
                }
                else
                {
                    eps_sample = eps_samples[n];
                }

                Eigen::MatrixXd u_sample = control_input + eps_sample;

                if (APPLY_INPUT_CONSTRAINTS)
                {
                    applyInputConstraints(u_sample);
                    eps_sample = u_sample - control_input;
                }

                if (use_block_xnes && enable_certificate_pruning_)
                {
                    auto [pruned, x_sample, y_sample, pruned_cost] =
                        forwardRolloutWithCertificatePruning(u_sample, incumbent_best_cost);
                    if (pruned)
                    {
                        eps[n] = std::move(eps_sample);
                        u_samples[n] = std::move(u_sample);
                        x[n] = std::move(x_sample);
                        y[n] = std::move(y_sample);
                        scalarized_costs[n] = pruned_cost;
                        total_costs[n] = static_cast<double>(pruned_cost);
                        continue;
                    }
                    auto [total_cost, scalarized_cost] =
                        calculateTotalSampleCost(x_sample, y_sample, eps_sample, correction_weights, cost_func);
                    eps[n] = std::move(eps_sample);
                    u_samples[n] = std::move(u_sample);
                    x[n] = std::move(x_sample);
                    y[n] = std::move(y_sample);
                    total_costs[n] = total_cost;
                    scalarized_costs[n] = scalarized_cost;
                }
                else
                {
                    auto [x_sample, y_sample] = forwardRollout(u_sample);
                    auto [total_cost, scalarized_cost] =
                        calculateTotalSampleCost(x_sample, y_sample, eps_sample, correction_weights, cost_func);
                    eps[n] = std::move(eps_sample);
                    u_samples[n] = std::move(u_sample);
                    x[n] = std::move(x_sample);
                    y[n] = std::move(y_sample);
                    total_costs[n] = total_cost;
                    scalarized_costs[n] = scalarized_cost;
                }
            }
            catch (...)
            {
                projection_error_occurred.store(true);
            }
        }
    }

    if (projection_error_occurred.load())
    {
        throw std::runtime_error("Projection domain error occurred in parallel processing");
    }

    // IGO Quantile Ranking & Block-Diagonal xNES Natural Gradient Update (Section V-B)
    const int n_elite = std::max(2, static_cast<int>( std::round(elite_quantile_ * n_samples) ));
    std::vector<int> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(
        indices.begin(),
        indices.begin() + n_elite,
        indices.end(),
        [&](int a, int b)
        {
            if (scalarized_costs[a] != scalarized_costs[b])
            {
                return scalarized_costs[a] < scalarized_costs[b];
            }
            return total_costs[a] < total_costs[b];
        });

    std::vector<int> elite_indices(indices.begin(), indices.begin() + n_elite);
    if (use_block_xnes)
    {
        FOMIGOBlockUpdater::updateBlockXNES(
            control_input, A_blocks, block_slices_, eps, elite_indices, lr_mean_, lr_cov_);
        if (APPLY_INPUT_CONSTRAINTS)
        {
            applyInputConstraints(control_input);
        }
    }
    else
    {
        const double w_i = 1.0 / static_cast<double>(n_elite);
        for (int idx : elite_indices)
        {
            control_input += w_i * eps[idx];
        }
    }

    int best_overall_sample_idx = indices[0];
    uint64_t best_overall_cost = scalarized_costs[best_overall_sample_idx];

    return {control_input, x, u_samples[best_overall_sample_idx], best_overall_cost};
}

// Calculate cost
std::tuple<double, uint64_t> MPPIPlanner::calculateTotalSampleCost(const Eigen::MatrixXd &x,
                                                                   const Eigen::MatrixXd &y,
                                                                   const Eigen::MatrixXd &eps,
                                                                   const std::vector<Eigen::RowVectorXd> &correction_weights,
                                                                   BaseCostFunction *cost_function) const
{
    double input_cost = 0.0;
    for (int k = 0; k < K_ - 1; ++k)
    {
        input_cost += (correction_weights[k] * eps.col(k)).sum();
    }
    uint64_t output_cost = cost_function->evaluate(y);

    return {input_cost + static_cast<double>(output_cost), output_cost};
}

// Create a new cost function instance for thread-local use
std::unique_ptr<BaseCostFunction> MPPIPlanner::createLocalCostFunction() const
{
    return std::make_unique<LexicographicCostFunction>(this->data_, *cfg_ptr_, this->K_);
}

// Initialize cost functions based on parallelization settings
void MPPIPlanner::initializeCostFunction(const Config &cfg)
{
    thread_local_cost_functions_.resize(max_threads_);

    for (int i = 0; i < max_threads_; ++i)
    {
        thread_local_cost_functions_[i] = createLocalCostFunction();
        thread_local_cost_functions_[i]->setProfiler(&profiler_);
    }
}

// Calculate remaining input cost
double MPPIPlanner::calculateRemainingInputCost(const Eigen::MatrixXd &u) const
{
    Eigen::MatrixXd R_ = beta_schedule_.back() * initial_lambda_ * initial_covariance_.inverse();

    double cost = 0.0;
    for (int k = 0; k < K_; ++k)
    {
        cost += 0.5 * u.col(k).transpose() * R_ * u.col(k);
    }
    return cost;
}

// Precompute all eps samples and thread ranges for all iterations
void MPPIPlanner::precomputeSamples()
{
    precomputed_eps_.reserve(num_iterations_);
    thread_sample_ranges_.reserve(num_iterations_);

    for (int iteration = 0; iteration < num_iterations_; ++iteration)
    {
        int n_samples = sample_count_schedule_[iteration];
        Eigen::MatrixXd sqrt_cov = covariance_schedule_[iteration].array().sqrt();

        std::vector<Eigen::MatrixXd> eps_samples;
        eps_samples.reserve(n_samples);

        if (sampling_method_ == "pureRandom")
        {
            std::mt19937 gen(seed_);
            std::normal_distribution<double> dist(0.0, 1.0);

            for (int n = 0; n < n_samples; ++n)
            {
                eps_samples.push_back(generateRandomSample(dist, gen, sqrt_cov));
            }
        }
        else if (sampling_method_ == "randomLHS" || sampling_method_ == "improvedLHS" || sampling_method_ == "maximinLHS")
        {
            auto lhs_utility = std::make_unique<LHSDiagonal>(seed_);

            Eigen::MatrixXd lhs_batch = generateLHSSample(n_samples, sqrt_cov, lhs_utility.get());
            for (int n = 0; n < n_samples; ++n)
            {
                eps_samples.push_back(lhs_batch.row(n).reshaped(u_dim_, K_));
            }
        }
        else
        {
            throw std::runtime_error("Unknown sampling method: " + sampling_method_);
        }
        precomputed_eps_.push_back(std::move(eps_samples));

        std::vector<std::pair<int, int>> thread_ranges;
        thread_ranges.reserve(max_threads_);
        for (int thread_id = 0; thread_id < max_threads_; ++thread_id)
        {
            thread_ranges.push_back(calculateSampleRange(thread_id, n_samples, max_threads_));
        }
        thread_sample_ranges_.push_back(std::move(thread_ranges));
    }
}

std::pair<int, int> MPPIPlanner::calculateSampleRange(int thread_id, int n_samples, int n_threads) const
{
    int start_idx = (n_samples * thread_id) / n_threads;
    int end_idx = (n_samples * (thread_id + 1)) / n_threads;

    return {start_idx, end_idx};
}

double MPPIPlanner::calculateBeta(int iteration) const
{
    if (beta_decay_method_ == "classic")
    {
        return std::sqrt(std::pow(gamma_, iteration));
    }
    else if (beta_decay_method_ == "linear")
    {
        double ratio = static_cast<double>(iteration) / (num_iterations_ - 1);
        return 1.0 - (1.0 - beta_min_) * ratio;
    }
    else if (beta_decay_method_ == "cosine")
    {
        double ratio = 0.5 * (1 - std::cos(M_PI * iteration / (num_iterations_ - 1)));
        return 1.0 - (1.0 - beta_min_) * ratio;
    }
    else
    {
        throw std::runtime_error("Unknown beta decay method: " + beta_decay_method_);
    }
}

int MPPIPlanner::calculateNumberOfSamples(int iteration) const
{
    if (sample_count_decay_method_ == "constant")
    {
        return n_samples_initial_;
    }
    else if (sample_count_decay_method_ == "linear")
    {
        double ratio = static_cast<double>(iteration) / (num_iterations_ - 1);
        return static_cast<int>(n_samples_initial_ - (n_samples_initial_ - n_samples_final_) * ratio);
    }
    else if (sample_count_decay_method_ == "cosine")
    {
        double ratio = 0.5 * (1 - std::cos(M_PI * iteration / (num_iterations_ - 1)));
        return static_cast<int>(n_samples_initial_ - (n_samples_initial_ - n_samples_final_) * ratio);
    }
    else
    {
        throw std::runtime_error("Unknown sample-count decay method: " + sample_count_decay_method_);
    }
}

void MPPIPlanner::initializeParameterValues()
{
    beta_schedule_.reserve(num_iterations_);
    covariance_schedule_.reserve(num_iterations_);
    lambda_schedule_.reserve(num_iterations_);
    sample_count_schedule_.reserve(num_iterations_);

    for (int i = 0; i < num_iterations_; ++i)
    {
        double beta = calculateBeta(i);
        beta_schedule_.push_back(beta);
        covariance_schedule_.push_back(beta * initial_covariance_);
        lambda_schedule_.push_back(beta * beta * initial_lambda_);
        sample_count_schedule_.push_back(calculateNumberOfSamples(i));
    }
}

MPPIPlanner::ParameterSchedule MPPIPlanner::getParameterSchedule() const
{
    return MPPIPlanner::ParameterSchedule{beta_schedule_, covariance_schedule_, lambda_schedule_, sample_count_schedule_};
}

void MPPIPlanner::printPlannerInfo() const
{
    int total_samples = 0;
    for (int samples : sample_count_schedule_)
    {
        total_samples += samples;
    }

    std::cout << "\n=== FOM-IGO (Block-xNES + Certificate Pruning) Planner ===" << std::endl;
    std::cout << "Iterations:       " << num_iterations_ << std::endl;
    std::cout << "Total samples:    " << total_samples << std::endl;
    std::cout << "Block steps:      " << block_steps_ << std::endl;
    std::cout << "Elite quantile:   " << elite_quantile_ << std::endl;
    std::cout << "Cert. pruning:    " << (enable_certificate_pruning_ ? "enabled" : "disabled") << std::endl;
    std::cout << "==========================================================\n"
              << std::endl;
}

std::vector<std::string> MPPIPlanner::ruleNames() const
{
    return thread_local_cost_functions_[0]->getRulebookRuleNames();
}

Eigen::MatrixXd MPPIPlanner::initializeControlInput(const Eigen::MatrixXd &u_init) const
{
    Eigen::MatrixXd u;
    if (u_init.size() > 0)
    {
        if (u_init.rows() != u_dim_ || u_init.cols() != K_)
        {
            std::cerr << "Warning: provided u_init has wrong dimensions (" << u_init.rows() << "x" << u_init.cols() << "), initializing to zero matrix." << std::endl;
            u = Eigen::MatrixXd::Zero(u_dim_, K_);
        }
        else
        {
            u = u_init;
        }
    }
    else
    {
        u = Eigen::MatrixXd::Zero(u_dim_, K_);
    }
    return u;
}

Eigen::MatrixXd MPPIPlanner::generateRandomSample(std::normal_distribution<double> &distribution,
                                                  std::mt19937 &generator,
                                                  const Eigen::MatrixXd &sqrt_covariance) const
{
    Eigen::MatrixXd eps(u_dim_, K_);

    for (int n = 0; n < u_dim_; ++n)
    {
        for (int k = 0; k < K_; ++k)
        {
            eps(n, k) = distribution(generator) * sqrt_covariance(n, n);
        }
    }
    return eps;
}

Eigen::MatrixXd MPPIPlanner::generateLHSSample(int n_samples, const Eigen::MatrixXd &sqrt_covariance, LHSDiagonal *lhs_utility) const
{
    const int total_dims = u_dim_ * K_;

    Eigen::MatrixXd lhs_samples;

    if (sampling_method_ == "randomLHS")
    {
        lhs_samples = lhs_utility->randomLHS(n_samples, total_dims);
    }
    else if (sampling_method_ == "improvedLHS")
    {
        lhs_samples = lhs_utility->improvedLHS(n_samples, total_dims);
    }
    else if (sampling_method_ == "maximinLHS")
    {
        lhs_samples = lhs_utility->maximinLHS(n_samples, total_dims);
    }

    Eigen::MatrixXd normal_samples = LHSDiagonal::uniformToNormal(lhs_samples);

    Eigen::VectorXd sqrt_var_u(u_dim_);
    for (int u = 0; u < u_dim_; ++u)
    {
        sqrt_var_u(u) = sqrt_covariance(u, u);
    }
    Eigen::VectorXd stddev_vector = LHSDiagonal::createTimeSeriesStddevVector(sqrt_var_u, K_);

    return LHSDiagonal::applyDiagonalScaling(normal_samples, stddev_vector);
}
