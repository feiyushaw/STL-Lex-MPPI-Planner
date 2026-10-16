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

    if (num_iterations_ < 0)
    {
        throw std::invalid_argument("IGO num_iterations must be non-negative");
    }
    if (n_samples_initial_ <= 0)
    {
        throw std::invalid_argument("IGO n_samples_initial must be positive");
    }
    if (n_samples_final_ <= 0)
    {
        throw std::invalid_argument("IGO n_samples_final must be positive");
    }
    if (u_dim_ <= 0 || K_ <= 1)
    {
        throw std::invalid_argument("IGO requires positive input dimension and horizon");
    }
    control_horizon_ = K_ - 1;
    if (initial_covariance_.rows() != u_dim_ ||
        initial_covariance_.cols() != u_dim_ ||
        !initial_covariance_.allFinite() ||
        !initial_covariance_.isDiagonal())
    {
        throw std::invalid_argument("IGO initial covariance must be finite, square, and diagonal");
    }
    if ((initial_covariance_.diagonal().array() <= 0.0).any())
    {
        throw std::invalid_argument("IGO initial covariance must be positive definite on the diagonal");
    }

    // Compute bit weight of highest-priority rule (OPERATIONAL_LIMITS) for Corollary 1 prefix pruning
    int total_bits = 0;
    std::vector<int> rule_bits;
    for (const auto &r : cfg.rules.rulebook_order)
    {
        int n_int = cfg.rules.rule_discretization.at(r).n_intervals;
        if (n_int <= 0)
        {
            throw std::invalid_argument("Rule discretization must contain a positive interval count");
        }
        int b = static_cast<int>(std::ceil(std::log2(n_int + 1)));
        rule_bits.push_back(b);
        total_bits += b;
    }
    if (total_bits >= 64)
    {
        throw std::invalid_argument("Lexicographic rule encoding exceeds uint64 capacity");
    }
    if (!rule_bits.empty() && !cfg.rules.rulebook_order.empty() && cfg.rules.rulebook_order[0] == OPERATIONAL_LIMITS)
    {
        op_limits_weight_ = static_cast<uint64_t>(1) << (total_bits - rule_bits[0]);
    }

    const auto &igo_cfg = cfg.planner.mppi.igo;
    block_steps_ = igo_cfg.block_steps;
    elite_quantile_ = igo_cfg.elite_quantile;
    lr_mean_ = igo_cfg.lr_mean;
    lr_cov_ = igo_cfg.lr_cov;
    enable_certificate_pruning_ = igo_cfg.enable_certificate_pruning;

    if (block_steps_ <= 0 || elite_quantile_ <= 0.0 || elite_quantile_ > 1.0 ||
        lr_mean_ < 0.0 || lr_cov_ < 0.0)
    {
        throw std::invalid_argument("Invalid FOM-IGO configuration");
    }

    block_slices_ = FOMIGOBlockUpdater::buildBlockSlices(control_horizon_, block_steps_);

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

    for (const auto &block_slice : block_slices_)
    {
        const int num_cols = block_slice.second - block_slice.first;
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
                // Complete the state/output trajectory after the certificate
                // decision so pruned samples remain structurally valid for
                // visualization and diagnostics.
                for (int rem_k = k + 1; rem_k < K_ - 1; ++rem_k)
                {
                    x.col(rem_k + 1) = system_.f(x.col(rem_k), control_input.col(rem_k));
                    y.col(rem_k) = system_.g(x.col(rem_k), control_input.col(rem_k));
                }
                y.col(K_ - 1) = system_.g(x.col(K_ - 1), control_input.col(K_ - 1));
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
    Eigen::MatrixXd initial_y = std::get<1>(forwardRollout(u));
    BaseCostFunction *initial_cost_func = thread_local_cost_functions_[0].get();
    uint64_t best_overall_cost = initial_cost_func->evaluate(initial_y);

    const auto start_time = std::chrono::high_resolution_clock::now();

    for (int iteration = 0; iteration < num_iterations_; ++iteration)
    {
        auto [u_new, samples, u_best, cost_best] = igoStep(u, iteration, A_blocks, best_overall_cost);

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

// Perform a single FOM-IGO (block-xNES + certificate pruning) improvement step.
std::tuple<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>, Eigen::MatrixXd, uint64_t> MPPIPlanner::igoStep(
    Eigen::MatrixXd control_input,
    int iteration,
    std::vector<Eigen::MatrixXd> &A_blocks,
    uint64_t incumbent_best_cost)
{
    if (verbose_)
    {
        std::cout << "Processing FOM-IGO iteration: " << iteration + 1 << "/" << num_iterations_ << std::endl;
    }

    const int n_samples = sample_count_schedule_[iteration];

    std::vector<Eigen::MatrixXd> standard_samples(n_samples);
    std::vector<Eigen::MatrixXd> u_samples(n_samples);
    std::vector<Eigen::MatrixXd> x(n_samples);
    std::vector<Eigen::MatrixXd> y(n_samples);
    Eigen::VectorXd total_costs(n_samples);
    std::vector<uint64_t> scalarized_costs(n_samples);

    std::atomic<bool> projection_error_occurred{false};

    // Precomputed samples are already whitened latent vectors s ~ N(0,I).
    const std::vector<Eigen::MatrixXd> &latent_samples =
        precomputed_standard_samples_[iteration];
    const std::vector<std::pair<int, int>> &thread_ranges = thread_sample_ranges_[iteration];

#pragma omp parallel if (max_threads_ > 1)
    {
        int thread_id = omp_get_thread_num();
        auto [start_idx, end_idx] = thread_ranges[thread_id];
        BaseCostFunction *cost_func = thread_local_cost_functions_[thread_id].get();

        for (int n = start_idx; n < end_idx; ++n)
        {
            try
            {
                const Eigen::MatrixXd &s_sample = latent_samples[n];
                if (s_sample.rows() != u_dim_ || s_sample.cols() != control_horizon_ ||
                    !s_sample.allFinite())
                {
                    throw std::runtime_error("Invalid precomputed IGO latent sample");
                }

                Eigen::MatrixXd eps_sample = Eigen::MatrixXd::Zero(u_dim_, K_);
                for (size_t b = 0; b < block_slices_.size(); ++b)
                {
                    const int s_col = block_slices_[b].first;
                    const int n_cols = block_slices_[b].second - block_slices_[b].first;
                    const int dim_b = u_dim_ * n_cols;
                    const Eigen::VectorXd s_vec =
                        Eigen::Map<const Eigen::VectorXd>(
                            s_sample.middleCols(s_col, n_cols).data(), dim_b);
                    const Eigen::VectorXd d_vec = A_blocks[b] * s_vec;
                    eps_sample.middleCols(s_col, n_cols) =
                        Eigen::Map<const Eigen::MatrixXd>(
                            d_vec.data(), u_dim_, n_cols);
                }

                Eigen::MatrixXd u_sample = control_input + eps_sample;

                if (APPLY_INPUT_CONSTRAINTS)
                {
                    applyInputConstraints(u_sample);
                    // The natural-gradient statistics remain in the original
                    // whitened coordinates s, before projection.
                }

                if (enable_certificate_pruning_)
                {
                    auto [pruned, x_sample, y_sample, pruned_cost] =
                        forwardRolloutWithCertificatePruning(u_sample, incumbent_best_cost);
                    if (pruned)
                    {
                        standard_samples[n] = s_sample;
                        u_samples[n] = std::move(u_sample);
                        x[n] = std::move(x_sample);
                        y[n] = std::move(y_sample);
                        scalarized_costs[n] = pruned_cost;
                        total_costs[n] = static_cast<double>(pruned_cost);
                        continue;
                    }
                    auto [total_cost, scalarized_cost] =
                        calculateSampleObjective(y_sample, u_sample, cost_func);
                    standard_samples[n] = s_sample;
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
                        calculateSampleObjective(y_sample, u_sample, cost_func);
                    standard_samples[n] = s_sample;
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
    const int n_elite = std::max(
        1,
        std::min(
            n_samples,
            static_cast<int>(std::ceil(elite_quantile_ * static_cast<double>(n_samples)))));
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
    FOMIGOBlockUpdater::updateBlockXNES(
        control_input, A_blocks, block_slices_, standard_samples, elite_indices, lr_mean_, lr_cov_);
    if (APPLY_INPUT_CONSTRAINTS)
    {
        applyInputConstraints(control_input);
    }

    int best_overall_sample_idx = indices[0];
    uint64_t best_overall_cost = scalarized_costs[best_overall_sample_idx];

    return {control_input, x, u_samples[best_overall_sample_idx], best_overall_cost};
}

// Evaluate the Lex-STL objective. IGO uses rank/quantile selection, so no
// MPPI exponential temperature or importance-sampling correction is applied.
std::tuple<double, uint64_t> MPPIPlanner::calculateSampleObjective(
    const Eigen::MatrixXd &y,
    const Eigen::MatrixXd &u,
    BaseCostFunction *cost_function) const
{
    const uint64_t lex_cost = cost_function->evaluate(y);
    double secondary_cost = 0.0;
    if (K_ > 1)
    {
        // Only a tie-breaker: this can never override the Lex-STL key.
        secondary_cost = 1e-6 * u.leftCols(K_ - 1).squaredNorm();
    }
    return {secondary_cost, lex_cost};
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
    // Reporting-only effort term; it is not part of IGO ranking.
    Eigen::MatrixXd R_ = initial_lambda_ * initial_covariance_.inverse();

    double cost = 0.0;
    for (int k = 0; k < control_horizon_; ++k)
    {
        cost += 0.5 * u.col(k).transpose() * R_ * u.col(k);
    }
    return cost;
}

// Precompute whitened latent samples s ~ N(0,I) and thread ranges.
void MPPIPlanner::precomputeSamples()
{
    precomputed_standard_samples_.reserve(num_iterations_);
    thread_sample_ranges_.reserve(num_iterations_);

    for (int iteration = 0; iteration < num_iterations_; ++iteration)
    {
        int n_samples = sample_count_schedule_[iteration];
        std::vector<Eigen::MatrixXd> standard_samples;
        standard_samples.reserve(n_samples);

        if (sampling_method_ == "pureRandom")
        {
            std::mt19937 gen(seed_ + static_cast<unsigned int>(iteration));
            std::normal_distribution<double> dist(0.0, 1.0);

            for (int n = 0; n < n_samples; ++n)
            {
                standard_samples.push_back(generateStandardNormalSample(dist, gen));
            }
        }
        else if (sampling_method_ == "randomLHS" || sampling_method_ == "improvedLHS" || sampling_method_ == "maximinLHS")
        {
            auto lhs_utility = std::make_unique<LHSDiagonal>(
                seed_ + static_cast<unsigned int>(iteration));

            Eigen::MatrixXd lhs_batch =
                generateStandardLHSSample(n_samples, lhs_utility.get());
            for (int n = 0; n < n_samples; ++n)
            {
                standard_samples.push_back(lhs_batch.row(n).reshaped(u_dim_, control_horizon_));
            }
        }
        else
        {
            throw std::runtime_error("Unknown sampling method: " + sampling_method_);
        }
        precomputed_standard_samples_.push_back(std::move(standard_samples));

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

int MPPIPlanner::calculateNumberOfSamples(int iteration) const
{
    if (sample_count_decay_method_ == "constant")
    {
        return n_samples_initial_;
    }
    else if (sample_count_decay_method_ == "linear")
    {
        if (num_iterations_ <= 1)
        {
            return n_samples_initial_;
        }
        double ratio = static_cast<double>(iteration) / (num_iterations_ - 1);
        return static_cast<int>(n_samples_initial_ - (n_samples_initial_ - n_samples_final_) * ratio);
    }
    else if (sample_count_decay_method_ == "cosine")
    {
        if (num_iterations_ <= 1)
        {
            return n_samples_initial_;
        }
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
        // Compatibility-only schedule values. IGO keeps covariance in A and
        // has no beta/temperature shrinkage.
        beta_schedule_.push_back(1.0);
        covariance_schedule_.push_back(initial_covariance_);
        lambda_schedule_.push_back(initial_lambda_);
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
    std::cout << "Mean LR:          " << lr_mean_ << std::endl;
    std::cout << "Covariance LR:    " << lr_cov_ << std::endl;
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

Eigen::MatrixXd MPPIPlanner::generateStandardNormalSample(
    std::normal_distribution<double> &distribution,
    std::mt19937 &generator) const
{
    Eigen::MatrixXd sample(u_dim_, control_horizon_);
    for (int n = 0; n < u_dim_; ++n)
    {
        for (int k = 0; k < control_horizon_; ++k)
        {
            sample(n, k) = distribution(generator);
        }
    }
    return sample;
}

Eigen::MatrixXd MPPIPlanner::generateStandardLHSSample(
    int n_samples, LHSDiagonal *lhs_utility) const
{
    const int total_dims = u_dim_ * control_horizon_;
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
    else
    {
        throw std::runtime_error("Unknown LHS sampling method: " + sampling_method_);
    }

    return LHSDiagonal::uniformToNormal(lhs_samples);
}
