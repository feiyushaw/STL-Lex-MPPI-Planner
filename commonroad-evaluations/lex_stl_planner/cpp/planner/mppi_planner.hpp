#pragma once

#include <vector>
#include <tuple>
#include <utility>
#include <chrono>
#include <random>
#include <atomic>
#include <memory>

#include <Eigen/Dense>
#include <omp.h>

#include "base_planner.hpp"
#include "lhs_utility.hpp"
#include "fom_igo_planner.hpp"

// Legacy class name retained for ABI/config compatibility. The active
// optimization backend is FOM-IGO (xNES), not MPPI.
class MPPIPlanner : public BasePlanner
{
public:
    // Parameter schedules indexed by iteration
    struct ParameterSchedule
    {
        std::vector<double> beta_schedule;
        std::vector<Eigen::MatrixXd> covariance_schedule;
        std::vector<double> lambda_schedule;
        std::vector<int> sample_count_schedule;
    };

    // Constructor
    explicit MPPIPlanner(const Config &cfg,
                         const std::vector<std::vector<double>> &reference_path,
                         double corridor_width,
                         const std::vector<std::vector<double>> &speed_limits,
                         const std::vector<std::vector<double>> &stop_signs,
                         const std::vector<std::vector<double>> &intersections,
                         const std::vector<std::vector<double>> &bus_stops);

    // Get parameter schedules over iterations
    ParameterSchedule getParameterSchedule() const;

    // Initialize cost function
    void initializeCostFunction(const Config &cfg) override;

    // Get rule names
    std::vector<std::string> ruleNames() const override;

protected:
    // Planning implementation
    PlannerResult planImpl(const Eigen::MatrixXd &u_init = Eigen::MatrixXd()) override;

private:
    // Core algorithm methods
    std::tuple<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>, Eigen::MatrixXd, uint64_t> igoStep(
        Eigen::MatrixXd control_input,
        int iteration,
        std::vector<Eigen::MatrixXd> &A_blocks,
        uint64_t incumbent_best_cost);

    std::tuple<bool, Eigen::MatrixXd, Eigen::MatrixXd, uint64_t> forwardRolloutWithOperationalLimitPruning(
        const Eigen::MatrixXd &control_input,
        uint64_t incumbent_best_cost) const;

    std::tuple<double, uint64_t> calculateSampleObjective(const Eigen::MatrixXd &y,
                                                           const Eigen::MatrixXd &u,
                                                           BaseCostFunction *cost_function) const;

    double calculateRemainingInputCost(const Eigen::MatrixXd &u) const;

    // Cost function management
    std::unique_ptr<BaseCostFunction> createLocalCostFunction() const;

    // Iteration/sample schedule. Beta and temperature remain exposed only
    // through ParameterSchedule for compatibility; IGO does not use them.
    int calculateNumberOfSamples(int iteration) const;
    void initializeParameterValues();

    // Sampling
    void precomputeSamples();
    std::pair<int, int> calculateSampleRange(int thread_id, int n_samples, int n_threads) const;
    Eigen::MatrixXd generateStandardNormalSample(std::normal_distribution<double> &distribution,
                                                  std::mt19937 &generator) const;
    Eigen::MatrixXd generateStandardLHSSample(int n_samples, LHSDiagonal *lhs_utility) const;

    // Utility methods
    void printPlannerInfo() const;
    Eigen::MatrixXd initializeControlInput(const Eigen::MatrixXd &u_init) const;
    std::vector<Eigen::MatrixXd> initializeBlockFactors() const;

    // Configuration parameters
    unsigned int seed_;
    int control_horizon_;
    int n_samples_initial_;
    int num_iterations_;
    int max_threads_;

    // Active IGO configuration. The covariance factor A is evolved by xNES;
    // no MPPI temperature or beta schedule participates in the optimizer.
    Eigen::MatrixXd initial_covariance_;
    double reporting_effort_weight_;
    std::string sample_count_decay_method_;
    int n_samples_final_;
    std::string sampling_method_;
    bool return_best_sample_;

    // FOM-IGO Block-xNES & Certificate Pruning parameters
    int block_steps_ = 5;
    double elite_quantile_ = 0.05;
    double lr_mean_ = 1.0;
    double lr_cov_ = 0.25;
    bool enable_operational_limit_pruning_ = true;
    std::vector<std::pair<int, int>> block_slices_;
    uint64_t op_limits_weight_ = 0;

    // Config reference for cost function creation
    const Config *cfg_ptr_;

    // Compatibility schedule. IGO keeps A as the covariance factor and does
    // not apply MPPI beta/lambda schedules during sampling or ranking.
    std::vector<double> beta_schedule_;
    std::vector<Eigen::MatrixXd> covariance_schedule_;
    std::vector<double> lambda_schedule_;
    std::vector<int> sample_count_schedule_;

    // Thread-local cost function instances
    std::vector<std::unique_ptr<BaseCostFunction>> thread_local_cost_functions_;

    // Precomputed whitened latent samples s ~ N(0,I) for all iterations.
    std::vector<std::vector<Eigen::MatrixXd>> precomputed_standard_samples_;

    // Precomputed sample ranges for each thread for all iterations
    std::vector<std::vector<std::pair<int, int>>> thread_sample_ranges_;
};
