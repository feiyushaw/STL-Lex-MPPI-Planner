#include <cmath>
#include <iostream>
#include <string>

#include "config.hpp"

int main()
{
    const Config cfg("config/default_config.yaml");

    if (cfg.planner.igo.sampling_method != "pureRandom")
    {
        std::cerr << "Canonical IGO sampling must default to iid Gaussian samples" << std::endl;
        return 1;
    }
    if (cfg.planner.mppi.sampling_method != "randomLHS")
    {
        std::cerr << "Legacy MPPI configuration should remain independently readable" << std::endl;
        return 1;
    }
    if (cfg.planner.igo.num_iterations != 20 ||
        cfg.planner.igo.n_samples_initial != 1000 ||
        cfg.planner.igo.n_samples_final != 100)
    {
        std::cerr << "Canonical IGO iteration/sample settings were not parsed" << std::endl;
        return 1;
    }
    if (cfg.planner.igo.initial_covariance.rows() != 2 ||
        cfg.planner.igo.initial_covariance.cols() != 2 ||
        !cfg.planner.igo.initial_covariance.allFinite())
    {
        std::cerr << "Canonical IGO initial covariance was not parsed" << std::endl;
        return 1;
    }
    if (!cfg.planner.igo.enable_operational_limit_pruning)
    {
        std::cerr << "Operational-limit prefix pruning should be enabled by the default config" << std::endl;
        return 1;
    }
    if (std::abs(cfg.planner.igo.elite_quantile - 0.05) > 1e-12 ||
        cfg.planner.igo.block_steps != 5)
    {
        std::cerr << "Canonical IGO xNES settings were not parsed" << std::endl;
        return 1;
    }

    std::cout << "Canonical IGO config and legacy-config isolation passed" << std::endl;
    return 0;
}
