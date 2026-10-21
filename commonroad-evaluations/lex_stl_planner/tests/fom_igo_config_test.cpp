#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include <yaml-cpp/yaml.h>
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

    // Remove the canonical section from a temporary copy to exercise the
    // backwards-compatible fallback to planner.mppi and planner.mppi.igo.
    YAML::Node legacy_yaml = YAML::LoadFile("config/default_config.yaml");
    legacy_yaml["planner"].remove("igo");
    const std::filesystem::path legacy_path =
        std::filesystem::temp_directory_path() / "fom_igo_legacy_config_fallback.yaml";
    YAML::Emitter emitter;
    emitter << legacy_yaml;
    {
        std::ofstream legacy_file(legacy_path);
        if (!legacy_file)
        {
            std::cerr << "Could not create temporary legacy config" << std::endl;
            return 1;
        }
        legacy_file << emitter.c_str();
    }
    const Config legacy_cfg(legacy_path.string());
    std::filesystem::remove(legacy_path);
    if (legacy_cfg.planner.igo.num_iterations != legacy_cfg.planner.mppi.num_iterations ||
        legacy_cfg.planner.igo.n_samples_initial != legacy_cfg.planner.mppi.n_samples_initial ||
        legacy_cfg.planner.igo.sampling_method != "pureRandom")
    {
        std::cerr << "Legacy YAML did not seed the canonical IGO config correctly" << std::endl;
        return 1;
    }

    std::cout << "Canonical IGO config and legacy-config isolation passed" << std::endl;
    return 0;
}
