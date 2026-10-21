#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

#include "fom_igo_planner.hpp"

int main()
{
    const auto require = [](bool condition, const char *message) {
        if (!condition) {
            std::cerr << "FAIL: " << message << std::endl;
            std::exit(1);
        }
    };

    require(isSupportedFOMIGOSamplingMethod("pureRandom"),
            "iid Gaussian latent sampling should be supported");
    require(isSupportedFOMIGOSamplingMethod("randomLHS"),
            "LHS should remain available as an opt-in comparison mode");
    require(!isSupportedFOMIGOSamplingMethod("mppi"),
            "unknown sampling policies must be rejected");

    const auto slices = FOMIGOBlockUpdater::buildBlockSlices(7, 3);
    require(slices.size() == 3, "three temporal blocks expected");
    require(slices[0] == std::make_pair(0, 3), "first block mismatch");
    require(slices[1] == std::make_pair(3, 6), "second block mismatch");
    require(slices[2] == std::make_pair(6, 7), "last partial block mismatch");

    Eigen::MatrixXd mean = Eigen::MatrixXd::Zero(1, 4);
    std::vector<Eigen::MatrixXd> factors{
        Eigen::MatrixXd::Identity(2, 2),
        Eigen::MatrixXd::Identity(2, 2)};
    const auto block_slices = FOMIGOBlockUpdater::buildBlockSlices(4, 2);

    std::vector<Eigen::MatrixXd> samples(4, Eigen::MatrixXd::Zero(1, 4));
    samples[0](0, 0) = -2.0;
    samples[1](0, 0) = -1.0;
    samples[2](0, 0) = 1.0;
    samples[3](0, 0) = 2.0;

    FOMIGOBlockUpdater::updateBlockXNES(
        mean,
        factors,
        block_slices,
        samples,
        std::vector<int>{0, 1},
        1.0,
        0.25);

    require(std::abs(mean(0, 0) + 1.5) < 1e-12,
            "mean update must use only the two selected whitened elite samples");
    require(std::abs(mean(0, 1)) < 1e-12,
            "zero-mean latent coordinate must not move the control mean");
    require(std::abs(factors[0](0, 0) - std::exp(0.3125)) < 1e-12,
            "xNES factor update must match A exp(lr_cov * G_M / 2)");
    require(std::abs(factors[0](1, 1) - std::exp(-0.125)) < 1e-12,
            "xNES covariance update must include the -I baseline term");
    require(factors[0].allFinite(), "first covariance factor is finite");
    require(factors[1].allFinite(), "second covariance factor is finite");
    require(std::abs(factors[0].determinant()) > 0.0, "first covariance factor remains nonsingular");
    require(std::abs(factors[1].determinant()) > 0.0, "second covariance factor remains nonsingular");

    // Regression: verify that half-open slices (0,3), (3,6), (6,7)
    // produce covariance blocks of dimensions 3, 3, and 1 for u_dim=1.
    Eigen::MatrixXd partial_mean = Eigen::MatrixXd::Zero(1, 7);
    const auto partial_slices = FOMIGOBlockUpdater::buildBlockSlices(7, 3);
    std::vector<Eigen::MatrixXd> partial_factors{
        Eigen::MatrixXd::Identity(3, 3),
        Eigen::MatrixXd::Identity(3, 3),
        Eigen::MatrixXd::Identity(1, 1)};
    std::vector<Eigen::MatrixXd> partial_samples(2, Eigen::MatrixXd::Zero(1, 7));
    partial_samples[0].setConstant(-1.0);
    partial_samples[1].setConstant(-0.5);
    FOMIGOBlockUpdater::updateBlockXNES(
        partial_mean,
        partial_factors,
        partial_slices,
        partial_samples,
        std::vector<int>{0, 1},
        1.0,
        0.0);
    require(partial_mean.allFinite(), "partial-block mean update is finite");
    require(partial_factors[0].rows() == 3 && partial_factors[0].cols() == 3,
            "first partial covariance block has correct dimension");
    require(partial_factors[1].rows() == 3 && partial_factors[1].cols() == 3,
            "second partial covariance block has correct dimension");
    require(partial_factors[2].rows() == 1 && partial_factors[2].cols() == 1,
            "last partial covariance block has correct dimension");

    bool threw = false;
    try
    {
        FOMIGOBlockUpdater::updateBlockXNES(
            mean,
            factors,
            block_slices,
            samples,
            std::vector<int>{99},
            1.0,
            0.25);
    }
    catch (const std::out_of_range &)
    {
        threw = true;
    }
    require(threw, "invalid elite index must be rejected");

    return 0;
}
