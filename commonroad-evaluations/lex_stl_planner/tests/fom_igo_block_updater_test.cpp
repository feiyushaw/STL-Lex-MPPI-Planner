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

    require(mean(0, 0) < 0.0, "elite samples should move the mean");
    require(factors[0].allFinite(), "first covariance factor is finite");
    require(factors[1].allFinite(), "second covariance factor is finite");
    require(std::abs(factors[0].determinant()) > 0.0, "first covariance factor remains nonsingular");
    require(std::abs(factors[1].determinant()) > 0.0, "second covariance factor remains nonsingular");

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
