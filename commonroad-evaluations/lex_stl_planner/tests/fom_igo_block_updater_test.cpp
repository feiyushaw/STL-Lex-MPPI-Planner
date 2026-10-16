#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

#include "fom_igo_planner.hpp"

int main()
{
    const auto slices = FOMIGOBlockUpdater::buildBlockSlices(7, 3);
    assert(slices.size() == 3);
    assert(slices[0] == std::make_pair(0, 3));
    assert(slices[1] == std::make_pair(3, 6));
    assert(slices[2] == std::make_pair(6, 7));

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

    assert(mean(0, 0) < 0.0);
    assert(factors[0].allFinite());
    assert(factors[1].allFinite());
    assert(std::abs(factors[0].determinant()) > 0.0);
    assert(std::abs(factors[1].determinant()) > 0.0);

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
    assert(threw);

    return 0;
}
