#pragma once

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

// FOM-IGO primal search utilities.
//
// The active optimizer follows the implicit-FOM primal realization described
// in Section V-B of Wei et al. (arXiv:2603.16231v3):
//   u = m + A s,  s ~ N(0,I)
// followed by IGO quantile selection and blockwise xNES updates.
struct FOMIGOBlockUpdater
{
    static std::vector<std::pair<int, int>> buildBlockSlices(int K, int block_steps)
    {
        if (K <= 0)
        {
            throw std::invalid_argument("K must be positive");
        }
        if (block_steps <= 0)
        {
            throw std::invalid_argument("block_steps must be positive");
        }

        std::vector<std::pair<int, int>> slices;
        for (int start = 0; start < K; start += block_steps)
        {
            slices.emplace_back(start, std::min(K, start + block_steps));
        }
        return slices;
    }

    // Apply the xNES natural-gradient update to a block-diagonal Gaussian.
    // standard_samples contains the actual whitened latent samples s_i used
    // to generate each rollout; no inverse-A reconstruction is required.
    static void updateBlockXNES(
        Eigen::MatrixXd &mean_control,
        std::vector<Eigen::MatrixXd> &A_blocks,
        const std::vector<std::pair<int, int>> &block_slices,
        const std::vector<Eigen::MatrixXd> &standard_samples,
        const std::vector<int> &elite_indices,
        double lr_mean,
        double lr_cov)
    {
        if (elite_indices.empty())
        {
            throw std::invalid_argument("xNES requires at least one elite sample");
        }
        if (lr_mean < 0.0 || lr_cov < 0.0)
        {
            throw std::invalid_argument("xNES learning rates must be non-negative");
        }
        if (block_slices.size() != A_blocks.size())
        {
            throw std::invalid_argument("Block slices and covariance factors must have equal size");
        }

        const double weight = 1.0 / static_cast<double>(elite_indices.size());

        for (size_t b = 0; b < block_slices.size(); ++b)
        {
            const auto [start_col, end_col] = block_slices[b];
            const int num_cols = end_col - start_col;
            const int dim_b = mean_control.rows() * num_cols;
            Eigen::MatrixXd &A_b = A_blocks[b];

            if (A_b.rows() != dim_b || A_b.cols() != dim_b)
            {
                throw std::invalid_argument("Covariance block has inconsistent dimensions");
            }

            Eigen::VectorXd grad_delta = Eigen::VectorXd::Zero(dim_b);
            Eigen::MatrixXd grad_M = Eigen::MatrixXd::Zero(dim_b, dim_b);

            for (int sample_idx : elite_indices)
            {
                if (sample_idx < 0 || sample_idx >= static_cast<int>(standard_samples.size()))
                {
                    throw std::out_of_range("Elite sample index is outside the sample set");
                }

                const Eigen::MatrixXd &s_sample = standard_samples[sample_idx];
                if (s_sample.rows() != mean_control.rows() ||
                    start_col + num_cols > s_sample.cols())
                {
                    throw std::invalid_argument("Whitened sample has inconsistent dimensions");
                }
                if (!s_sample.allFinite())
                {
                    throw std::invalid_argument("Whitened sample contains non-finite values");
                }

                const Eigen::MatrixXd s_block =
                    s_sample.middleCols(start_col, num_cols);
                const Eigen::VectorXd s_vec =
                    Eigen::Map<const Eigen::VectorXd>(s_block.data(), dim_b);

                grad_delta += weight * s_vec;
                grad_M += weight * (s_vec * s_vec.transpose());
            }

            grad_M -= Eigen::MatrixXd::Identity(dim_b, dim_b);
            const Eigen::MatrixXd symmetric_grad =
                0.5 * (grad_M + grad_M.transpose());

            const Eigen::VectorXd delta_m =
                lr_mean * (A_b * grad_delta);
            mean_control.middleCols(start_col, num_cols) +=
                Eigen::Map<const Eigen::MatrixXd>(
                    delta_m.data(), mean_control.rows(), num_cols);

            const Eigen::MatrixXd A_new =
                A_b * (0.5 * lr_cov * symmetric_grad).exp();

            if (!A_new.allFinite())
            {
                throw std::runtime_error("xNES covariance update produced non-finite values");
            }
            A_b = A_new;
        }
    }
};
