#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

/**
 * @brief Implicit FOM with Certificate-Guided Block-Diagonal xNES (IGO) utilities.
 * Implements Section III-B (Corollary 1 & Proposition 4 prefix pruning) and
 * Section V-B (Block-Diagonal xNES natural gradient updates) from
 * Wei et al. (arXiv:2603.16231v3) over Lexicographic Bit-Shifted STL Costs.
 */
struct FOMIGOBlockUpdater
{
    static std::vector<std::pair<int, int>> buildBlockSlices(int horizon_steps, int block_steps)
    {
        std::vector<std::pair<int, int>> slices;
        for (int s = 0; s < horizon_steps; s += block_steps)
        {
            int len = std::min(block_steps, horizon_steps - s);
            slices.emplace_back(s, len);
        }
        return slices;
    }

    static void updateBlockXNES(
        Eigen::MatrixXd &control_input,
        std::vector<Eigen::MatrixXd> &A_blocks,
        const std::vector<std::pair<int, int>> &block_slices,
        const std::vector<Eigen::MatrixXd> &eps_samples,
        const std::vector<int> &elite_indices,
        double lr_mean = 1.0,
        double lr_cov = 0.25)
    {
        const int n_elite = static_cast<int>(elite_indices.size());
        if (n_elite <= 0)
        {
            return;
        }
        const int u_dim = static_cast<int>(control_input.rows());
        const double w_i = 1.0 / static_cast<double>(n_elite);

        for (size_t b = 0; b < block_slices.size(); ++b)
        {
            const int start_col = block_slices[b].first;
            const int num_cols = block_slices[b].second;
            const int dim_b = u_dim * num_cols;

            const Eigen::MatrixXd &A_b = A_blocks[b];
            Eigen::FullPivLU<Eigen::MatrixXd> lu_A(A_b);

            Eigen::VectorXd grad_delta = Eigen::VectorXd::Zero(dim_b);
            Eigen::MatrixXd grad_M = -Eigen::MatrixXd::Identity(dim_b, dim_b);

            for (int idx : elite_indices)
            {
                Eigen::MatrixXd eps_block = eps_samples[idx].middleCols(start_col, num_cols);
                Eigen::VectorXd eps_vec = Eigen::Map<const Eigen::VectorXd>(eps_block.data(), dim_b);
                Eigen::VectorXd s_ib = lu_A.solve(eps_vec);

                grad_delta.noalias() += w_i * s_ib;
                grad_M.noalias() += w_i * (s_ib * s_ib.transpose());
            }

            Eigen::VectorXd delta_m = lr_mean * (A_b * grad_delta);
            control_input.middleCols(start_col, num_cols) +=
                Eigen::Map<const Eigen::MatrixXd>(delta_m.data(), u_dim, num_cols);

            Eigen::MatrixXd scaled_grad_M = 0.5 * lr_cov * grad_M;
            A_blocks[b] = A_b * scaled_grad_M.exp();
        }
    }
};
