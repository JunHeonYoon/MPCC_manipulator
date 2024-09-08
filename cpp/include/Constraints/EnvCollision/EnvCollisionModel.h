#ifndef MPCC_ENV_COLLISION_H
#define MPCC_ENV_COLLISION_H

#include <config.h>
#include <iostream>
#include <vector>
#include <torch/script.h>
#include <torch/torch.h>
#include <memory>
#include <filesystem>
#include <cassert>

namespace mpcc
{
    class EnvCollNNmodel
    {
        public:
            EnvCollNNmodel();
            EnvCollNNmodel(const std::string& file_path);
            ~EnvCollNNmodel();
            void setNeuralNetwork(std::vector<int> voxel_size, int joint_size);
            std::pair<Eigen::VectorXd, Eigen::MatrixXd> forward(std::vector<float> voxel_data, std::vector<float> joint_data); 
        private:
            bool loadNetwork();

            std::string file_path_;
            bool is_loaded;
            torch::jit::script::Module model_;
            torch::Device device_;
            std::vector<int> voxel_size_;
            int joint_size_;
            // std::vector<float> voxel_data_; // size: [batch size, channel size(1), voxel size(36, 36, 36)]
            // std::vector<float> joint_data_; // size: [batch size, dof(7)]
    };
}

#endif