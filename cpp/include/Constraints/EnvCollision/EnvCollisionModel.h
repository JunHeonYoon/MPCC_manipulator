#ifndef MPCC_ENV_COLLISION_H
#define MPCC_ENV_COLLISION_H

#include <config.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <Eigen/Dense>

namespace mpcc
{
    class EnvCollNNmodel
    {
        struct MLP
        {
            ~MLP() { std::cout << "MLP terminate" << std::endl; }
            std::vector<Eigen::MatrixXd> weight;
            std::vector<Eigen::VectorXd> bias;
            std::vector<Eigen::VectorXd> hidden;
            std::vector<Eigen::MatrixXd> hidden_derivative;

            std::vector<std::string> w_path;
            std::vector<std::string> b_path;

            std::vector<std::ifstream> weight_files;
            std::vector<std::ifstream> bias_files;

            int n_input;
            int n_output;
            Eigen::VectorXd n_hidden;
            int n_layer;

            Eigen::VectorXd input;
            Eigen::VectorXd output;
            Eigen::MatrixXd output_derivative;

            bool is_nerf;
            Eigen::VectorXd input_nerf;

            bool loadweightfile_verbose = false;
            bool loadbiasfile_verbose = false;
        };
        
        public:
            EnvCollNNmodel();
            EnvCollNNmodel(const std::string & file_path);
            ~EnvCollNNmodel();
            void setNeuralNetwork(int n_input, int n_output, Eigen::VectorXd n_hidden, bool is_nerf);
            std::pair<Eigen::VectorXd, Eigen::MatrixXd> calculateMlpOutput(Eigen::VectorXd input, bool time_verbose);
        private:
            std::string file_path_;
            MLP mlp_;

            void readWeightFile(int weight_num);
            void readBiasFile(int bias_num);
            void loadNetwork();
            void initializeNetwork(int n_input, int n_output, Eigen::VectorXd n_hidden, bool is_nerf);

            double ReLU(double input)
            {
                return std::max(0.0, input);
            }
            double ReLU_derivative(double input)
            {
                return (input > 0)? 1.0 : 0.0;
            }
    };
}

#endif