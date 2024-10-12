#include "Constraints/EnvCollision/EnvCollisionModel.h"

namespace mpc
{

EnvCollNNmodel::EnvCollNNmodel()
: device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU)
{
    EnvCollNNmodel(pkg_path + "NNmodel/env_collision.pt");
}

EnvCollNNmodel::EnvCollNNmodel(const std::string& file_path)
: file_path_(file_path), device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU)
{
    is_loaded = loadNetwork();
}

EnvCollNNmodel::~EnvCollNNmodel()
{
    std::cout<<"Env NN model terminate" <<std::endl;
}

bool EnvCollNNmodel::loadNetwork()
{
    if(!std::filesystem::exists(file_path_))
    {
        std::cerr << "Env Collision Model is not exist in " << file_path_ << std::endl;
        return false;
    }
    try 
    {
        model_ = torch::jit::load(file_path_);
        // device_ = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        model_.to(device_);
    } 
    catch (const c10::Error& e) 
    {
        std::cerr << "Error loading the Env Collision Model" << std::endl;
    }
    return false;
}

void EnvCollNNmodel::setNeuralNetwork(std::vector<int> voxel_size, int joint_size)
{
    assert(voxel_size.size() == 3);
    voxel_size_ = voxel_size;
    joint_size_ = joint_size;
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> EnvCollNNmodel::forward(std::vector<float> voxel_data, std::vector<float> joint_data)
{    
    if(is_loaded = false)
    {
        std::cerr << " Env Collision Model is not loaded" << std::endl;
        return std::pair<Eigen::VectorXd, Eigen::MatrixXd>();
    }
    if(voxel_size_.size() == 0)
    {
        std::cerr << "Env Collision Model does not setNeuralNetwork" << std::endl;
        return std::pair<Eigen::VectorXd, Eigen::MatrixXd>();
    }

    int batch_size = int(joint_data.size() / joint_size_);
    if(batch_size != int(voxel_data.size() / (voxel_size_[0]*voxel_size_[1]*voxel_size_[2])))
    {
        std::cerr << "Batch size for voxel data(" << int(voxel_data.size() / (voxel_size_[0]*voxel_size_[1]*voxel_size_[2])) 
                  << ") and joint data(" << batch_size << ") must be same" << std::endl;
        return std::pair<Eigen::VectorXd, Eigen::MatrixXd>();
    }
    try
    {
        at::Tensor x_q = torch::from_blob(joint_data.data(), {batch_size, joint_size_}, torch::kFloat32).clone().to(device_);
        at::Tensor x_occ = torch::from_blob(voxel_data.data(), {batch_size, 1, voxel_size_[0], voxel_size_[1], voxel_size_[2]}, torch::kFloat32).clone().to(device_);
        // std::cout << x_q << std::endl;
        // std::cout << x_occ << std::endl;

        // Set requires_grad_ to true for x_q to enable gradient computation
        x_q.set_requires_grad(true);

        // Forward pass through the model
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(x_q);
        inputs.push_back(x_occ);

        // for(auto& val : inputs) std::cout << inputs <<std::endl;

        torch::Tensor output = model_.forward(inputs).toTensor().cpu();

        std::cout << output<<std::endl;

        // Convert the output tensor to a vector for easy handling
        Eigen::VectorXf env_min_dist_pred(output.numel());
        std::memcpy(env_min_dist_pred.data(), output.data_ptr<float>(), output.numel() * sizeof(float));

        // Calculate the Jacobian
        int output_size = output.size(1);
        Eigen::MatrixXf jacobian_pred(output_size, x_q.size(1));
        for (int i=0; i<output_size; ++i) 
        {
            at::Tensor grad_output = torch::zeros_like(output);
            grad_output.index_put_({torch::indexing::Slice(), i}, 1);

            // Compute gradients with respect to x_q
            at::Tensor grad_x_q = torch::autograd::grad({output}, {x_q}, {grad_output})[0];
            grad_x_q = grad_x_q.cpu();

            // Convert gradient to std::vector and store in jacobian_pred
            std::memcpy(jacobian_pred.data() + i * x_q.size(1), grad_x_q.data_ptr<float>(), x_q.size(1) * sizeof(float));
        }

        return std::make_pair(env_min_dist_pred.cast<double>(), jacobian_pred.cast<double>());
    }
    catch(const std::exception& e)
    {
        std::cerr << "Exception in modelForward: " << e.what();
        return std::make_pair(Eigen::VectorXd(), Eigen::MatrixXd());
    }
}
}