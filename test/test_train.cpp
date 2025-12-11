#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/layer/linear.hpp>
#include <mlpack/methods/ann/layer/log_softmax.hpp>
#include <mlpack/methods/ann/layer/c_relu.hpp>
#include <ensmallen.hpp>
#include "../src/trainer.hpp"
#include "../src/build_model.hpp"
#include "../src/arch_parser.hpp"

using namespace mlpack;
using namespace mlpack::ann;

int main()
{
    // Step 1: Parse architecture
    auto arch = ParseArchitectureFile("/mnt/c/gsoc_proj/cfg/model.txt");
    std::cout << "Architecture parsed OK\n";

    // Step 2: Build model
    auto model = BuildModel(arch);
    std::cout << "Model built OK\n";

    // Step 3: Load data
    arma::mat X;
    arma::mat y;

    data::Load("/mnt/c/gsoc_proj//data/train.csv", X, true);
    data::Load("/mnt/c/gsoc_proj//data/label.csv", y, true);

    std::cout << "X shape = " << X.n_rows << "x" << X.n_cols << "\n";
    std::cout << "y shape = " << y.n_rows << "x" << y.n_cols << "\n";
    




    y.transform([](double v){ return std::round(v); });

    arma::mat labels = y; 

    // Convert y to Row<size_t>
    // arma::Row<size_t> labels(y.n_elem);
    // for (size_t i = 0; i < y.n_elem; i++){
    //     labels[i] = static_cast<size_t>(y[i]);
    // }
    // Step 4: Train
        std::cout << "Labels converted: ";
    labels.print();

std::cout << "Min label: " << labels.min() << "\n";
std::cout << "Max label: " << labels.max() << "\n";

    ens::Adam optimizer(0.01, 32, 0.9, 0.999, 1e-8, 20);
    std::cout << "Training model...\n"; 
    model.Train(X, labels, optimizer);

    std::cout << "Training finished OK!\n";

    return 0;
}
