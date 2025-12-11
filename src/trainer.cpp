#include "trainer.hpp"
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>

using namespace mlpack;
using namespace mlpack::ann;

void TrainModel(
    const std::vector<LayerSpec>& arch,
    const TrainConfig& config)
{
    // ---- Build Model ----
    auto model = BuildModel(arch);

    // ---- Load Data ----
    arma::mat X;
    arma::mat y;

    if (!data::Load(config.trainFile, X, true))
        throw std::runtime_error("Cannot load training file: " + config.trainFile);

    if (!data::Load(config.labelsFile, y, true))
        throw std::runtime_error("Cannot load label file: " + config.labelsFile);

    // ---- Debug: Print shapes ----
    std::cout << "[DEBUG] X shape = " << X.n_rows << "x" << X.n_cols << "\n";
    std::cout << "[DEBUG] y shape = " << y.n_rows << "x" << y.n_cols << "\n";



    // ---- Convert to Row<size_t> ----
if (y.n_rows != 1)
    y = y.t();

// now convert values to integers but keep arma::mat type
y.transform([](double v){ return std::round(v); });

arma::mat labels = y; 

    size_t maxIterations = config.epochs;
    ens::StandardSGD optimizer(config.stepSize,
                               config.batchSize,
                               maxIterations);

    model.Train(X, labels,optimizer);

    std::cout << "[DEBUG] Training completed.\n";

    // ---- Save Model ----
    if (!data::Save(config.saveModel, "model", model))
        throw std::runtime_error("Failed to save model!");

    std::cout << "Training complete. Saved model to " << config.saveModel << "\n";
}

// void TrainModel(const std::vector<LayerSpec>& arch,
//                 const TrainConfig& config)
// {
//     auto model = BuildModel(arch);

//     arma::mat X, yraw;
//     data::Load(config.trainFile, X, true);
//     data::Load(config.labelsFile, yraw, true);

//     if (yraw.n_rows != 1)
//         yraw = yraw.t();

//     // Convert labels to (1 × N) integer matrix
//     arma::mat labels = yraw;
//     labels.transform([](double v) { return std::round(v); });

//     // Optimizer
//     size_t maxIterations = config.epochs;
//     ens::StandardSGD optimizer(config.stepSize,
//                                config.batchSize,
//                                maxIterations);

//     std::cout << "[DEBUG] Starting training...\n";
//     model.Train(X, labels, optimizer);

//     data::Save(config.saveModel, "model", model);
//     std::cout << "Training complete.\n";
// }
