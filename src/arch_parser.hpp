#ifndef ARCH_PARSER_HPP
#define ARCH_PARSER_HPP

#include <string>
#include <vector>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>

// Parsed line from architecture file.
struct LayerSpec
{
    std::string name;           // e.g. "linear", "relu"
    std::vector<double> params; // e.g. {4,16}
};

// Parse "cfg/model.txt"
std::vector<LayerSpec> ParseArchitectureFile(const std::string& path);

// Build an mlpack FFN model from parsed architecture.
mlpack::ann::FFN<
    mlpack::ann::NegativeLogLikelihood<>,
    mlpack::ann::RandomInitialization
> BuildModel(const std::vector<LayerSpec>& arch);

#endif
