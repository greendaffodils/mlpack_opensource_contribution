#pragma once
#include "arch_parser.hpp"

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>

#include <mlpack/methods/ann/layer/linear.hpp>
#include <mlpack/methods/ann/layer/c_relu.hpp>
#include <mlpack/methods/ann/layer/hard_tanh.hpp>
#include <mlpack/methods/ann/activation_functions/hard_sigmoid_function.hpp>
#include <mlpack/methods/ann/layer/log_softmax.hpp>

using namespace mlpack;
using namespace mlpack::ann;

FFN<NegativeLogLikelihood<>, RandomInitialization>
BuildModel(const std::vector<LayerSpec>& layers);
