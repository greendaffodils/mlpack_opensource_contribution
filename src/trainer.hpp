#pragma once

#include "arch_parser.hpp"
#include "build_model.hpp"
#include "train_config.hpp"

#include <mlpack/methods/ann/ffn.hpp>
#include <ensmallen.hpp>

void TrainModel(
    const std::vector<LayerSpec>& arch,
    const TrainConfig& config);
