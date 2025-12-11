#pragma once
#include <string>

struct TrainConfig
{
    std::string trainFile = "";
    std::string labelsFile = "";

    size_t epochs = 20;
    double stepSize = 0.01;
    size_t batchSize = 32;

    std::string optimizer = "sgd";   // "sgd" or "adam"

    std::string saveModel = "model.bin";
};
