#include "build_model.hpp"

FFN<NegativeLogLikelihood<>, RandomInitialization>
BuildModel(const std::vector<LayerSpec>& layers)
{
    FFN<NegativeLogLikelihood<>, RandomInitialization> model;

    for (const auto& L : layers)
    {
        if (L.name == "linear")
        {
            if (L.params.size() != 2)
                throw std::runtime_error("linear layer requires 2 params");

            model.Add(new Linear<>(L.params[0], L.params[1]));
        }
        else if (L.name == "relu")
        {
            model.Add(new ReLULayer<>());
        }
        else if (L.name == "sigmoid")
        {
            model.Add(new SigmoidLayer<>());
        }
        else if (L.name == "tanh")
        {
            model.Add(new TanHLayer<>());
        }
        else if (L.name == "logsoftmax")
        {
            model.Add(new LogSoftMax<>());
        }
        else {
            throw std::runtime_error("Unknown layer type: " + L.name);
        }
    }

    return model;
}
