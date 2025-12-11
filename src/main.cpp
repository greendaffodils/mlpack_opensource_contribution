#include "arch_parser.hpp"
#include "train_config.hpp"
#include "trainer.hpp"

int main()
{
    // Load architecture
    auto arch = ParseArchitectureFile("cfg/model.txt");

    // Create training configuration
    TrainConfig cfg;
    cfg.trainFile = "data/train.csv";
    cfg.labelsFile = "data/label.csv";
    cfg.epochs = 20;
    cfg.stepSize = 0.01;
    cfg.batchSize = 32;
    cfg.optimizer = "adam";
    cfg.saveModel = "trained.bin";

    // Train
    TrainModel(arch, cfg);

    return 0;
}
