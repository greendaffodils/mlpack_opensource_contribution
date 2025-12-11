// cli.cpp
#include <iostream>
#include <string>
#include <vector>

#include <boost/program_options.hpp>
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>

#include "arch_parser.hpp"
#include "build_model.hpp"
#include "trainer.hpp"

namespace po = boost::program_options;
using namespace mlpack;
using namespace mlpack::ann;

static void PrintHeader()
{
  std::cout << "ann_cli — train / predict / evaluate (mlpack 3.x)\n";
  std::cout << "------------------------------------------------\n";
}

int main(int argc, char** argv)
{
  PrintHeader();

  // Common options
  po::options_description common("Common options");
  common.add_options()
    ("help,h", "Print this help message")
    ("mode,m", po::value<std::string>()->default_value("train"),
     "Mode: train | predict | evaluate")
    ("cfg,c", po::value<std::string>()->default_value("cfg/model.txt"),
     "Architecture file (text)")
    ("train", po::value<std::string>()->default_value("data/train.csv"),
     "Training file (csv)")
    ("labels", po::value<std::string>()->default_value("data/label.csv"),
     "Labels file (csv)")
    ("test", po::value<std::string>()->default_value("data/test.csv"),
     "Test/predict file (csv)")
    ("test_labels", po::value<std::string>()->default_value("data/test_labels.csv"),
     "Test labels for evaluation (csv)")
    ("save", po::value<std::string>()->default_value("trained.bin"),
     "File to save model")
    ("load", po::value<std::string>()->default_value("trained.bin"),
     "Model file to load")
    ("epochs", po::value<size_t>()->default_value(20),
     "Epochs")
    ("batchsize", po::value<size_t>()->default_value(32),
     "Batch size")
    ("stepsize", po::value<double>()->default_value(0.01),
     "Step size")
    ("optimizer", po::value<std::string>()->default_value("sgd"),
     "Optimizer: sgd (mlpack 3.x supports StandardSGD for FFN)")
    ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, common), vm);
  po::notify(vm);

  if (vm.count("help"))
  {
    std::cout << common << std::endl;
    return 0;
  }

  const std::string mode = vm["mode"].as<std::string>();
  const std::string cfgFile = vm["cfg"].as<std::string>();
  const std::string trainFile = vm["train"].as<std::string>();
  const std::string labelsFile = vm["labels"].as<std::string>();
  const std::string testFile = vm["test"].as<std::string>();
  const std::string testLabelsFile = vm["test_labels"].as<std::string>();
  const std::string saveFile = vm["save"].as<std::string>();
  const std::string loadFile = vm["load"].as<std::string>();
  const size_t epochs = vm["epochs"].as<size_t>();
  const size_t batchSize = vm["batchsize"].as<size_t>();
  const double stepSize = vm["stepsize"].as<double>();
  const std::string optimizer = vm["optimizer"].as<std::string>();

  try
  {
    if (mode == "train")
    {
      // Parse architecture
      std::cout << "[CLI] Parsing architecture: " << cfgFile << std::endl;
      auto arch = ParseArchitectureFile(cfgFile);

      // Build model
      std::cout << "[CLI] Building model..." << std::endl;
      auto model = BuildModel(arch);

      // Build config
      TrainConfig cfg;
      cfg.trainFile = trainFile;
      cfg.labelsFile = labelsFile;
      cfg.epochs = epochs;
      cfg.batchSize = batchSize;
      cfg.stepSize = stepSize;
      cfg.optimizer = optimizer;
      cfg.saveModel = saveFile;

      // Train (uses your TrainModel implementation)
      TrainModel(arch, cfg);

      std::cout << "[CLI] Training finished. Model saved to " << saveFile << std::endl;
    }
    else if (mode == "predict")
    {
      // Load model
      std::cout << "[CLI] Loading model: " << loadFile << std::endl;
      FFN<NegativeLogLikelihood<arma::mat, arma::mat>, RandomInitialization> model;
      data::Load(loadFile, "model", model);

      // Load test data
      arma::mat X;
      data::Load(testFile, X, true);

      // Predict: output matrix: classes x samples
      arma::mat output;
      model.Predict(X, output);

      // Convert to predicted labels (1-based for mlpack 3.x)
      arma::Row<size_t> preds(output.n_cols);
      for (size_t i = 0; i < output.n_cols; ++i)
      {
        // index_max exists on column vectors / subviews
        arma::uword idx = output.col(i).index_max();
        preds(i) = static_cast<size_t>(idx) + 1; // keep 1-based labeling
      }

      // Save predictions to CSV (1xN)
      data :: Save("preds.csv", preds, true);
      std::cout << "[CLI] Predictions saved to preds.csv\n";
    }
    else if (mode == "evaluate")
    {
      // Load model
      std::cout << "[CLI] Loading model: " << loadFile << std::endl;
      FFN<NegativeLogLikelihood<arma::mat, arma::mat>, RandomInitialization> model;
      data::Load(loadFile, "model", model);

      // Load test data and labels
      arma::mat X, y;
      data::Load(testFile, X, true);
      data::Load(testLabelsFile, y, true);

      if (y.n_rows != 1) y = y.t(); // ensure 1xN
      // convert to 1-based labels to match mlpack expectation
      y.transform([](double v){ return std::round(v) + 1.0; });
      arma::Row<size_t> trueLabels(y.n_cols);
      for (size_t i = 0; i < y.n_cols; ++i) trueLabels(i) = static_cast<size_t>(y(0, i));

      // Predict
      arma::mat output;
      model.Predict(X, output);

      // Map predictions to 1-based class ids
      arma::Row<size_t> preds(output.n_cols);
      for (size_t i = 0; i < output.n_cols; ++i)
      {
        arma::uword idx = output.col(i).index_max();
        preds(i) = static_cast<size_t>(idx) + 1;
      }

      // Compute accuracy
      size_t correct = 0;
      for (size_t i = 0; i < preds.n_elem; ++i)
        if (preds(i) == trueLabels(i)) ++correct;
      double acc = double(correct) / double(preds.n_elem);

      std::cout << "[CLI] Accuracy: " << (acc * 100.0) << "% (" << correct << "/"
                << preds.n_elem << ")\n";
    }
    else
    {
      std::cerr << "Unknown mode: " << mode << "\n";
      return 1;
    }
  }
  catch (std::exception& ex)
  {
    std::cerr << "[ERROR] " << ex.what() << std::endl;
    return 1;
  }

  return 0;
}
