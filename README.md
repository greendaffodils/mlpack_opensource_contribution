📦 ANN CLI Builder for mlpack (v3.4.2)

A lightweight, extensible command-line tool to define, build, train, and evaluate neural networks using mlpack.

🔍 Overview

This project implements a simple, text-based configuration system and CLI that allows users to:

Define neural network architectures in a human-readable .txt format

Parse these architectures into layer specifications

Automatically construct mlpack ANN models from those specifications

Train these models on provided datasets

Save trained models for later inference

Run inference (predictions) from the command line

This tool is inspired by discussions with mlpack maintainers and mirrors the direction of “architecture parsing + ANN CLI” needed for future GSoC contributions.

🧠 Motivation

mlpack is a fast and flexible machine-learning library in C++.
However:

ANN model construction requires manual C++ coding.

No built-in CLI exists for building arbitrary neural networks from a config file.

ANN tutorials require writing full code even for simple experiments.

This project solves these issues by introducing:

✔ A tiny architecture language
✔ A parser
✔ A model builder
✔ A training CLI

Together, they form the foundation for a future generalized ANN CLI within mlpack.

📁 Project Structure
gsoc_proj/
│
├── cfg/
│   └── model.txt            # Architecture file
│
├── data/
│   ├── train.csv            # Training inputs (features)
│   └── label.csv            # Labels (integer classes)
│
├── src/
│   ├── arch_parser.hpp
│   ├── arch_parser.cpp      # Parses architecture file
│   ├── build_model.hpp
│   ├── build_model.cpp      # Builds mlpack ANN model
│   ├── trainer.hpp
│   ├── trainer.cpp          # Handles training logic
│   ├── ann_cli.cpp          # Final CLI executable
│   └── train_config.hpp     # Holds user configuration
│
├── test/
│   ├── test_parse.cpp       # Unit test for parser
│   ├── test_build.cpp       # Unit test for model builder
│   └── test_train.cpp       # Lightweight training test
│
└── CMakeLists.txt           # Modern CMake project setup

🏗 Architecture File Format

The ANN architecture is defined using a simple, line-based language:

Example — cfg/model.txt:

# Tiny classifier example
linear 2 8
relu
linear 8 2
logsoftmax


Supported layers:

Layer	Syntax	Notes
Linear	linear <in> <out>	Fully connected
ReLU	relu	Activation
Sigmoid	sigmoid	Activation
Tanh	tanh	Activation
LogSoftmax	logsoftmax	Output layer for classification
⚙️ Build Instructions
1. Configure CMake project:
mkdir build
cd build
cmake ..
make -j4

Output binary:
build/ann_cli

🚀 Usage
Train a model
./ann_cli --mode train \
  --cfg ../cfg/model.txt \
  --train ../data/train.csv \
  --labels ../data/label.csv \
  --epochs 30 \
  --batchsize 16 \
  --stepsize 0.01 \
  --save trained.bin

Options
Flag	Meaning
--mode train	Training mode
--cfg	Path to architecture file
--train	Training data (CSV)
--labels	Label file (CSV, single row)
--epochs	Number of passes
--batchsize	Minibatch size
--stepsize	Learning rate
--save	Output model name
Run inference
./ann_cli --mode predict \
  --load trained.bin \
  --input ../data/test.csv \
  --output preds.csv

🧪 Testing
Parser test:
./test_parse

Model building test:
./test_build

Training logic test:
./test_train

🧩 Design Highlights
1. Architecture Parser

Minimal tokenizer

Ignores comments and whitespace

Validates parameters

Produces a std::vector<LayerSpec>

2. Model Builder

Uses mlpack v3.4.2 ANN API:

FFN<NegativeLogLikelihood<>, RandomInitialization> model;
model.Add(new Linear<>(...));
model.Add(new ReLULayer<>());
model.Add(new LogSoftMax<>());

3. Trainer

Handles:

Dataset loading

Label conversion → Row<size_t>

Optimizer selection (SGD, Adam)

Training loop

Saving model via mlpack data::Save

4. ANN CLI

Command-line utility wrapping everything:

Train

Predict

Load/Save model

🔧 Why mlpack v3.4.2?

mlpack 4.x removed:

ANN layers in C++

FFN class

The entire ANN module

Thus, ANN code only exists in 3.x versions.

This project targets ANN support, therefore mlpack v3.4.2 is the correct and stable base.

⭐ Future Work

Add YAML config loader

Add more layers (Dropout, BatchNorm, Convolution)

Implement evaluation metrics (accuracy, F1 score)

Export models to ONNX

Integrate callbacks from ensmallen

Submit upstream to mlpack as GSoC proposal foundation

🤝 Contributing

Pull requests are welcome!

You can open issues for:

New layer support

CLI improvements

Documentation fixes

Performance optimizations

📜 License

MIT License (or whatever you choose).
