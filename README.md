ANN CLI Builder for mlpack (v3.4.2)
====================================

A lightweight, extensible command-line tool to define, build, train, and evaluate neural networks using mlpack.

-----------------------------------------------------
Overview
-----------------------------------------------------

This project implements a simple, text-based configuration system and CLI that allows users to:

• Define neural network architectures in a human-readable .txt format  
• Parse these architectures into layer specifications  
• Automatically construct mlpack ANN models  
• Train these models on CSV datasets  
• Save trained models  
• Predict using trained models via CLI  

This project is part of a future GSoC contribution idea for mlpack.

-----------------------------------------------------
Folder Structure
-----------------------------------------------------


gsoc_proj

├── cfg

│ └── model.txt

├── data

│ ├── train.csv

│ └── label.csv

├── src

│ ├── arch_parser.hpp


│ ├── arch_parser.cpp


│ ├── build_model.hpp


│ ├── build_model.cpp


│ ├── trainer.hpp


│ ├── trainer.cpp


│ ├── ann_cli.cpp


│ └── train_config.hpp


├── test


│ ├── test_parse.cpp


│ ├── test_build.cpp


│ └── test_train.cpp


└── CMakeLists.txt




-----------------------------------------------------
Architecture Config File Format
-----------------------------------------------------

The ANN architecture is defined using a simple line-based syntax.

Example (cfg/model.txt):

# Tiny classifier
linear - 2 , 8

relu

linear - 8 , 2

logsoftmax

Supported layer keywords:


linear <in> <out>

relu

sigmoid

tanh

logsoftmax

-----------------------------------------------------
Build Instructions
-----------------------------------------------------

mkdir build

cd build

cmake ..

make -j4


The built CLI binary appears as:

build/ann_cli

-----------------------------------------------------
Training a Model
-----------------------------------------------------

Run:

./ann_cli --mode train \
  --cfg ../cfg/model.txt \
  --train ../data/train.csv \
  --labels ../data/label.csv \
  --epochs 30 \
  --batchsize 16 \
  --stepsize 0.01 \
  --save trained.bin


Parameter meaning:
--mode train         : training mode

--cfg                : architecture file

--train              : CSV input features

--labels             : CSV labels (single column or row)

--epochs             : training epochs

--batchsize          : mini-batch size

--stepsize           : learning rate

--save               : output trained model

-----------------------------------------------------
Prediction Mode
-----------------------------------------------------

./ann_cli --mode predict \
  --load trained.bin \
  --input ../data/test.csv \
  --output preds.csv

-----------------------------------------------------
Tests
-----------------------------------------------------

Parser test:
./test_parse

Model builder test:
./test_build

Training test:
./test_train

-----------------------------------------------------
Design Components
-----------------------------------------------------

1) Architecture Parser

    - Reads cfg/model.txt

   - Supports comments (#)

   - Produces vector<LayerSpec>

3) Model Builder

   - Uses mlpack 3.4.2 ANN API

   - Adds layers using model.Add(new LayerType<>);

5) Trainer

   - Loads CSV training data

   - Converts labels → Row<size_t>

   - Uses SGD or Adam optimizers

   - Saves model using data::Save

7) CLI (ann_cli.cpp)

    - Dispatches based on mode: train or predict

   - Reads configurations from command line

-----------------------------------------------------
Why mlpack v3.4.2?
-----------------------------------------------------

mlpack 4.x removed the ANN module (FFN, Linear, ReLU, etc.)
ANN code exists only in mlpack 3.x.

Because this project uses ANN functionality, mlpack v3.4.2 is the correct version.

-----------------------------------------------------
Future Extensions
-----------------------------------------------------

• Add YAML config support  

• Add convolution, dropout, batchnorm layers  

• Add metrics such as accuracy, F1-score  

• Allow ONNX export  

• Evaluate against mlpack maintainers' upcoming refactor  

-----------------------------------------------------
License
-----------------------------------------------------

MIT License (or any preferred license)

-----------------------------------------------------
Contributing
-----------------------------------------------------

Pull requests and issues are welcome.
