#include <iostream>
#include "arch_parser.hpp"

#include "build_model.hpp"

int main()
{
    auto layers = ParseArchitectureFile("cfg/model.txt");

    auto model = BuildModel(layers);

    std::cout << "Model built successfully!\n";
    std::cout << "Layer count: " << model.Model().size() << "\n";
    

    return 0;
}
