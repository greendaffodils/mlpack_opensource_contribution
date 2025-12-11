#include "arch_parser.hpp"

#include <iostream>

int main(int argc, char** argv)
{
    const char* path = "cfg/model.txt";
    if (argc > 1) path = argv[1];

    try
    {
        auto specs = ParseArchitectureFile(path);
        std::cout << "Parsed " << specs.size() << " layers from " << path << ":\n";
        for (size_t i = 0; i < specs.size(); ++i)
        {
            std::cout << i << ": " << specs[i].name;
            if (!specs[i].params.empty())
            {
                std::cout << " [";
                for (size_t j = 0; j < specs[i].params.size(); ++j)
                {
                    if (j) std::cout << ", ";
                    std::cout << specs[i].params[j];
                }
                std::cout << "]";
            }
            std::cout << "\n";
        }
    }
    catch (const std::exception& ex)
    {
        std::cerr << "ERROR: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
