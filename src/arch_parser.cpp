#include "arch_parser.hpp"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <iostream>

// trim helpers
static inline std::string ltrim(const std::string& s)
{
    size_t start = s.find_first_not_of(" \t\r\n");
    return (start == std::string::npos) ? std::string() : s.substr(start);
}
static inline std::string rtrim(const std::string& s)
{
    size_t end = s.find_last_not_of(" \t\r\n");
    return (end == std::string::npos) ? std::string() : s.substr(0, end+1);
}
static inline std::string trim(const std::string& s)
{
    return rtrim(ltrim(s));
}

// lowercase helper
static inline std::string toLower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::tolower(c); });
    return s;
}

std::vector<LayerSpec> ParseArchitectureFile(const std::string& path)
{
    std::ifstream ifs(path);
    if (!ifs.is_open())
        throw std::runtime_error("Could not open architecture file: " + path);

    std::vector<LayerSpec> specs;
    std::string line;
    size_t lineno = 0;

    while (std::getline(ifs, line))
    {
        ++lineno;
        // remove comments starting with '#'
        auto commentPos = line.find('#');
        if (commentPos != std::string::npos)
            line = line.substr(0, commentPos);

        line = trim(line);
        if (line.empty()) continue;

        std::istringstream iss(line);
        std::string token;
        if (!(iss >> token)) continue;

        token = toLower(token);
        LayerSpec spec;
        spec.name = token;

        // parse remaining tokens as doubles (numbers)
        std::string numtok;
        while (iss >> numtok)
        {
            try
            {
                double v = std::stod(numtok);
                spec.params.push_back(v);
            }
            catch (const std::exception& e)
            {
                // not a number -> error with line number
                throw std::runtime_error("Parse error in " + path + " at line " +
                                         std::to_string(lineno) + ": invalid token '" + numtok + "'");
            }
        }

        specs.push_back(std::move(spec));
    }

    return specs;
}

