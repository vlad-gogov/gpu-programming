#include "Utils.h"

#include <fstream>
#include <iostream>

std::string readFile(const std::string& path) {
    std::ifstream file(path);
    std::string content, str;
    while (std::getline(file, str)) {
        content += str;
        content.push_back('\n');
    }
    return content;
}

bool checkMatrix(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); i++)
        if (std::abs(a[i] - b[i]) >= 1e-2f)
            return false;
    return true;
}