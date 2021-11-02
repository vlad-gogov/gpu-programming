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