#pragma once

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

std::string readFile(const std::string& path);

template <typename T>
void randomMatrix(std::vector<T>& arr) {
    std::random_device rd;
    std::mt19937 mersenne(rd());
    std::uniform_real_distribution<> urd(-1.0, 1.0);
    size_t size = arr.size();
    for (T& el : arr)
        el = urd(mersenne);
}

bool checkMatrix(const std::vector<float>& a, const std::vector<float>& b);