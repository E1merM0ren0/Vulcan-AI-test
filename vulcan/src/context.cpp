#include "context.h"
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace ctx {

bool loadBin(const std::string& path, int& n, int& feat,
             std::vector<float>& X, std::vector<float>& y, std::string* err) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        if (err) *err = "cannot open " + path;
        return false;
    }
    uint32_t n32 = 0, feat32 = 0;
    f.read((char*)&n32, 4);
    f.read((char*)&feat32, 4);
    n = (int)n32;
    feat = (int)feat32;
    X.resize((size_t)n * feat);
    y.resize((size_t)n);
    f.read((char*)X.data(), (std::streamsize)(X.size() * sizeof(float)));
    f.read((char*)y.data(), (std::streamsize)(y.size() * sizeof(float)));
    f.close();
    if (err) err->clear();
    return true;
}

bool saveBin(const std::string& path, int n, int feat,
             const std::vector<float>& X, const std::vector<float>& y, std::string* err) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) {
        if (err) *err = "cannot open " + path + " for writing";
        return false;
    }
    uint32_t n32 = (uint32_t)n, feat32 = (uint32_t)feat;
    f.write((const char*)&n32, 4);
    f.write((const char*)&feat32, 4);
    f.write((const char*)X.data(), (std::streamsize)(X.size() * sizeof(float)));
    f.write((const char*)y.data(), (std::streamsize)(y.size() * sizeof(float)));
    f.close();
    if (err) err->clear();
    return true;
}

float accuracy(const std::vector<float>& pred, const std::vector<float>& y) {
    if (pred.empty() || pred.size() != y.size()) return 0.0f;
    int hit = 0;
    for (size_t i = 0; i < y.size(); i++) {
        if (pred[i] == y[i]) hit++;
    }
    return (float)hit / (float)y.size();
}

std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> parts;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        if (!item.empty()) parts.push_back(item);
    }
    return parts;
}

}
