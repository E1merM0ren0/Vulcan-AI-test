#pragma once
#include <string>
#include <vector>

namespace vk {
class Context;
}

namespace ctx {

bool loadBin(const std::string& path, int& n, int& feat, std::vector<float>& X, std::vector<float>& y, std::string* err);
bool saveBin(const std::string& path, int n, int feat, const std::vector<float>& X, const std::vector<float>& y, std::string* err);

float accuracy(const std::vector<float>& pred, const std::vector<float>& y);

std::vector<std::string> split(const std::string& s, char delim);

}
