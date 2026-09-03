// context.h — Small dataset-file + helper utilities shared by the trainers.
//
// The .bin dataset format used everywhere in this repo:
//   uint32 n      – number of rows
//   uint32 feat   – feature count per row (784 for the byte models)
//   n*feat float32 – row-major pixel/byte features, each in 0..255
//   n float32     – class labels (a class id)
//
// See AGENTS.md ("Data pipeline") for how these are generated from Hugging Face.

#pragma once
#include <string>
#include <vector>

namespace vk {
class Context;
}

namespace ctx {

// Load a .bin dataset (see header comment for the format) into X (n*feat floats)
// and y (n labels). On failure returns false and, if `err` is non-null, fills it
// with a human-readable message.
bool loadBin(const std::string& path, int& n, int& feat, std::vector<float>& X, std::vector<float>& y, std::string* err);
// Inverse of loadBin(): write X/y to `path` in the same binary format.
bool saveBin(const std::string& path, int n, int feat, const std::vector<float>& X, const std::vector<float>& y, std::string* err);

// Fraction of rows where pred[i] == y[i] (0.0 if sizes differ or empty).
float accuracy(const std::vector<float>& pred, const std::vector<float>& y);

// Split `s` on `delim`, dropping empty tokens (used for CLI / config parsing).
std::vector<std::string> split(const std::string& s, char delim);

}
