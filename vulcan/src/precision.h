// precision.h — Selectable compute precision for the training graphs (--fp flag).
//
// W1.58 weight quantization / A8 absmax activation quantization are bitnet's
// own schemes and stay FIXED. This header adds an orthogonal knob: the storage
// precision of the intermediate FP32 tensors in the compute graph. It provides
// CPU-side mini-float encode/decode (FP8 E4M3, FP4 E2M1) used by:
//   - the CPU oracle (quantizeTo / CPUModel::q) to mirror the GPU at the same
//     graph boundaries, and
//   - as the bit-exact reference the GLSL shader (shaders/quantize_fp.comp) is
//     written to match, so loss_diff == grad_diff == 0 under fp8/fp4.
//
// The encode functions are deliberately spelled out bit-by-bit (no third-party
// float-conversion intrinsics) so the rounding is identical on CPU and GPU.

#pragma once
#include <cmath>
#include <cstdint>

// Which precision the whole compute graph runs at. FP32 is the default and
// leaves the graph untouched (identity quantization).
enum class Precision {
    FP32 = 0,
    FP8 = 1,
    FP4 = 2,
};

inline const char* precisionName(Precision p) {
    switch (p) {
        case Precision::FP32: return "fp32";
        case Precision::FP8:  return "fp8";
        case Precision::FP4:  return "fp4";
    }
    return "fp32";
}

// FP8 E4M3: 1 sign, 4 exponent, 3 mantissa.  (NVIDIA-standard "fp8-e4m3")
inline uint8_t fp8E4M3Encode(float f) {
    uint32_t uu;
    __builtin_memcpy(&uu, &f, 4);
    uint32_t sign = (uu >> 31) & 1u;
    int32_t exp = (int32_t)((uu >> 23) & 0xffu) - 127;
    uint32_t mant = uu & 0x7fffffu;

    if ((uu & 0x7fffffffu) == 0) return (uint8_t)(sign << 7);  // +/-0

    // E4M3: bias 7, exp range [-6, 8], mantissa 3 bits (leading 1 implicit).
    // Max finite ~448, min subnormal ~2^-9.
    if (exp > 8) {
        // overflow -> clamp to max finite (or inf is not allowed in E4M3;
        // clamp to 448 for positive, -448 for negative)
        return (uint8_t)((sign << 7) | 0x7Eu);
    }
    if (exp < -6) {
        // subnormal or zero
        if (exp < -8) return (uint8_t)(sign << 7);
        // subnormal: value = mantissa2 * 2^-9, mantissa2 in [1, 7]
        int shift = 8 - (exp + 6);  // align
        uint32_t m2 = (1u | (mant >> 20)) >> shift;
        // round to nearest even on the retained sub-mantissa
        return (uint8_t)((sign << 7) | (uint32_t)m2);
    }
    uint32_t m2 = mant >> 20;  // top 3 mantissa bits, rounding below
    uint32_t rem = mant & 0xfffffu;
    // round to nearest, ties to even
    if (rem > 0x80000u || (rem == 0x80000u && (m2 & 1u))) m2++;
    uint32_t biased = (uint32_t)(exp + 7);
    if (m2 > 7) { m2 = 0; biased++; }  // carry into exponent
    return (uint8_t)((sign << 7) | (biased << 3) | m2);
}

inline float fp8E4M3Decode(uint8_t b) {
    uint32_t sign = (b >> 7) & 1u;
    uint32_t exp = (b >> 3) & 0xfu;
    uint32_t mant = b & 0x7u;
    float val;
    if (exp == 0) {
        val = (float)mant * 0.001953125f;  // 2^-9
    } else {
        int32_t e = (int32_t)exp - 7;
        uint32_t mm = (1u << 3) | mant;  // implicit leading 1
        val = ldexpf((float)mm, e - 3);   // mm * 2^(e-3)
    }
    return sign ? -val : val;
}

// FP4 E2M1: 1 sign, 2 exponent, 1 mantissa. (NVIDIA-style "fp4-e2m1")
// Standard grid (positive): 0, 0.5, 1, 1.5, 2, 3, 4, 6.
// Bits: [sign][exp2][exp1][mant]. e==0 -> subnormal m*0.5; else (2+m)*2^(e-2).
inline uint8_t fp4E2M1Encode(float f) {
    float a = f < 0.0f ? -f : f;
    uint32_t uu;
    __builtin_memcpy(&uu, &f, 4);
    uint8_t sign = (uint8_t)((uu >> 31) & 1u);
    // work on abs value; binary search nearest of the 8 grid magnitudes
    static const float grid[4][2] = {
        {0.0f, 0.5f}, {1.0f, 1.5f}, {2.0f, 3.0f}, {4.0f, 6.0f}};
    uint8_t bestE = 0, bestM = 0;
    float bestD = a;
    for (uint32_t e = 0; e < 4; e++) {
        for (uint32_t m = 0; m < 2; m++) {
            float v = grid[e][m];
            float d = a > v ? a - v : v - a;
            if (d <= bestD + 1e-30f) {  // prefer lower index on ties (round-to-zero-ish)
                bestD = d;
                bestE = (uint8_t)e;
                bestM = (uint8_t)m;
            }
        }
    }
    return (uint8_t)((sign << 3) | (bestE << 1) | bestM);
}

inline float fp4E2M1Decode(uint8_t b) {
    uint32_t sign = (b >> 3) & 1u;
    uint32_t e = (b >> 1) & 0x3u;
    uint32_t m = b & 0x1u;
    float val;
    if (e == 0) {
        val = (float)m * 0.5f;  // subnormal: 0 or 0.5
    } else {
        val = (float)(2u + m) * ldexpf(1.0f, (int)e - 2);  // (2+m)*2^(e-2)
    }
    return sign ? -val : val;
}

// Quantize a float to target precision (round-trip VALUE in FP32 storage).
// FP32 is identity. This is Buck's sole quantization entry point that both the
// CPU oracle and (conceptually) the quantize_fp.comp shader must match.
inline float quantizeTo(Precision p, float f) {
    switch (p) {
        case Precision::FP8: return fp8E4M3Decode(fp8E4M3Encode(f));
        case Precision::FP4: return fp4E2M1Decode(fp4E2M1Encode(f));
        default: return f;
    }
}
