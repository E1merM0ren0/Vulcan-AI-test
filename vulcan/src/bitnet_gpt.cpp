// vulcan_bitnet_gpt: BitNet-1.58 (W1.58A8) gpt2-style byte LM on the Vulkan engine.
//
// Architecture: learned token+position embeddings, BLOCKS transformer blocks
// (RMSNorm -> bitLinear q/k/v (shared activation quantization) -> per-head
// causal attention -> bitLinear output -> residual; RMSNorm -> bitLinear up ->
// GELU -> bitLinear down -> residual), final RMSNorm, untied bitLinear lm head.
// No biases, no layer scaling.
//
// Task: a row is 784 bytes followed by a SEP token (SEQ = 785). Targets are the
// next byte (positions 0..783) and the class id (position 784). Class id sits in
// [256, 256+NCLS); V = 256 + NCLS + 1.
//
// Every GPU training step is gated against a full CPU oracle:
//   loss_diff < 5e-3  and  pre-Adam grad maxdiff < 1e-3  =>  apply Adam
// Any mismatch aborts (ORACLE FAIL) - the graph math must be a pixel-exact
// mirror of the CPU reference.
//
// GGUF export uses llama.cpp "bitnet" tensor names; row-major weights [out, in]
// are written with dims {in, out} (fastest-dim-first, no data transposition).

#include "graph.h"
#include "context.h"
#include "gguf.h"
#include "vulkan_api.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

static const int E = 64;
static const int HEADS = 4;
static const int D = 16;
static const int FF = 256;
static const int BLOCKS = 2;
static const int BATCH = 4;
static const int SEQ = 785;
static const int BYTES = 784;
static const float LN_EPS = 1e-5f;
static const float LR = 1e-3f;
static const float LOSS_TOL = 5e-3f;
static const float GRAD_TOL = 1e-3f;

static int f32bits_i(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (int)u;
}

static std::string findSPVDir() {
    std::vector<std::string> cands = {
        "vulcan/build/spv", "build/spv", "spv",
        "vulcan\\build\\spv", "build\\spv",
    };
    for (const auto& c : cands) {
        std::ifstream f(c + "/matmul.spv");
        if (f.good()) return c;
    }
    return "vulcan/build/spv";
}

static float roundEven(float x) { return std::nearbyint(x); }
static float geluF(float x) {
    float c = std::sqrt(2.0f / 3.14159265358979323846f);
    return 0.5f * x * (1.0f + std::tanh(c * (x + 0.044715f * x * x * x)));
}
static float geluD(float x) {
    float c = std::sqrt(2.0f / 3.14159265358979323846f);
    float u = c * (x + 0.044715f * x * x * x);
    float t = std::tanh(u);
    float dudx = c * (1.0f + 3.0f * 0.044715f * x * x);
    return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * dudx;
}

// ---------------------------------------------------------------------------
// CPU oracle (exact float mirror of the GPU graph)
// ---------------------------------------------------------------------------

struct CPUModel {
    int V = 0, NCLS = 0, n = 0;
    std::vector<float> tokEmb, posEmb, finalNormW, lmHeadW;
    struct B {
        std::vector<float> attnNormW, qW, kW, vW, oW, ffnNormW, fUpW, fDnW;
    } b[BLOCKS];
    std::vector<float> x, h, z, logits;
    std::vector<float> dHcur, dQ, dK, dV;
    std::vector<float> dZ, dG, dB, dH1b, dXres, dAttnM, dN1, dXa;
    std::vector<float> dLogitsC, lmXq, lmWq, lmSx, lmSw;
    std::vector<float> fwdH0;
    struct BI {
        std::vector<float> xIn, a, q, k, v, attnM, attnOut, h1, bn, gpre, g, ff;
        std::vector<float> p[BATCH][HEADS];  // [SEQ*SEQ] per (batch, head)
    } bi[BLOCKS];
    std::vector<float> ids, targets;

    void initRandom(int ncls, std::mt19937& rng) {
        NCLS = ncls;
        V = 256 + ncls + 1;
        n = BATCH * SEQ;
        std::normal_distribution<float> nd(0.0f, 0.02f);
        auto fill = [&](std::vector<float>& w, int sz) {
            w.resize((size_t)sz);
            for (auto& v : w) v = nd(rng);
        };
        fill(tokEmb, V * E);
        fill(posEmb, SEQ * E);
        fill(finalNormW, E);
        fill(lmHeadW, V * E);
        for (int l = 0; l < BLOCKS; l++) {
            fill(b[l].attnNormW, E);
            fill(b[l].qW, E * E);
            fill(b[l].kW, E * E);
            fill(b[l].vW, E * E);
            fill(b[l].oW, E * E);
            fill(b[l].ffnNormW, E);
            fill(b[l].fUpW, FF * E);
            fill(b[l].fDnW, E * FF);
        }
        x.assign((size_t)n * E, 0);
        h.assign((size_t)n * E, 0);
        z.assign((size_t)n * E, 0);
        logits.assign((size_t)n * V, 0);
        for (int l = 0; l < BLOCKS; l++) {
            auto& r = bi[l];
            r.xIn.assign((size_t)n * E, 0);
            r.a.assign((size_t)n * E, 0);
            r.q.assign((size_t)n * E, 0);
            r.k.assign((size_t)n * E, 0);
            r.v.assign((size_t)n * E, 0);
            r.attnM.assign((size_t)n * E, 0);
            r.attnOut.assign((size_t)n * E, 0);
            r.h1.assign((size_t)n * E, 0);
            r.bn.assign((size_t)n * E, 0);
            r.gpre.assign((size_t)n * FF, 0);
            r.g.assign((size_t)n * FF, 0);
            r.ff.assign((size_t)n * E, 0);
            for (int bb = 0; bb < BATCH; bb++)
                for (int hh = 0; hh < HEADS; hh++) r.p[bb][hh].assign((size_t)SEQ * SEQ, 0);
        }
        ids.assign((size_t)n, 0);
        targets.assign((size_t)n, 0);
    }

    static void qAbsmax(int n_, int k, const float* X, float* Xq, float* sx) {
        for (int r = 0; r < n_; r++) {
            float mx = 0.0f;
            for (int j = 0; j < k; j++) mx = std::max(mx, std::fabs(X[r * k + j]));
            float s = mx == 0.0f ? 0.0f : mx / 127.0f;
            sx[r] = s;
            for (int j = 0; j < k; j++) {
                float q = s == 0.0f ? 0.0f : roundEven(X[r * k + j] / s);
                q = std::max(-127.0f, std::min(127.0f, q));
                Xq[r * k + j] = q;
            }
        }
    }
    static void qTernary(int n_, int k, const float* W, float* Wq, float* sw) {
        for (int r = 0; r < n_; r++) {
            float acc = 0.0f;
            for (int j = 0; j < k; j++) acc += std::fabs(W[r * k + j]);
            float s = acc / (float)k;
            sw[r] = s;
            for (int j = 0; j < k; j++) {
                float q = s == 0.0f ? 0.0f : roundEven(W[r * k + j] / s);
                q = std::max(-1.0f, std::min(1.0f, q));
                Wq[r * k + j] = q;
            }
        }
    }
    static void blFwd(int n_, int k, int o, const float* X, const float* W, float* Y,
                      std::vector<float>& Xq, std::vector<float>& Wq,
                      std::vector<float>& sx, std::vector<float>& sw) {
        Xq.resize((size_t)n_ * k);
        Wq.resize((size_t)o * k);
        sx.resize((size_t)n_);
        sw.resize((size_t)o);
        qAbsmax(n_, k, X, Xq.data(), sx.data());
        qTernary(o, k, W, Wq.data(), sw.data());
        for (int r = 0; r < n_; r++)
            for (int oo = 0; oo < o; oo++) {
                float acc = 0.0f;
                for (int i = 0; i < k; i++) acc += Xq[r * k + i] * Wq[oo * k + i];
                Y[r * o + oo] = (acc * sx[r]) * sw[oo];
            }
    }
    static void blBwd(int n_, int k, int o, const float* dY, float* dX, float* dW,
                      const std::vector<float>& Xq, const std::vector<float>& Wq,
                      const std::vector<float>& sx, const std::vector<float>& sw) {
        for (int r = 0; r < n_; r++)
            for (int i = 0; i < k; i++) {
                float acc = 0.0f;
                for (int oo = 0; oo < o; oo++) acc += dY[r * o + oo] * sw[oo] * Wq[oo * k + i];
                dX[r * k + i] = sx[r] * acc;
            }
        for (int oo = 0; oo < o; oo++)
            for (int i = 0; i < k; i++) {
                float acc = 0.0f;
                for (int r = 0; r < n_; r++) acc += dY[r * o + oo] * sx[r] * Xq[r * k + i];
                dW[oo * k + i] = sw[oo] * acc;
            }
    }
    static void rmsFwd(int n_, int cols, const float* X, const float* W, float* O) {
        for (int r = 0; r < n_; r++) {
            float sum = 0.0f;
            for (int j = 0; j < cols; j++) sum += X[r * cols + j] * X[r * cols + j];
            float denom = std::sqrt(sum / (float)cols + LN_EPS);
            for (int j = 0; j < cols; j++) O[r * cols + j] = X[r * cols + j] / denom * W[j];
        }
    }
    static void rmsBwd(int n_, int cols, const float* X, const float* W,
                       const float* dO, float* dX, float* dW) {
        std::vector<float> dw((size_t)cols, 0.0f);
        for (int r = 0; r < n_; r++) {
            float sum = 0.0f;
            for (int j = 0; j < cols; j++) sum += X[r * cols + j] * X[r * cols + j];
            float denom = std::sqrt(sum / (float)cols + LN_EPS);
            float dot = 0.0f;
            for (int j = 0; j < cols; j++) dot += dO[r * cols + j] * W[j] * X[r * cols + j];
            float scale = dot / ((float)cols * denom * denom);
            for (int j = 0; j < cols; j++) {
                dX[r * cols + j] = dO[r * cols + j] * W[j] / denom - X[r * cols + j] / denom * scale;
                dw[j] += dO[r * cols + j] * X[r * cols + j] / denom;
            }
        }
        for (int j = 0; j < cols; j++) dW[j] = dw[j];
    }

    void forward() {
        for (int r = 0; r < n; r++) {
            int id = (int)ids[r];
            int t = r % SEQ;
            for (int j = 0; j < E; j++) x[r * E + j] = tokEmb[id * E + j] + posEmb[t * E + j];
        }
        const float* in = x.data();
        for (int l = 0; l < BLOCKS; l++) {
            auto& r = bi[l];
            auto& B_ = b[l];
            r.xIn.assign(in, in + (size_t)n * E);
            rmsFwd(n, E, in, B_.attnNormW.data(), r.a.data());
            std::vector<float> Xq, sx;
            Xq.resize((size_t)n * E);
            sx.resize((size_t)n);
            qAbsmax(n, E, r.a.data(), Xq.data(), sx.data());
            {
                std::vector<float> Wq, sw;
                Wq.resize((size_t)E * E);
                sw.resize((size_t)E);
                qTernary(E, E, B_.qW.data(), Wq.data(), sw.data());
                for (int rr = 0; rr < n; rr++)
                    for (int oo = 0; oo < E; oo++) {
                        float acc = 0.0f;
                        for (int i = 0; i < E; i++) acc += Xq[rr * E + i] * Wq[oo * E + i];
                        r.q[rr * E + oo] = (acc * sx[rr]) * sw[oo];
                    }
                std::vector<float> Wqk, swk;
                Wqk.resize((size_t)E * E);
                swk.resize((size_t)E);
                qTernary(E, E, B_.kW.data(), Wqk.data(), swk.data());
                for (int rr = 0; rr < n; rr++)
                    for (int oo = 0; oo < E; oo++) {
                        float acc = 0.0f;
                        for (int i = 0; i < E; i++) acc += Xq[rr * E + i] * Wqk[oo * E + i];
                        r.k[rr * E + oo] = (acc * sx[rr]) * swk[oo];
                    }
                std::vector<float> Wqv, swv;
                Wqv.resize((size_t)E * E);
                swv.resize((size_t)E);
                qTernary(E, E, B_.vW.data(), Wqv.data(), swv.data());
                for (int rr = 0; rr < n; rr++)
                    for (int oo = 0; oo < E; oo++) {
                        float acc = 0.0f;
                        for (int i = 0; i < E; i++) acc += Xq[rr * E + i] * Wqv[oo * E + i];
                        r.v[rr * E + oo] = (acc * sx[rr]) * swv[oo];
                    }
            }
            std::fill(r.attnM.begin(), r.attnM.end(), 0.0f);
            std::vector<float> s((size_t)SEQ * SEQ), o((size_t)SEQ * D);
            for (int bb = 0; bb < BATCH; bb++) {
                for (int hh = 0; hh < HEADS; hh++) {
                    for (int p = 0; p < SEQ; p++) {
                        int g = bb * SEQ + p;
                        for (int jj = 0; jj <= p; jj++) {
                            float acc = 0.0f;
                            for (int dd = 0; dd < D; dd++) {
                                acc += r.q[g * E + hh * D + dd] * r.k[(bb * SEQ + jj) * E + hh * D + dd];
                            }
                            s[p * SEQ + jj] = acc;
                        }
                        for (int jj = p + 1; jj < SEQ; jj++) s[p * SEQ + jj] = -1e30f;
                        float m = -1e30f;
                        for (int jj = 0; jj < SEQ; jj++) m = std::max(m, s[p * SEQ + jj]);
                        float sum = 0.0f;
                        for (int jj = 0; jj < SEQ; jj++) {
                            float e = std::exp(s[p * SEQ + jj] - m);
                            r.p[bb][hh][p * SEQ + jj] = e;
                            sum += e;
                        }
                        for (int jj = 0; jj < SEQ; jj++) r.p[bb][hh][p * SEQ + jj] /= sum;
                        for (int dd = 0; dd < D; dd++) {
                            float acc = 0.0f;
                            for (int jj = 0; jj < SEQ; jj++) {
                                acc += r.p[bb][hh][p * SEQ + jj] * r.v[(bb * SEQ + jj) * E + hh * D + dd];
                            }
                            o[p * D + dd] = acc;
                        }
                    }
                    for (int p = 0; p < SEQ; p++)
                        for (int dd = 0; dd < D; dd++)
                            r.attnM[(bb * SEQ + p) * E + hh * D + dd] = o[p * D + dd];
                }
            }
            std::vector<float> Xq2, Wq2, sx2, sw2;
            blFwd(n, E, E, r.attnM.data(), B_.oW.data(), r.attnOut.data(), Xq2, Wq2, sx2, sw2);
            for (int i = 0; i < n * E; i++) r.h1[i] = in[i] + r.attnOut[i];
            rmsFwd(n, E, r.h1.data(), B_.ffnNormW.data(), r.bn.data());
            blFwd(n, E, FF, r.bn.data(), B_.fUpW.data(), r.gpre.data(), Xq2, Wq2, sx2, sw2);
            for (int i = 0; i < n * FF; i++) r.g[i] = geluF(r.gpre[i]);
            blFwd(n, FF, E, r.g.data(), B_.fDnW.data(), r.ff.data(), Xq2, Wq2, sx2, sw2);
            for (int i = 0; i < n * E; i++) h[i] = r.h1[i] + r.ff[i];
            if (l == 0) fwdH0 = h;
            in = h.data();
        }
        rmsFwd(n, E, in, finalNormW.data(), z.data());
        std::vector<float> Xq, Wq, sx, sw;
        blFwd(n, E, V, z.data(), lmHeadW.data(), logits.data(), Xq, Wq, sx, sw);
    }

    float ceLoss() const {
        float total = 0.0f;
        for (int r = 0; r < n; r++) {
            int t = (int)targets[r];
            float m = -1e30f;
            for (int j = 0; j < V; j++) m = std::max(m, logits[r * V + j]);
            float sum = 0.0f;
            for (int j = 0; j < V; j++) sum += std::exp(logits[r * V + j] - m);
            total += ((m + std::log(sum)) - logits[r * V + t]) / (float)n;
        }
        return total;
    }

    std::vector<float> gTokEmb, gPosEmb, gFinalNormW, gLmHeadW;
    struct GB {
        std::vector<float> attnNormW, qW, kW, vW, oW, ffnNormW, fUpW, fDnW;
    } gb[BLOCKS];

    void initGrads() {
        gTokEmb.assign((size_t)V * E, 0);
        gPosEmb.assign((size_t)SEQ * E, 0);
        gFinalNormW.assign(E, 0);
        gLmHeadW.assign((size_t)V * E, 0);
        for (int l = 0; l < BLOCKS; l++) {
            gb[l].attnNormW.assign(E, 0);
            gb[l].qW.assign((size_t)E * E, 0);
            gb[l].kW.assign((size_t)E * E, 0);
            gb[l].vW.assign((size_t)E * E, 0);
            gb[l].oW.assign((size_t)E * E, 0);
            gb[l].ffnNormW.assign(E, 0);
            gb[l].fUpW.assign((size_t)FF * E, 0);
            gb[l].fDnW.assign((size_t)E * FF, 0);
        }
    }

    void backward() {
        initGrads();
        std::vector<float> dLogits((size_t)n * V, 0.0f);
        for (int r = 0; r < n; r++) {
            int t = (int)targets[r];
            float m = -1e30f;
            for (int j = 0; j < V; j++) m = std::max(m, logits[r * V + j]);
            float sum = 0.0f;
            for (int j = 0; j < V; j++) sum += std::exp(logits[r * V + j] - m);
            for (int j = 0; j < V; j++) {
                float p = std::exp(logits[r * V + j] - m) / sum;
                dLogits[r * V + j] = (p - (j == t ? 1.0f : 0.0f)) / (float)n;
            }
        }
        this->dLogitsC = dLogits;
        std::vector<float> dZ((size_t)n * E, 0.0f);
        {
            std::vector<float> Xq, Wq, sx, sw;
            blFwd(n, E, V, z.data(), lmHeadW.data(), logits.data(), Xq, Wq, sx, sw);
            blBwd(n, E, V, dLogits.data(), dZ.data(), gLmHeadW.data(), Xq, Wq, sx, sw);
            this->lmXq = Xq; this->lmWq = Wq; this->lmSx = sx; this->lmSw = sw;
        }
        this->dZ = dZ;
        std::vector<float> dH((size_t)n * E, 0.0f);
        rmsBwd(n, E, h.data(), finalNormW.data(), dZ.data(),
               dH.data(), gFinalNormW.data());

        std::vector<float> dQ((size_t)n * E, 0.0f), dK((size_t)n * E, 0.0f), dV((size_t)n * E, 0.0f);
        for (int l = BLOCKS - 1; l >= 0; l--) {
            auto& r = bi[l];
            auto& B_ = b[l];
            auto& G_ = gb[l];
            std::vector<float> dG((size_t)n * FF, 0.0f);
            {
                std::vector<float> Xq, Wq, sx, sw;
                blFwd(n, FF, E, r.g.data(), B_.fDnW.data(), r.ff.data(), Xq, Wq, sx, sw);
                blBwd(n, FF, E, dH.data(), dG.data(), G_.fDnW.data(), Xq, Wq, sx, sw);
            }
            for (int i = 0; i < n * FF; i++) dG[i] *= geluD(r.gpre[i]);
            this->dG = dG;
            std::vector<float> dB((size_t)n * E, 0.0f);
            {
                std::vector<float> Xq, Wq, sx, sw;
                blFwd(n, E, FF, r.bn.data(), B_.fUpW.data(), r.gpre.data(), Xq, Wq, sx, sw);
                blBwd(n, E, FF, dG.data(), dB.data(), G_.fUpW.data(), Xq, Wq, sx, sw);
            }
            this->dB = dB;
            std::vector<float> dH1b((size_t)n * E, 0.0f);
            rmsBwd(n, E, r.h1.data(), B_.ffnNormW.data(), dB.data(),
                   dH1b.data(), G_.ffnNormW.data());
            this->dH1b = dH1b;
            std::vector<float> dXres((size_t)n * E, 0.0f);
            for (int i = 0; i < n * E; i++) dXres[i] = dH[i] + dH1b[i];
            this->dXres = dXres;
            std::vector<float> dAttnM((size_t)n * E, 0.0f);
            {
                std::vector<float> Xq, Wq, sx, sw;
                blFwd(n, E, E, r.attnM.data(), B_.oW.data(), r.attnOut.data(), Xq, Wq, sx, sw);
                blBwd(n, E, E, dXres.data(), dAttnM.data(), G_.oW.data(), Xq, Wq, sx, sw);
            }
            this->dAttnM = dAttnM;
            for (int bb = 0; bb < BATCH; bb++) {
                for (int hh = 0; hh < HEADS; hh++) {
                    std::vector<float> dP((size_t)SEQ * SEQ, 0.0f);
                    for (int p = 0; p < SEQ; p++) {
                        int g = bb * SEQ + p;
                        for (int jj = 0; jj < SEQ; jj++) {
                            float acc = 0.0f;
                            for (int dd = 0; dd < D; dd++) {
                                acc += dAttnM[g * E + hh * D + dd] *
                                       r.v[(bb * SEQ + jj) * E + hh * D + dd];
                            }
                            dP[p * SEQ + jj] = acc;
                        }
                    }
                    for (int p = 0; p < SEQ; p++) {
                        float dot = 0.0f;
                        for (int jj = 0; jj < SEQ; jj++)
                            dot += r.p[bb][hh][p * SEQ + jj] * dP[p * SEQ + jj];
                        for (int jj = 0; jj < SEQ; jj++) {
                            float dS = r.p[bb][hh][p * SEQ + jj] * (dP[p * SEQ + jj] - dot);
                            int g = bb * SEQ + p;
                            int gj = bb * SEQ + jj;
                            for (int dd = 0; dd < D; dd++) {
                                dQ[g * E + hh * D + dd] += dS * r.k[gj * E + hh * D + dd];
                                dK[gj * E + hh * D + dd] += dS * r.q[g * E + hh * D + dd];
                                dV[gj * E + hh * D + dd] +=
                                    r.p[bb][hh][p * SEQ + jj] * dAttnM[g * E + hh * D + dd];
                            }
                        }
                    }
                }
            }
            std::vector<float> dN1((size_t)n * E, 0.0f);
            {
                std::vector<float> Xq, sx;
                Xq.resize((size_t)n * E);
                sx.resize((size_t)n);
                qAbsmax(n, E, r.a.data(), Xq.data(), sx.data());
                std::vector<float> tmp((size_t)n * E, 0.0f);
                std::vector<float> Wq, sw;
                Wq.resize((size_t)E * E);
                sw.resize((size_t)E);
                qTernary(E, E, B_.qW.data(), Wq.data(), sw.data());
                blBwd(n, E, E, dQ.data(), tmp.data(), G_.qW.data(), Xq, Wq, sx, sw);
                for (int i = 0; i < n * E; i++) dN1[i] += tmp[i];
                qTernary(E, E, B_.kW.data(), Wq.data(), sw.data());
                blBwd(n, E, E, dK.data(), tmp.data(), G_.kW.data(), Xq, Wq, sx, sw);
                for (int i = 0; i < n * E; i++) dN1[i] += tmp[i];
                qTernary(E, E, B_.vW.data(), Wq.data(), sw.data());
                blBwd(n, E, E, dV.data(), tmp.data(), G_.vW.data(), Xq, Wq, sx, sw);
                for (int i = 0; i < n * E; i++) dN1[i] += tmp[i];
            }
            this->dN1 = dN1;
            std::vector<float> dXa((size_t)n * E, 0.0f);
            rmsBwd(n, E, r.xIn.data(), B_.attnNormW.data(), dN1.data(),
                   dXa.data(), G_.attnNormW.data());
            this->dXa = dXa;
            for (int i = 0; i < n * E; i++) dH[i] = dXres[i] + dXa[i];
        }
        dHcur = dH;
        this->dQ = dQ; this->dK = dK; this->dV = dV;
        for (int i = 0; i < n * E; i++) {
            int t = i / E;
            gPosEmb[(t % SEQ) * E + (i % E)] += dH[i];
        }
        for (int r = 0; r < n; r++) {
            int id = (int)ids[r];
            for (int j = 0; j < E; j++) gTokEmb[id * E + j] += dH[r * E + j];
        }
    }
};

// ---------------------------------------------------------------------------
// GPU model + graph builder
// ---------------------------------------------------------------------------

struct GPUModel {
    vk::Context ctx;
    Graph g;

    int V = 0, NCLS = 0, n = 0, SEP_ID = 0;

    Tensor *ids = nullptr, *targets = nullptr, *mask = nullptr, *ones = nullptr;
    Tensor *lossPartial = nullptr, *loss = nullptr;
    Tensor *x0 = nullptr, *z = nullptr, *zq = nullptr, *sz = nullptr;
    Tensor *wqLM = nullptr, *swLM = nullptr, *logits = nullptr;

    Tensor *tokEmbW = nullptr, *posEmbW = nullptr, *finalNormW = nullptr, *lmHeadW = nullptr;
    Tensor *dTokEmbW = nullptr, *dPosEmbW = nullptr, *dFinalNormW = nullptr, *dLmHeadW = nullptr;
    std::vector<Tensor*> params, grads, m, v;

    struct B {
        Tensor *attnNormW, *qW, *kW, *vW, *oW, *ffnNormW, *fUpW, *fDnW;
        Tensor *dAttnNormW, *dQW, *dKW, *dVW, *dOW, *dFfnNormW, *dFuW, *dFdW;
        Tensor *wqQ, *swQ, *wqK, *swK, *wqV, *swV, *wqO, *swO, *wqFU, *swFU, *wqFD, *swFD;
        Tensor *a, *Xq, *sx, *q, *k, *v;
        Tensor *qh[BATCH][HEADS], *kh[BATCH][HEADS], *vh[BATCH][HEADS], *p[BATCH][HEADS];
        Tensor *attnM, *Xqo, *sxo, *attnOut, *h1, *bn, *Xqfu, *sxfu, *gpre, *g, *Xqfd, *sxfd, *ff, *h;
        Tensor *M1oW, *M2oW, *M1fU, *M2fU, *M1fD, *M2fD, *M1q, *M2q, *M1k, *M2k, *M1v, *M2v;
    } blk[BLOCKS];

    Tensor *sS, *smS, *oS, *dPS, *dSS, *dAttnhS, *dQhS, *dKhS, *dVhS;
    Tensor *dZ, *dLogits, *M1lm, *M2lm;
    Tensor *dHcur, *dXres, *dXa, *dXout, *dG, *dB, *dAttnM, *dH1b, *dwP, *dQ, *dK, *dV, *dN1;

    void allocParams() {
        auto addParam = [&](Tensor* p) {
            params.push_back(p);
            Tensor* gd = g.alloc(p->n, p->cols);
            grads.push_back(gd);
            m.push_back(g.alloc(p->n, p->cols));
            v.push_back(g.alloc(p->n, p->cols));
        };
        tokEmbW = g.alloc(V, E);
        posEmbW = g.alloc(SEQ, E);
        finalNormW = g.alloc(1, E);
        lmHeadW = g.alloc(V, E);
        addParam(tokEmbW);
        addParam(posEmbW);
        addParam(finalNormW);
        addParam(lmHeadW);
        for (int l = 0; l < BLOCKS; l++) {
            auto& b = blk[l];
            b.attnNormW = g.alloc(1, E);
            b.qW = g.alloc(E, E);
            b.kW = g.alloc(E, E);
            b.vW = g.alloc(E, E);
            b.oW = g.alloc(E, E);
            b.ffnNormW = g.alloc(1, E);
            b.fUpW = g.alloc(FF, E);
            b.fDnW = g.alloc(E, FF);
            addParam(b.attnNormW);
            addParam(b.qW);
            addParam(b.kW);
            addParam(b.vW);
            addParam(b.oW);
            addParam(b.ffnNormW);
            addParam(b.fUpW);
            addParam(b.fDnW);
        }
        dTokEmbW = grads[0];
        dPosEmbW = grads[1];
        dFinalNormW = grads[2];
        dLmHeadW = grads[3];
    }

    void allocAll() {
        ids = g.alloc(n, 1);
        targets = g.alloc(n, 1);
        ones = g.alloc(8192, 1);
        std::fill(ones->data.begin(), ones->data.end(), 1.0f);
        mask = g.alloc(SEQ, SEQ);
        for (int i = 0; i < SEQ; i++)
            for (int j = 0; j < SEQ; j++) mask->data[i * SEQ + j] = (j <= i) ? 0.0f : -1e30f;
        lossPartial = g.alloc(n, 1);
        loss = g.alloc(1, 1);

        x0 = g.alloc(n, E);
        z = g.alloc(n, E);
        zq = g.alloc(n, E);
        sz = g.alloc(n, 1);
        wqLM = g.alloc(V, E);
        swLM = g.alloc(V, 1);
        logits = g.alloc(n, V);

        dZ = g.alloc(n, E);
        dLogits = g.alloc(n, V);
        M1lm = g.alloc(n, V);
        M2lm = g.alloc(n, E);
        dHcur = g.alloc(n, E);
        dXres = g.alloc(n, E);
        dXa = g.alloc(n, E);
        dXout = g.alloc(n, E);
        dG = g.alloc(n, FF);
        dB = g.alloc(n, E);
        dAttnM = g.alloc(n, E);
        dH1b = g.alloc(n, E);
        dwP = g.alloc(n, E);
        dQ = g.alloc(n, E);
        dK = g.alloc(n, E);
        dV = g.alloc(n, E);
        dN1 = g.alloc(n, E);

        sS = g.alloc(SEQ, SEQ);
        smS = g.alloc(SEQ, SEQ);
        oS = g.alloc(SEQ, D);
        dPS = g.alloc(SEQ, SEQ);
        dSS = g.alloc(SEQ, SEQ);
        dAttnhS = g.alloc(SEQ, D);
        dQhS = g.alloc(SEQ, D);
        dKhS = g.alloc(SEQ, D);
        dVhS = g.alloc(SEQ, D);

        for (int l = 0; l < BLOCKS; l++) {
            auto& b = blk[l];
            b.dAttnNormW = g.alloc(1, E);
            b.dQW = g.alloc(E, E);
            b.dKW = g.alloc(E, E);
            b.dVW = g.alloc(E, E);
            b.dOW = g.alloc(E, E);
            b.dFfnNormW = g.alloc(1, E);
            b.dFuW = g.alloc(FF, E);
            b.dFdW = g.alloc(E, FF);
            b.wqQ = g.alloc(E, E);
            b.swQ = g.alloc(E, 1);
            b.wqK = g.alloc(E, E);
            b.swK = g.alloc(E, 1);
            b.wqV = g.alloc(E, E);
            b.swV = g.alloc(E, 1);
            b.wqO = g.alloc(E, E);
            b.swO = g.alloc(E, 1);
            b.wqFU = g.alloc(FF, E);
            b.swFU = g.alloc(FF, 1);
            b.wqFD = g.alloc(E, FF);
            b.swFD = g.alloc(E, 1);

            b.a = g.alloc(n, E);
            b.Xq = g.alloc(n, E);
            b.sx = g.alloc(n, 1);
            b.q = g.alloc(n, E);
            b.k = g.alloc(n, E);
            b.v = g.alloc(n, E);
            for (int bb = 0; bb < BATCH; bb++)
                for (int hh = 0; hh < HEADS; hh++) {
                    b.qh[bb][hh] = g.alloc(SEQ, D);
                    b.kh[bb][hh] = g.alloc(SEQ, D);
                    b.vh[bb][hh] = g.alloc(SEQ, D);
                    b.p[bb][hh] = g.alloc(SEQ, SEQ);
                }
            b.attnM = g.alloc(n, E);
            b.Xqo = g.alloc(n, E);
            b.sxo = g.alloc(n, 1);
            b.attnOut = g.alloc(n, E);
            b.h1 = g.alloc(n, E);
            b.bn = g.alloc(n, E);
            b.Xqfu = g.alloc(n, E);
            b.sxfu = g.alloc(n, 1);
            b.gpre = g.alloc(n, FF);
            b.g = g.alloc(n, FF);
            b.Xqfd = g.alloc(n, FF);
            b.sxfd = g.alloc(n, 1);
            b.ff = g.alloc(n, E);
            b.h = g.alloc(n, E);
            b.M1oW = g.alloc(n, E);
            b.M2oW = g.alloc(n, E);
            b.M1fU = g.alloc(n, FF);
            b.M2fU = g.alloc(n, E);
            b.M1fD = g.alloc(n, E);
            b.M2fD = g.alloc(n, FF);
            b.M1q = g.alloc(n, E);
            b.M2q = g.alloc(n, E);
            b.M1k = g.alloc(n, E);
            b.M2k = g.alloc(n, E);
            b.M1v = g.alloc(n, E);
            b.M2v = g.alloc(n, E);
        }
    }

    void addOp(OpKind kind, Tensor* a, Tensor* b, Tensor* c, Tensor* d, Tensor* e,
               Tensor* out, std::vector<int32_t> pc, bool bwd) {
        Op op;
        op.kind = kind;
        op.a = a;
        op.b = b;
        op.c = c;
        op.d = d;
        op.e = e;
        op.out = out;
        for (size_t i = 0; i < 8 && i < pc.size(); i++) op.pc[i] = pc[i];
        op.bwd = bwd;
        g.add(op);
    }

    void addBlFwd(Tensor* x, Tensor* w, Tensor* xq, Tensor* sx, Tensor* wq, Tensor* sw, Tensor* y) {
        addOp(OP_QUANTIZE_ABSMAX, x, xq, sx, nullptr, nullptr, nullptr, {0, 0}, false);
        addOp(OP_QUANTIZE_TERNARY, w, wq, sw, nullptr, nullptr, nullptr, {0, 0}, false);
        addOp(OP_MATMUL, xq, wq, nullptr, nullptr, nullptr, y, {0, 0, 0, 0, 1}, false);
        addOp(OP_SCALE_ROWS, y, sx, sw, nullptr, nullptr, y, {0, 0, 0}, false);
    }

    // dX = sx .* (dY*sw^T) @ wq  ;  dW = sw .* (M1^T @ xq), M1 = dY*sx
    void addBlBwd(Tensor* dY, Tensor* wq, Tensor* sw, Tensor* xq, Tensor* sx,
                  Tensor* ones, Tensor* M1, Tensor* M2, Tensor* dW, Tensor* dXacc, int acc) {
        addOp(OP_SCALE_ROWS, dY, sx, ones, nullptr, nullptr, M1, {0, 0, 0}, true);
        addOp(OP_MATMUL, M1, xq, nullptr, nullptr, nullptr, dW, {0, 0, 0, 1, 0}, true);
        addOp(OP_SCALE_ROWS, dW, ones, sw, nullptr, nullptr, dW, {0, 0, 0}, true);
        addOp(OP_SCALE_ROWS, dY, ones, sw, nullptr, nullptr, M2, {0, 0, 0}, true);
        addOp(OP_MATMUL, M2, wq, nullptr, nullptr, nullptr, M2, {0, 0, 0, 0, 0}, true);
        addOp(OP_SCALE_ROWS, M2, sx, ones, nullptr, nullptr, dXacc, {0, 0, (int32_t)acc}, true);
    }

    void build() {
        // ---- forward ----
        addOp(OP_EMBEDDING, ids, tokEmbW, nullptr, nullptr, nullptr, x0, {0, 0}, false);
        addOp(OP_ADD_POS, x0, posEmbW, nullptr, nullptr, nullptr, x0, {0, 0, SEQ}, false);

        for (int l = 0; l < BLOCKS; l++) {
            auto& b = blk[l];
            Tensor* in = (l == 0) ? x0 : blk[l - 1].h;
            addOp(OP_RMS_NORM, in, b.attnNormW, nullptr, nullptr, nullptr, b.a,
                  {0, 0, f32bits_i(LN_EPS)}, false);
            addOp(OP_QUANTIZE_ABSMAX, b.a, b.Xq, b.sx, nullptr, nullptr, nullptr, {0, 0}, false);
            addOp(OP_QUANTIZE_TERNARY, b.qW, b.wqQ, b.swQ, nullptr, nullptr, nullptr, {0, 0}, false);
            addOp(OP_MATMUL, b.Xq, b.wqQ, nullptr, nullptr, nullptr, b.q, {0, 0, 0, 0, 1}, false);
            addOp(OP_SCALE_ROWS, b.q, b.sx, b.swQ, nullptr, nullptr, b.q, {0, 0, 0}, false);
            addOp(OP_QUANTIZE_TERNARY, b.kW, b.wqK, b.swK, nullptr, nullptr, nullptr, {0, 0}, false);
            addOp(OP_MATMUL, b.Xq, b.wqK, nullptr, nullptr, nullptr, b.k, {0, 0, 0, 0, 1}, false);
            addOp(OP_SCALE_ROWS, b.k, b.sx, b.swK, nullptr, nullptr, b.k, {0, 0, 0}, false);
            addOp(OP_QUANTIZE_TERNARY, b.vW, b.wqV, b.swV, nullptr, nullptr, nullptr, {0, 0}, false);
            addOp(OP_MATMUL, b.Xq, b.wqV, nullptr, nullptr, nullptr, b.v, {0, 0, 0, 0, 1}, false);
            addOp(OP_SCALE_ROWS, b.v, b.sx, b.swV, nullptr, nullptr, b.v, {0, 0, 0}, false);

            addOp(OP_ZERO, nullptr, nullptr, nullptr, nullptr, nullptr, b.attnM, {0, 0}, false);
            for (int bb = 0; bb < BATCH; bb++) {
                for (int hh = 0; hh < HEADS; hh++) {
                    std::vector<int32_t> sp = {SEQ, HEADS, D, hh, bb, 0, 0, 0};
                    addOp(OP_SPLIT_HEADS, b.q, nullptr, nullptr, nullptr, nullptr, b.qh[bb][hh], sp, false);
                    addOp(OP_SPLIT_HEADS, b.k, nullptr, nullptr, nullptr, nullptr, b.kh[bb][hh], sp, false);
                    addOp(OP_SPLIT_HEADS, b.v, nullptr, nullptr, nullptr, nullptr, b.vh[bb][hh], sp, false);
                    addOp(OP_MATMUL, b.qh[bb][hh], b.kh[bb][hh], nullptr, nullptr, nullptr, sS,
                          {0, 0, 0, 0, 1}, false);
                    addOp(OP_ADD, sS, mask, nullptr, nullptr, nullptr, smS, {0, 0}, false);
                    addOp(OP_SOFTMAX, smS, nullptr, nullptr, nullptr, nullptr, b.p[bb][hh], {0, 0}, false);
                    addOp(OP_MATMUL, b.p[bb][hh], b.vh[bb][hh], nullptr, nullptr, nullptr, oS,
                          {0, 0, 0, 0, 0}, false);
                    addOp(OP_MERGE_HEADS, oS, nullptr, nullptr, nullptr, nullptr, b.attnM, sp, false);
                }
            }
            addBlFwd(b.attnM, b.oW, b.Xqo, b.sxo, b.wqO, b.swO, b.attnOut);
            addOp(OP_ADD, in, b.attnOut, nullptr, nullptr, nullptr, b.h1, {0, 0}, false);
            addOp(OP_RMS_NORM, b.h1, b.ffnNormW, nullptr, nullptr, nullptr, b.bn,
                  {0, 0, f32bits_i(LN_EPS)}, false);
            addBlFwd(b.bn, b.fUpW, b.Xqfu, b.sxfu, b.wqFU, b.swFU, b.gpre);
            addOp(OP_GELU, b.gpre, nullptr, nullptr, nullptr, nullptr, b.g, {0, 0}, false);
            addBlFwd(b.g, b.fDnW, b.Xqfd, b.sxfd, b.wqFD, b.swFD, b.ff);
            addOp(OP_ADD, b.h1, b.ff, nullptr, nullptr, nullptr, b.h, {0, 0}, false);
        }
        addOp(OP_RMS_NORM, blk[BLOCKS - 1].h, finalNormW, nullptr, nullptr, nullptr, z,
              {0, 0, f32bits_i(LN_EPS)}, false);
        addBlFwd(z, lmHeadW, zq, sz, wqLM, swLM, logits);
        addOp(OP_CROSS_ENTROPY, logits, targets, nullptr, nullptr, nullptr, lossPartial, {0, 0}, false);
        addOp(OP_REDUCE_SUM_ROWS, lossPartial, nullptr, nullptr, nullptr, nullptr, loss, {0, 0}, false);

        // ---- backward ----
        addOp(OP_CROSS_ENTROPY_BWD, logits, targets, nullptr, nullptr, nullptr, dLogits,
              {0, 0}, true);
        addBlBwd(dLogits, wqLM, swLM, zq, sz, ones, M1lm, M2lm, dLmHeadW, dZ, 0);
        addOp(OP_RMS_NORM_BWD, blk[BLOCKS - 1].h, finalNormW, dZ, dwP, nullptr, dHcur,
              {0, 0, f32bits_i(LN_EPS)}, true);
        addOp(OP_REDUCE_SUM_ROWS, dwP, nullptr, nullptr, nullptr, nullptr, dFinalNormW, {0, 0}, true);

        for (int l = BLOCKS - 1; l >= 0; l--) {
            auto& b = blk[l];
            Tensor* in = (l == 0) ? x0 : blk[l - 1].h;
            addBlBwd(dHcur, b.wqFD, b.swFD, b.Xqfd, b.sxfd, ones, b.M1fD, b.M2fD, b.dFdW, dG, 0);
            addOp(OP_GELU_BWD, b.gpre, dG, nullptr, nullptr, nullptr, dG, {0, 0}, true);
            addBlBwd(dG, b.wqFU, b.swFU, b.Xqfu, b.sxfu, ones, b.M1fU, b.M2fU, b.dFuW, dB, 0);
            addOp(OP_RMS_NORM_BWD, b.h1, b.ffnNormW, dB, dwP, nullptr, dH1b,
                  {0, 0, f32bits_i(LN_EPS)}, true);
            addOp(OP_REDUCE_SUM_ROWS, dwP, nullptr, nullptr, nullptr, nullptr, b.dFfnNormW, {0, 0}, true);
            addOp(OP_ADD, dHcur, dH1b, nullptr, nullptr, nullptr, dXres, {0, 0}, true);
            addBlBwd(dXres, b.wqO, b.swO, b.Xqo, b.sxo, ones, b.M1oW, b.M2oW, b.dOW, dAttnM, 0);

            addOp(OP_ZERO, nullptr, nullptr, nullptr, nullptr, nullptr, dQ, {0, 0}, true);
            addOp(OP_ZERO, nullptr, nullptr, nullptr, nullptr, nullptr, dK, {0, 0}, true);
            addOp(OP_ZERO, nullptr, nullptr, nullptr, nullptr, nullptr, dV, {0, 0}, true);
            for (int bb = 0; bb < BATCH; bb++) {
                for (int hh = 0; hh < HEADS; hh++) {
                    std::vector<int32_t> sp = {SEQ, HEADS, D, hh, bb, 0, 0, 0};
                    addOp(OP_SPLIT_HEADS, dAttnM, nullptr, nullptr, nullptr, nullptr, dAttnhS, sp, true);
                    addOp(OP_MATMUL, dAttnhS, b.vh[bb][hh], nullptr, nullptr, nullptr, dPS,
                          {0, 0, 0, 0, 1}, true);
                    addOp(OP_SOFTMAX_BWD, b.p[bb][hh], dPS, nullptr, nullptr, nullptr, dSS, {0, 0}, true);
                    addOp(OP_MATMUL, dSS, b.kh[bb][hh], nullptr, nullptr, nullptr, dQhS,
                          {0, 0, 0, 0, 0}, true);
                    addOp(OP_MATMUL, dSS, b.qh[bb][hh], nullptr, nullptr, nullptr, dKhS,
                          {0, 0, 0, 1, 0}, true);
                    addOp(OP_MATMUL, b.p[bb][hh], dAttnhS, nullptr, nullptr, nullptr, dVhS,
                          {0, 0, 0, 1, 0}, true);
                    addOp(OP_MERGE_HEADS, dQhS, nullptr, nullptr, nullptr, nullptr, dQ, sp, true);
                    addOp(OP_MERGE_HEADS, dKhS, nullptr, nullptr, nullptr, nullptr, dK, sp, true);
                    addOp(OP_MERGE_HEADS, dVhS, nullptr, nullptr, nullptr, nullptr, dV, sp, true);
                }
            }
            addBlBwd(dQ, b.wqQ, b.swQ, b.Xq, b.sx, ones, b.M1q, b.M2q, b.dQW, dN1, 0);
            addBlBwd(dK, b.wqK, b.swK, b.Xq, b.sx, ones, b.M1k, b.M2k, b.dKW, dN1, 1);
            addBlBwd(dV, b.wqV, b.swV, b.Xq, b.sx, ones, b.M1v, b.M2v, b.dVW, dN1, 1);
            addOp(OP_RMS_NORM_BWD, in, b.attnNormW, dN1, dwP, nullptr, dXa,
                  {0, 0, f32bits_i(LN_EPS)}, true);
            addOp(OP_REDUCE_SUM_ROWS, dwP, nullptr, nullptr, nullptr, nullptr, b.dAttnNormW, {0, 0}, true);
            addOp(OP_ADD, dXres, dXa, nullptr, nullptr, nullptr, dXout, {0, 0}, true);
            dHcur = dXout;
        }
        addOp(OP_ADD_POS_BWD, dHcur, dXout, dPosEmbW, nullptr, nullptr, nullptr,
              {0, 0, SEQ}, true);
        addOp(OP_EMBEDDING_BWD, ids, dHcur, nullptr, nullptr, nullptr, dTokEmbW,
              {0, 0, V}, true);
    }

    bool init(const CPUModel& cpu) {
        allocParams();
        allocAll();
        build();
        copyFromCPU(cpu);
        return true;
    }

    void copyFromCPU(const CPUModel& cpu) {
        auto cp = [&](Tensor* t, const std::vector<float>& src) {
            t->data = src;
            t->dirtyUpload = true;
        };
        cp(tokEmbW, cpu.tokEmb);
        cp(posEmbW, cpu.posEmb);
        cp(finalNormW, cpu.finalNormW);
        cp(lmHeadW, cpu.lmHeadW);
        for (int l = 0; l < BLOCKS; l++) {
            cp(blk[l].attnNormW, cpu.b[l].attnNormW);
            cp(blk[l].qW, cpu.b[l].qW);
            cp(blk[l].kW, cpu.b[l].kW);
            cp(blk[l].vW, cpu.b[l].vW);
            cp(blk[l].oW, cpu.b[l].oW);
            cp(blk[l].ffnNormW, cpu.b[l].ffnNormW);
            cp(blk[l].fUpW, cpu.b[l].fUpW);
            cp(blk[l].fDnW, cpu.b[l].fDnW);
        }
    }

    void copyToCPU(CPUModel& cpu) {
        auto cp = [&](Tensor* t, std::vector<float>& dst) {
            if (t->buf != VK_NULL_HANDLE && t->bufSize > 0)
                ctx.downloadBuffer(t->buf, t->bufSize, t->data.data());
            dst = t->data;
        };
        cp(tokEmbW, cpu.tokEmb);
        cp(posEmbW, cpu.posEmb);
        cp(finalNormW, cpu.finalNormW);
        cp(lmHeadW, cpu.lmHeadW);
        for (int l = 0; l < BLOCKS; l++) {
            cp(blk[l].attnNormW, cpu.b[l].attnNormW);
            cp(blk[l].qW, cpu.b[l].qW);
            cp(blk[l].kW, cpu.b[l].kW);
            cp(blk[l].vW, cpu.b[l].vW);
            cp(blk[l].oW, cpu.b[l].oW);
            cp(blk[l].ffnNormW, cpu.b[l].ffnNormW);
            cp(blk[l].fUpW, cpu.b[l].fUpW);
            cp(blk[l].fDnW, cpu.b[l].fDnW);
        }
    }

    float gradMaxDiff(const CPUModel& cpu) const {
        float md = 0.0f;
        auto cmp = [&](const char* name, Tensor* t, const std::vector<float>& r) {
            size_t k = std::min(t->data.size(), r.size());
            float d = 0.0f;
            for (size_t i = 0; i < k; i++) d = std::max(d, std::fabs(t->data[i] - r[i]));
            if (d > md) md = d;
            if (d > 1e-3f) std::cout << "  grad " << name << " diff=" << d << "\n";
        };
        cmp("tokEmb", dTokEmbW, cpu.gTokEmb);
        {
            auto it = [&](const char* name, const std::vector<float>& g, const std::vector<float>& c) {
                size_t k = std::min(g.size(), c.size());
                float d = 0.0f;
                size_t bi = 0;
                for (size_t i = 0; i < k; i++) {
                    float dd = std::fabs(g[i] - c[i]);
                    if (dd > d) { d = dd; bi = i; }
                }
                if (d > 1e-6f)
                    std::cout << "  it " << name << " d=" << d << " [" << (bi / E) << "," << (bi % E)
                              << "]cpu=" << c[bi] << " gpu=" << g[bi] << "\n";
            };
            it("dZ", dZ->data, cpu.dZ);
            it("x0", x0->data, cpu.x);
            it("h0", blk[0].h->data, cpu.fwdH0);
            it("h1", blk[1].h->data, cpu.h);
            it("z", z->data, cpu.z);
            it("dLogits", dLogits->data, cpu.dLogitsC);
            it("zq", zq->data, cpu.lmXq);
            it("sz", sz->data, cpu.lmSx);
            it("wqLM", wqLM->data, cpu.lmWq);
            it("swLM", swLM->data, cpu.lmSw);
            it("dG", dG->data, cpu.dG);
            it("dB", dB->data, cpu.dB);
            it("dH1b", dH1b->data, cpu.dH1b);
            it("dXres", dXres->data, cpu.dXres);
            it("dAttnM", dAttnM->data, cpu.dAttnM);
            it("dN1", dN1->data, cpu.dN1);
            it("dXa", dXa->data, cpu.dXa);
            it("dHout", dXout->data, cpu.dHcur);
            std::cout.flush();
        }
        {
            std::vector<float> ref((size_t)V * E, 0.0f);
            for (size_t r = 0; r < (size_t)n; r++) {
                int id = (int)ids->data[r];
                for (int j = 0; j < E; j++) ref[(size_t)id * E + j] += dHcur->data[r * E + j];
            }
            float d = 0.0f;
            for (size_t i = 0; i < ref.size(); i++)
                d = std::max(d, std::fabs(ref[i] - dTokEmbW->data[i]));
            std::cout << "  refTokEmb-vs-gpu d=" << d << "\n";
            std::cout.flush();
        }
        {
            int bad = 0;
            size_t k = std::min(dHcur->data.size(), cpu.dHcur.size());
            for (size_t i = 0; i < k; i++)
                if (std::fabs(dHcur->data[i] - cpu.dHcur[i]) > 1e-7f) bad++;
            std::cout << "  dH bad=" << bad << " of " << k;
            int shown = 0;
            for (size_t i = 0; i < k && shown < 5; i++) {
                float dd = std::fabs(dHcur->data[i] - cpu.dHcur[i]);
                if (dd > 1e-7f) {
                    std::cout << " [" << (i / E) << "," << (i % E) << "]cpu=" << cpu.dHcur[i]
                              << " gpu=" << dHcur->data[i];
                    shown++;
                }
            }
            std::cout << "\n";
            std::cout.flush();
        }
        cmp("posEmb", dPosEmbW, cpu.gPosEmb);
        cmp("finalNormW", dFinalNormW, cpu.gFinalNormW);
        cmp("lmHeadW", dLmHeadW, cpu.gLmHeadW);
        for (int l = 0; l < BLOCKS; l++) {
            std::string p = "blk" + std::to_string(l) + ".";
            cmp((p + "attnNormW").c_str(), blk[l].dAttnNormW, cpu.gb[l].attnNormW);
            cmp((p + "qW").c_str(), blk[l].dQW, cpu.gb[l].qW);
            cmp((p + "kW").c_str(), blk[l].dKW, cpu.gb[l].kW);
            cmp((p + "vW").c_str(), blk[l].dVW, cpu.gb[l].vW);
            cmp((p + "oW").c_str(), blk[l].dOW, cpu.gb[l].oW);
            cmp((p + "ffnNormW").c_str(), blk[l].dFfnNormW, cpu.gb[l].ffnNormW);
            cmp((p + "fUpW").c_str(), blk[l].dFuW, cpu.gb[l].fUpW);
            cmp((p + "fDnW").c_str(), blk[l].dFdW, cpu.gb[l].fDnW);
        }
        std::cout.flush();
        return md;
    }

    void applyAdam(int stepNum) {
        uint32_t pc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        pc[1] = (uint32_t)f32bits_i(LR);
        pc[2] = (uint32_t)stepNum;
        for (size_t i = 0; i < params.size(); i++) {
            size_t sz = (size_t)params[i]->n * (size_t)params[i]->cols;
            pc[0] = (uint32_t)sz;
            std::vector<VkBuffer> bufs = {params[i]->buf, m[i]->buf, v[i]->buf, grads[i]->buf};
            uint32_t w = (uint32_t)((sz + 63) / 64);
            ctx.runCompute("adam", w, 1, 1, bufs, pc, 32);
        }
    }

    void loadBatch(const std::vector<float>& tX, const std::vector<float>& tY, int feat,
                   const std::vector<int>& idx, int baseRow) {
        for (int r = 0; r < BATCH; r++) {
            int row = idx[baseRow + r];
            for (int t = 0; t < BYTES; t++) ids->data[r * SEQ + t] = tX[(size_t)row * feat + t];
            ids->data[r * SEQ + BYTES] = (float)SEP_ID;
            for (int t = 0; t < BYTES; t++) targets->data[r * SEQ + t] = ids->data[r * SEQ + t + 1];
            targets->data[r * SEQ + BYTES] = 256.0f + tY[row];
        }
        ids->dirtyUpload = true;
        targets->dirtyUpload = true;
    }

    bool trainStep(CPUModel& cpu, const std::vector<float>& tX, const std::vector<float>& tY,
                   int feat, const std::vector<int>& idx, int baseRow, int stepNum) {
        loadBatch(tX, tY, feat, idx, baseRow);
        g.forward();
        float gpuLoss = loss->data[0];

        copyToCPU(cpu);
        cpu.ids = ids->data;
        cpu.targets = targets->data;
        cpu.forward();
        float cpuLoss = cpu.ceLoss();
        cpu.backward();

        g.backward();
        float lossDiff = std::fabs(gpuLoss - cpuLoss);
        float gradDiff = gradMaxDiff(cpu);

        bool ok = lossDiff < LOSS_TOL && gradDiff < GRAD_TOL;
        if (ok) applyAdam(stepNum);

        std::cout << "step=" << stepNum << " train_loss=" << gpuLoss
                  << " loss_diff=" << lossDiff << " grad_diff=" << gradDiff
                  << (ok ? " ORACLE_PASS" : " ORACLE_FAIL") << "\n";
        std::cout.flush();
        return ok;
    }

    float eval(const std::vector<float>& eX, const std::vector<float>& eY, int feat, int eN) {
        int correct = 0;
        int nb = eN / BATCH;
        for (int bi = 0; bi < nb; bi++) {
            for (int r = 0; r < BATCH; r++) {
                int row = bi * BATCH + r;
                for (int t = 0; t < BYTES; t++) ids->data[r * SEQ + t] = eX[(size_t)row * feat + t];
                ids->data[r * SEQ + BYTES] = (float)SEP_ID;
            }
            ids->dirtyUpload = true;
            g.forward();
            for (int r = 0; r < BATCH; r++) {
                int row = bi * BATCH + r;
                size_t base = ((size_t)r * SEQ + BYTES) * (size_t)V + 256;
                int best = 0;
                for (int c = 1; c < NCLS; c++)
                    if (logits->data[base + c] > logits->data[base + best]) best = c;
                if (best == (int)eY[row]) correct++;
            }
        }
        return (float)correct / (float)(nb * BATCH);
    }

    bool exportGGUF(const std::string& path) {
        gguf::GGUFMeta meta;
        meta.strings.push_back({"general.name", "vulcan-bitnet-gpt2"});
        meta.u32s.push_back({"bitnet.block_count", (uint32_t)BLOCKS});
        meta.u32s.push_back({"bitnet.context_length", (uint32_t)SEQ});
        meta.u32s.push_back({"bitnet.embedding_length", (uint32_t)E});
        meta.u32s.push_back({"bitnet.feed_forward_length", (uint32_t)FF});
        meta.u32s.push_back({"bitnet.attention.head_count", (uint32_t)HEADS});
        meta.u32s.push_back({"bitnet.attention.head_count_kv", (uint32_t)HEADS});
        meta.f32s.push_back({"bitnet.attention.layer_norm_epsilon", LN_EPS});

        std::vector<gguf::TensorInfo> tensors;
        auto addT = [&](const std::string& name, Tensor* t, std::vector<uint32_t> dims) {
            gguf::TensorInfo ti;
            ti.name = name;
            ti.dims = dims;
            ti.data = t->data;
            tensors.push_back(ti);
        };
        addT("token_embd.weight", tokEmbW, {(uint32_t)E, (uint32_t)V});
        addT("output_norm.weight", finalNormW, {(uint32_t)E});
        addT("output.weight", lmHeadW, {(uint32_t)E, (uint32_t)V});
        for (int l = 0; l < BLOCKS; l++) {
            std::string p = "blk." + std::to_string(l) + ".";
            addT(p + "attn_norm.weight", blk[l].attnNormW, {(uint32_t)E});
            addT(p + "attn_q.weight", blk[l].qW, {(uint32_t)E, (uint32_t)E});
            addT(p + "attn_k.weight", blk[l].kW, {(uint32_t)E, (uint32_t)E});
            addT(p + "attn_v.weight", blk[l].vW, {(uint32_t)E, (uint32_t)E});
            addT(p + "attn_output.weight", blk[l].oW, {(uint32_t)E, (uint32_t)E});
            addT(p + "ffn_norm.weight", blk[l].ffnNormW, {(uint32_t)E});
            addT(p + "ffn_up.weight", blk[l].fUpW, {(uint32_t)E, (uint32_t)FF});
            addT(p + "ffn_down.weight", blk[l].fDnW, {(uint32_t)FF, (uint32_t)E});
        }
        std::string err;
        if (!gguf::writeGGUF(path, "bitnet", meta, tensors, &err)) {
            std::cerr << err << "\n";
            return false;
        }
        std::cout << "exported " << path << "\n";
        return true;
    }
};

int main(int argc, char** argv) {
    int trainUse = 6000, epochs = 5, seed = 42;
    std::string base = "mnist";
    if (argc > 1) trainUse = std::atoi(argv[1]);
    if (argc > 2) epochs = std::atoi(argv[2]);
    if (argc > 3) seed = std::atoi(argv[3]);
    if (argc > 4) base = argv[4];

    std::string trainPath = "vulcan/data/" + base + "_train.bin";
    std::string testPath = "vulcan/data/" + base + "_test.bin";
    int tN = 0, tF = 0;
    std::vector<float> tX, tY;
    std::string err;
    if (!ctx::loadBin(trainPath, tN, tF, tX, tY, &err)) {
        std::cerr << err << "\n";
        return 1;
    }
    if (tF < BYTES) {
        std::cerr << "train feat " << tF << " < " << BYTES << "\n";
        return 1;
    }
    int eN = 0, eF = 0;
    std::vector<float> eX, eY;
    if (!ctx::loadBin(testPath, eN, eF, eX, eY, &err)) {
        std::cerr << err << "\n";
        return 1;
    }
    int NCLS = 0;
    for (int i = 0; i < tN; i++) NCLS = std::max(NCLS, (int)tY[i]);
    NCLS += 1;
    trainUse = std::min(trainUse, tN);
    int SEP = 256 + NCLS;
    int V = 256 + NCLS + 1;

    std::cout << "vulcan_bitnet_gpt base=" << base << " trainUse=" << trainUse
              << " epochs=" << epochs << " NCLS=" << NCLS << " V=" << V << "\n";
    std::cout.flush();

    std::mt19937 rng(seed);
    CPUModel cpu;
    cpu.initRandom(NCLS, rng);

    GPUModel m;
    m.NCLS = NCLS;
    m.V = V;
    m.n = BATCH * SEQ;
    m.SEP_ID = SEP;
    m.g.ctx = &m.ctx;
    if (!m.ctx.init(findSPVDir())) {
        std::cerr << "vulkan init failed\n";
        return 1;
    }
    std::cout << "vulkan ok\n";
    std::cout.flush();
    m.init(cpu);
    std::cout << "model init ok\n";
    std::cout.flush();

    std::vector<int> idx(tN);
    for (int i = 0; i < tN; i++) idx[i] = i;

    for (int epoch = 0; epoch < epochs; epoch++) {
        std::shuffle(idx.begin(), idx.begin() + trainUse, rng);
        for (int step = 0; step < trainUse / BATCH; step++) {
            int stepNum = epoch * (trainUse / BATCH) + step + 1;
            bool ok = m.trainStep(cpu, tX, tY, tF, idx, step * BATCH, stepNum);
            if (!ok) {
                std::cerr << "ORACLE MISMATCH - aborting (step " << stepNum << ")\n";
                return 1;
            }
        }
    }

    float acc = m.eval(eX, eY, eF, eN);
    std::cout << "test_accuracy=" << acc * 100.0f << "%\n";
    m.exportGGUF("vulcan/data/" + base + "-bitnet-gpt2-f32.gguf");
    m.ctx.shutdown();
    return 0;
}
