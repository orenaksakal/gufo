// Exact Q8 vector reduction with production dot products and cold weight rings.
// Build: tools/bench/build.sh tools/qwen-flash/q8_reduce_bench.hip
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq_prelude.h"

#include <cmath>
#include <stdexcept>

namespace qfn_mmq {
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/mmvq.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/vecdotq.hpp"
}  // namespace qfn_mmq

using qfn_mmq::block_q8_0;
using qfn_mmq::block_q8_1;

static void Check(hipError_t status) {
  if (status != hipSuccess)
    throw std::runtime_error(hipGetErrorString(status));
}

template<int tokens, bool dpp>
__launch_bounds__(32, 1) __global__ void Projection(
    const block_q8_0* __restrict__ weights,
    const block_q8_1* __restrict__ input, float* __restrict__ output,
    int rows, int k, int input_stride) {
  const int lane = threadIdx.x;
  const int blocks = k / 32;
  const int row = blockIdx.x;
  float sum[tokens]{};
  for (int kb = lane / 4; kb < blocks; kb += 8) {
#pragma unroll
    for (int t = 0; t < tokens; ++t)
      sum[t] += qfn_mmq::vec_dot_q8_0_q8_1(
          weights, input + t * input_stride + kb, row * blocks + kb,
          2 * (lane % 4));
  }
#pragma unroll
  for (int t = 0; t < tokens; ++t) {
    if constexpr (dpp)
      sum[t] = qfn_mmq::mmvq_reduce_sum(sum[t]);
    else
      sum[t] = qfn_mmq::warp_reduce_sum<32>(sum[t]);
    if (lane == 0)
      output[t * rows + row] = sum[t];
  }
}

static std::uint32_t Random(std::uint32_t& state) {
  state ^= state << 13;
  state ^= state >> 17;
  return state ^= state << 5;
}

template<int tokens = 1>
static void Run(const block_q8_0* weights, const block_q8_1* input,
                float* output, int rows, int k, int ring) {
  const std::size_t weight_stride = std::size_t(rows) * k / 32;
  const int input_stride = GGML_PAD(k, MATRIX_ROW_PADDING) / 32;
  std::vector<float> expected(rows * tokens), actual(rows * tokens);
  for (int slot = 0; slot < ring; ++slot) {
    Projection<tokens, false><<<rows, 32>>>(
        weights + slot * weight_stride, input, output, rows, k, input_stride);
    Check(hipMemcpy(expected.data(), output, expected.size() * sizeof(float),
                    hipMemcpyDeviceToHost));
    Projection<tokens, true><<<rows, 32>>>(
        weights + slot * weight_stride, input, output, rows, k, input_stride);
    Check(hipMemcpy(actual.data(), output, actual.size() * sizeof(float),
                    hipMemcpyDeviceToHost));
    if (std::memcmp(expected.data(), actual.data(),
                    actual.size() * sizeof(float)) ||
        !std::all_of(actual.begin(), actual.end(),
                     [](float x) { return std::isfinite(x); }))
      throw std::runtime_error("DPP changed Q8 projection bytes");
  }
  hipEvent_t start, end;
  Check(hipEventCreate(&start));
  Check(hipEventCreate(&end));
  std::array<std::vector<float>, 2> times;
  for (int repeat = 0; repeat < 5; ++repeat) {
    for (int arm = 0; arm < 2; ++arm) {
      const bool dpp = (arm + repeat) % 2;
      Check(hipEventRecord(start));
      constexpr int iterations = 32;
      for (int i = 0; i < iterations; ++i) {
        const auto* w = weights + (i % ring) * weight_stride;
        if (dpp)
          Projection<tokens, true><<<rows, 32>>>(w, input, output, rows, k,
                                                input_stride);
        else
          Projection<tokens, false><<<rows, 32>>>(w, input, output, rows, k,
                                                 input_stride);
      }
      Check(hipEventRecord(end));
      Check(hipEventSynchronize(end));
      float ms;
      Check(hipEventElapsedTime(&ms, start, end));
      times[dpp].push_back(ms * 1000 / iterations);
    }
  }
  for (auto& values : times)
    std::sort(values.begin(), values.end());
  std::printf("rows=%d k=%d tokens=%d ring=%d baseline_us=%.3f "
              "dpp_us=%.3f exact=1\n",
              rows, k, tokens, ring, times[0][2], times[1][2]);
  std::fflush(stdout);
  Check(hipEventDestroy(end));
  Check(hipEventDestroy(start));
  if constexpr (tokens < 8)
    Run<tokens + 1>(weights, input, output, rows, k, ring);
}

int main() {
  try {
    for (auto [rows, k] : {std::pair{10240, 320}, {2560, 6144},
                          {16384, 2560}, {248320, 2560}}) {
      const std::size_t per_matrix = std::size_t(rows) * k / 32;
      const std::size_t bytes = per_matrix * sizeof(block_q8_0);
      const int ring = int(((64ULL << 20) + bytes - 1) / bytes);
      std::vector<block_q8_0> weights(per_matrix * ring);
      std::vector<block_q8_1> input(8 * GGML_PAD(k, MATRIX_ROW_PADDING) / 32);
      std::uint32_t seed = 73;
      for (auto& block : weights) {
        block.d = __float2half((1 + Random(seed) % 127) * 0.0001F);
        for (auto& x : block.qs)
          x = static_cast<std::int8_t>(int(Random(seed) % 255) - 127);
      }
      for (auto& block : input) {
        block.ds = __floats2half2_rn((1 + Random(seed) % 127) * 0.0001F, 0);
        for (auto& x : block.qs)
          x = static_cast<std::int8_t>(int(Random(seed) % 255) - 127);
      }
      block_q8_0* dw = nullptr;
      block_q8_1* dx = nullptr;
      float* dy = nullptr;
      Check(hipMalloc(&dw, weights.size() * sizeof(block_q8_0)));
      Check(hipMalloc(&dx, input.size() * sizeof(block_q8_1)));
      Check(hipMalloc(&dy, 8ULL * rows * sizeof(float)));
      Check(hipMemcpy(dw, weights.data(), weights.size() * sizeof(block_q8_0),
                      hipMemcpyHostToDevice));
      Check(hipMemcpy(dx, input.data(), input.size() * sizeof(block_q8_1),
                      hipMemcpyHostToDevice));
      Run(dw, dx, dy, rows, k, ring);
      Check(hipFree(dy));
      Check(hipFree(dx));
      Check(hipFree(dw));
    }
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
