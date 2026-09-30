#pragma once
#include "common.hpp"

constexpr int MMVQ_MAX_BATCH_SIZE = 8;
constexpr int MMVQ_MAX_ROUTED_BATCH = 64;

template<int mask>
static __device__ __forceinline__ float mmvq_xor_add_dpp(float x) {
  const int y = __builtin_amdgcn_update_dpp(
      0, __builtin_bit_cast(int, x), 0x160 | mask, 0xF, 0xF, false);
  return x + __builtin_bit_cast(float, y);
}

// Keep the original descending 32-lane XOR tree, including separate halves
// of wave64. The row exchange and four DPP adds avoid LDS permutes on gfx1151.
static __device__ __forceinline__ float mmvq_reduce_sum(float x) {
  const unsigned bits = __builtin_bit_cast(unsigned, x);
  x += __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(
                                    bits, bits, 0x76543210U, 0xfedcba98U,
                                    false, false));
  x = mmvq_xor_add_dpp<8>(x);
  x = mmvq_xor_add_dpp<4>(x);
  x = mmvq_xor_add_dpp<2>(x);
  return mmvq_xor_add_dpp<1>(x);
}

// One expert's independent inputs, in groups of eight. The anchor also
// identifies an inactive slot that must be zeroed.
struct MoeBatchGroup {
  int expert;
  int anchor;
  int token[MMVQ_MAX_BATCH_SIZE];
  uint32_t slots[MMVQ_MAX_BATCH_SIZE];
};

// Tuned routed-vector batch limits for gfx1151.
constexpr __host__ __device__ int mmvq_moe_max_batch(ggml_type type) {
    return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K ? 4 : 8;
}

void mul_mat_vec_q8_dispatch(const void* weights, const void* gate,
                            const block_q8_1* input, float* output,
                            int k, int rows, int tokens, int input_stride,
                            hipStream_t stream);
void mul_mat_vec_moe_dispatch(const void* weights, ggml_type type,
                             const block_q8_1* input, const int32_t* ids, float* output,
                             int k, int rows, int tokens, int experts_used,
                             int input_stride, hipStream_t stream);
void mul_mat_vec_moe_gated(const void* gate, const void* up, ggml_type type,
                           const block_q8_1* input, const int32_t* ids,
                           int32_t* groups, float* output, int k, int rows,
                           int tokens, int experts_used, int input_stride,
                           hipStream_t stream);

void mul_mat_vec_moe_batch_wave64(const void* gate, const void* up,
                                 const block_q8_1* input, const int32_t* groups,
                                 float* output, int k, int rows,
                                 int experts_used, int input_stride,
                                 hipStream_t stream);
