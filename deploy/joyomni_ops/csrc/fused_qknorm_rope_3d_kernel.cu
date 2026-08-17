/*
 * Fused QK-Norm + 3D RoPE kernel for DiT video models (bf16).
 * Applies per-head RMSNorm then interleaved RoPE using precomputed cos/sin.
 *
 * Adapted from the JoyOmni sgl-kernel fork (Apache-2.0, NVIDIA copyright below);
 * trimmed to the single `_paired` entry point actually used by the pipeline.
 *
 * Copyright (c) 2025, NVIDIA CORPORATION. Licensed under the Apache License 2.0.
 *
 * Tensor layout: q/k are [batch, seq_len * num_heads, head_dim] bf16.
 * cos/sin: [seq_len, head_dim/2] bf16.  weight: [head_dim] bf16.
 */
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include "joyomni_ops.h"

namespace joyomni_ops {
namespace {

constexpr unsigned kFullMask = 0xffffffffu;

template <typename T, int num>
struct packed_as;
template <>
struct packed_as<unsigned, 1> {
  using type = unsigned;
};
template <>
struct packed_as<unsigned, 2> {
  using type = uint2;
};
template <>
struct packed_as<unsigned, 4> {
  using type = uint4;
};

__inline__ __device__ float warpReduceSum(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(kFullMask, val, mask, 32);
  return val;
}

template <typename T>
inline __device__ __host__ T divUp(T m, T n) {
  return (m + n - 1) / n;
}

// One warp normalizes+rotates one head. Q and K are packed into a single grid:
// warps [0, N) handle Q, warps [N, 2N) handle K, where N = batch*seq*heads.
template <int head_dim>
__global__ void fusedQKNormRope3DPaired(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    int const batch_size,
    int const seq_len,
    int const num_heads,
    float const eps,
    __nv_bfloat16 const* __restrict__ q_weight,
    __nv_bfloat16 const* __restrict__ k_weight,
    __nv_bfloat16 const* __restrict__ cos_ptr,
    __nv_bfloat16 const* __restrict__ sin_ptr) {
  int const warpsPerBlock = blockDim.x / 32;
  int const warpId = threadIdx.x / 32;
  int const laneId = threadIdx.x % 32;
  int const globalWarpIdx = blockIdx.x * warpsPerBlock + warpId;

  int const total_qk_heads = batch_size * seq_len * num_heads * 2;
  if (globalWarpIdx >= total_qk_heads) return;

  int const is_k = globalWarpIdx / (batch_size * seq_len * num_heads);
  int const local_idx = globalWarpIdx % (batch_size * seq_len * num_heads);
  int const batch_idx = local_idx / (seq_len * num_heads);
  int const remaining = local_idx % (seq_len * num_heads);
  int const token_idx = remaining / num_heads;
  int const head_idx = remaining % num_heads;

  static_assert(head_dim % (32 * 2) == 0, "head_dim must be divisible by 64");
  constexpr int numElemsPerThread = head_dim / 32;
  float elements[numElemsPerThread];
  constexpr int elemSizeBytes = numElemsPerThread * sizeof(__nv_bfloat16);
  static_assert(elemSizeBytes % 4 == 0, "elemSizeBytes must be a multiple of 4");
  constexpr int vecSize = elemSizeBytes / 4;
  using vec_T = typename packed_as<unsigned, vecSize>::type;

  __nv_bfloat16* qk = is_k ? k : q;
  __nv_bfloat16 const* weight = is_k ? k_weight : q_weight;

  int const offsetWarp = batch_idx * seq_len * num_heads * head_dim +
                         (token_idx * num_heads + head_idx) * head_dim;
  int const offsetThread = offsetWarp + laneId * numElemsPerThread;

  float sumOfSquares = 0.0f;
  {
    vec_T vec = *reinterpret_cast<vec_T const*>(&qk[offsetThread]);
#pragma unroll
    for (int i = 0; i < vecSize; i++) {
      float2 vals = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<unsigned*>(&vec) + i));
      sumOfSquares += vals.x * vals.x;
      sumOfSquares += vals.y * vals.y;
      elements[2 * i] = vals.x;
      elements[2 * i + 1] = vals.y;
    }
  }

  sumOfSquares = warpReduceSum(sumOfSquares);
  float rms_rcp = rsqrtf(sumOfSquares / static_cast<float>(head_dim) + eps);

#pragma unroll
  for (int i = 0; i < numElemsPerThread; i++) {
    int dim = laneId * numElemsPerThread + i;
    elements[i] *= rms_rcp * __bfloat162float(weight[dim]);
  }

  int const cos_sin_row_offset = token_idx * (head_dim / 2);
  float rotated[numElemsPerThread];
#pragma unroll
  for (int i = 0; i < numElemsPerThread; i += 2) {
    int dim_idx = laneId * numElemsPerThread + i;
    int cos_sin_idx = dim_idx / 2;
    float c = __bfloat162float(cos_ptr[cos_sin_row_offset + cos_sin_idx]);
    float s = __bfloat162float(sin_ptr[cos_sin_row_offset + cos_sin_idx]);
    float x1 = elements[i];
    float x2 = elements[i + 1];
    rotated[i] = x1 * c - x2 * s;
    rotated[i + 1] = x1 * s + x2 * c;
  }

  {
    vec_T vec;
#pragma unroll
    for (int i = 0; i < vecSize; i++) {
      __nv_bfloat162 vals = __float22bfloat162_rn(make_float2(rotated[2 * i], rotated[2 * i + 1]));
      reinterpret_cast<__nv_bfloat162&>(*(reinterpret_cast<unsigned*>(&vec) + i)) = vals;
    }
    *reinterpret_cast<vec_T*>(&qk[offsetThread]) = vec;
  }
}

void launchPaired(
    void* q, void* k, int batch_size, int seq_len, int num_heads, int head_dim,
    float eps, void const* q_weight, void const* k_weight, void const* cos_ptr,
    void const* sin_ptr, cudaStream_t stream) {
  constexpr int blockSize = 256;
  int const warpsPerBlock = blockSize / 32;
  int const totalQKHeads = batch_size * seq_len * num_heads * 2;
  dim3 gridDim(divUp(totalQKHeads, warpsPerBlock));
  dim3 blockDim(blockSize);
  auto q16 = reinterpret_cast<__nv_bfloat16*>(q);
  auto k16 = reinterpret_cast<__nv_bfloat16*>(k);
  auto qw = reinterpret_cast<__nv_bfloat16 const*>(q_weight);
  auto kw = reinterpret_cast<__nv_bfloat16 const*>(k_weight);
  auto cs = reinterpret_cast<__nv_bfloat16 const*>(cos_ptr);
  auto sn = reinterpret_cast<__nv_bfloat16 const*>(sin_ptr);
#define LAUNCH(HD)                                                                            \
  fusedQKNormRope3DPaired<HD><<<gridDim, blockDim, 0, stream>>>(                              \
      q16, k16, batch_size, seq_len, num_heads, eps, qw, kw, cs, sn);                         \
  break
  switch (head_dim) {
    case 64:
      LAUNCH(64);
    case 128:
      LAUNCH(128);
    case 256:
      LAUNCH(256);
    default:
      TORCH_CHECK(false, "Unsupported head_dim for fused_qk_norm_rope_3d_paired: ", head_dim);
  }
#undef LAUNCH
}

}  // namespace

void fused_qk_norm_rope_3d_paired(
    torch::Tensor& q, torch::Tensor& k, int64_t seq_len, int64_t num_heads,
    double eps, torch::Tensor& q_weight, torch::Tensor& k_weight,
    torch::Tensor& cos, torch::Tensor& sin) {
  TORCH_CHECK(q.dim() == 3, "q must be 3D [batch, seq_len*num_heads, head_dim]");
  TORCH_CHECK(k.dim() == 3, "k must be 3D [batch, seq_len*num_heads, head_dim]");
  TORCH_CHECK(q.sizes() == k.sizes(), "q and k must have the same shape");
  TORCH_CHECK(q_weight.dim() == 1 && k_weight.dim() == 1, "weights must be 1D [head_dim]");
  TORCH_CHECK(cos.dim() == 2 && sin.dim() == 2, "cos/sin must be 2D [seq_len, head_dim/2]");
  JO_CHECK_INPUT(q, torch::kBFloat16);
  JO_CHECK_INPUT(k, torch::kBFloat16);
  JO_CHECK_INPUT(q_weight, torch::kBFloat16);
  JO_CHECK_INPUT(k_weight, torch::kBFloat16);
  JO_CHECK_INPUT(cos, torch::kBFloat16);
  JO_CHECK_INPUT(sin, torch::kBFloat16);

  int64_t batch_size = q.size(0);
  int64_t head_dim = q.size(2);
  TORCH_CHECK(q.size(1) == seq_len * num_heads, "q.size(1) must equal seq_len*num_heads");
  TORCH_CHECK(q_weight.size(0) == head_dim && k_weight.size(0) == head_dim, "weight size must match head_dim");
  TORCH_CHECK(cos.size(0) == seq_len && cos.size(1) == head_dim / 2, "cos shape must be [seq_len, head_dim/2]");
  TORCH_CHECK(sin.size(0) == seq_len && sin.size(1) == head_dim / 2, "sin shape must be [seq_len, head_dim/2]");

  const c10::cuda::CUDAGuard guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
  launchPaired(
      q.data_ptr(), k.data_ptr(), static_cast<int>(batch_size), static_cast<int>(seq_len),
      static_cast<int>(num_heads), static_cast<int>(head_dim), static_cast<float>(eps),
      q_weight.data_ptr(), k_weight.data_ptr(), cos.data_ptr(), sin.data_ptr(), stream);
}

}  // namespace joyomni_ops
