/*
 * Dynamic per-token FP8 (e4m3) quantization.
 *   For each row (token): scale = max(|x|) / 448;  q = round(x / scale) as fp8_e4m3.
 * Self-contained: no flashinfer / cutlass. bf16 or fp16 input.
 *
 *   input:    [num_tokens, hidden_dim]  bf16/fp16
 *   output_q: [num_tokens, hidden_dim]  fp8_e4m3 (torch.float8_e4m3fn)
 *   output_s: [num_tokens]              float32 (per-token scale)
 */
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "joyomni_ops.h"

namespace joyomni_ops {
namespace {

constexpr float kFP8E4M3Max = 448.0f;
constexpr int kWarpSize = 32;

__device__ __forceinline__ float warpReduceMax(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o, 32));
  return v;
}

__device__ __forceinline__ float blockReduceMax(float v) {
  __shared__ float sh[32];
  int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  v = warpReduceMax(v);
  if (lane == 0) sh[wid] = v;
  __syncthreads();
  int nwarps = (blockDim.x + 31) / 32;
  v = (lane < nwarps) ? sh[lane] : 0.0f;
  if (wid == 0) v = warpReduceMax(v);
  return v;
}

// One warp per token; kTokensPerCTA warps per block. Vectorized by kVec.
template <typename T, int kTokensPerCTA, int kVec>
__global__ void perTokenQuantWarp(
    const T* __restrict__ input, __nv_fp8_e4m3* __restrict__ out_q, float* __restrict__ out_s,
    const int64_t hidden_dim, const int64_t num_tokens) {
  const int warp_id = threadIdx.x / kWarpSize;
  const int lane_id = threadIdx.x & (kWarpSize - 1);
  const int token_id = blockIdx.x * kTokensPerCTA + warp_id;
  if (token_id >= num_tokens) return;

  const T* trow = input + token_id * hidden_dim;
  __nv_fp8_e4m3* orow = out_q + token_id * hidden_dim;
  const int num_vec = hidden_dim / kVec;

  float max_value = 0.f;
  for (int i = lane_id; i < num_vec; i += kWarpSize) {
    const T* p = trow + i * kVec;
#pragma unroll
    for (int j = 0; j < kVec; ++j) max_value = fmaxf(max_value, fabsf(static_cast<float>(p[j])));
  }
  float warp_max = warpReduceMax(max_value);
  float scale = warp_max / kFP8E4M3Max;
  if (lane_id == 0) out_s[token_id] = scale;
  float scale_inv = (scale == 0.f) ? 0.f : 1.0f / scale;

  for (int i = lane_id; i < num_vec; i += kWarpSize) {
    const T* p = trow + i * kVec;
    __nv_fp8_e4m3 tmp[kVec];
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      float val = static_cast<float>(p[j]) * scale_inv;
      val = fmaxf(fminf(val, kFP8E4M3Max), -kFP8E4M3Max);
      tmp[j] = static_cast<__nv_fp8_e4m3>(val);
    }
    if constexpr (kVec == 16) {
      *reinterpret_cast<uint4*>(orow + i * kVec) = *reinterpret_cast<uint4*>(tmp);
    } else {
#pragma unroll
      for (int j = 0; j < kVec; ++j) orow[i * kVec + j] = tmp[j];
    }
  }
}

// One block per token (small batch); block reduce.
template <typename T, int kVec>
__global__ void perTokenQuantBlock(
    const T* __restrict__ input, __nv_fp8_e4m3* __restrict__ out_q, float* __restrict__ out_s,
    const int64_t hidden_dim, const int64_t num_tokens) {
  const int token_idx = blockIdx.x;
  if (token_idx >= num_tokens) return;
  const int tid = threadIdx.x;
  const T* trow = input + token_idx * hidden_dim;
  __nv_fp8_e4m3* orow = out_q + token_idx * hidden_dim;
  const int num_vec = hidden_dim / kVec;

  float max_value = 0.f;
  for (int i = tid; i < num_vec; i += blockDim.x) {
    const T* p = trow + i * kVec;
#pragma unroll
    for (int j = 0; j < kVec; ++j) max_value = fmaxf(max_value, fabsf(static_cast<float>(p[j])));
  }
  max_value = blockReduceMax(max_value);
  __shared__ float scale;
  if (tid == 0) {
    scale = max_value / kFP8E4M3Max;
    out_s[token_idx] = scale;
  }
  __syncthreads();
  float scale_inv = (scale == 0.f) ? 0.f : 1.0f / scale;

  for (int i = tid; i < num_vec; i += blockDim.x) {
    const T* p = trow + i * kVec;
    __nv_fp8_e4m3 tmp[kVec];
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      float val = static_cast<float>(p[j]) * scale_inv;
      val = fmaxf(fminf(val, kFP8E4M3Max), -kFP8E4M3Max);
      tmp[j] = static_cast<__nv_fp8_e4m3>(val);
    }
    if constexpr (kVec == 16) {
      *reinterpret_cast<uint4*>(orow + i * kVec) = *reinterpret_cast<uint4*>(tmp);
    } else {
#pragma unroll
      for (int j = 0; j < kVec; ++j) orow[i * kVec + j] = tmp[j];
    }
  }
}

template <typename T>
void dispatch_vec(bool warp, dim3 grid, dim3 block, cudaStream_t stream, const T* in,
                  __nv_fp8_e4m3* q, float* s, int64_t hd, int64_t nt) {
  constexpr int TPC = 8;
  bool v16 = (hd % 16 == 0), v8 = (hd % 8 == 0);
  if (warp) {
    if (v16) perTokenQuantWarp<T, TPC, 16><<<grid, block, 0, stream>>>(in, q, s, hd, nt);
    else if (v8) perTokenQuantWarp<T, TPC, 8><<<grid, block, 0, stream>>>(in, q, s, hd, nt);
    else perTokenQuantWarp<T, TPC, 4><<<grid, block, 0, stream>>>(in, q, s, hd, nt);
  } else {
    if (v16) perTokenQuantBlock<T, 16><<<grid, block, 0, stream>>>(in, q, s, hd, nt);
    else if (v8) perTokenQuantBlock<T, 8><<<grid, block, 0, stream>>>(in, q, s, hd, nt);
    else perTokenQuantBlock<T, 4><<<grid, block, 0, stream>>>(in, q, s, hd, nt);
  }
}

}  // namespace

void sgl_per_token_quant_fp8(torch::Tensor input, torch::Tensor output_q, torch::Tensor output_s) {
  JO_CHECK_CUDA(input);
  JO_CHECK_CONTIGUOUS(input);
  JO_CHECK_CUDA(output_q);
  JO_CHECK_CUDA(output_s);
  TORCH_CHECK(input.dim() == 2, "input must be 2D [num_tokens, hidden_dim]");
  const int64_t num_tokens = input.size(0);
  const int64_t hidden_dim = input.size(1);
  TORCH_CHECK(hidden_dim % 4 == 0, "hidden_dim must be divisible by 4, got ", hidden_dim);

  const c10::cuda::CUDAGuard guard(input.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  constexpr int TPC = 8;
  const bool warp = (num_tokens >= sm_count * 2 * TPC);
  dim3 grid = warp ? dim3((num_tokens + TPC - 1) / TPC) : dim3((unsigned)num_tokens);
  dim3 block(256);

  auto* q = static_cast<__nv_fp8_e4m3*>(output_q.data_ptr());
  auto* s = static_cast<float*>(output_s.data_ptr());
  if (input.scalar_type() == torch::kBFloat16)
    dispatch_vec<__nv_bfloat16>(warp, grid, block, stream,
                                static_cast<const __nv_bfloat16*>(input.data_ptr()), q, s, hidden_dim, num_tokens);
  else if (input.scalar_type() == torch::kFloat16)
    dispatch_vec<__half>(warp, grid, block, stream,
                         static_cast<const __half*>(input.data_ptr()), q, s, hidden_dim, num_tokens);
  else
    TORCH_CHECK(false, "per_token_quant_fp8: input must be bf16 or fp16");
}

}  // namespace joyomni_ops
