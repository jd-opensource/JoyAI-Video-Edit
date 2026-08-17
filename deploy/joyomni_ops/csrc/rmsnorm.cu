/*
 * Standalone RMSNorm:  out[i] = (x[i] * rsqrt(mean(x^2) + eps)) * weight[i]
 * Self-written (no flashinfer). One block per row, warp/block reduction.
 * Supports bf16 / fp16 / fp32. x: [M, N] row-contiguous; weight: [N].
 */
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>

#include "joyomni_ops.h"

namespace joyomni_ops {
namespace {

__device__ __forceinline__ float warpSum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ float blockSum(float v) {
  __shared__ float sh[32];
  int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  v = warpSum(v);
  if (lane == 0) sh[wid] = v;
  __syncthreads();
  int nwarps = (blockDim.x + 31) / 32;
  v = (threadIdx.x < nwarps) ? sh[lane] : 0.0f;
  if (wid == 0) v = warpSum(v);
  return v;  // valid on thread 0
}

template <typename T>
__global__ void rmsnormKernel(
    T* __restrict__ out, const T* __restrict__ x, const T* __restrict__ weight,
    const int n, const float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  x += (size_t)row * n;
  out += (size_t)row * n;

  float local = 0.0f;
  for (int i = tid; i < n; i += blockDim.x) {
    float v = static_cast<float>(x[i]);
    local += v * v;
  }
  local = blockSum(local);
  __shared__ float s_rms;
  if (tid == 0) s_rms = rsqrtf(local / n + eps);
  __syncthreads();
  const float rms = s_rms;

  for (int i = tid; i < n; i += blockDim.x) {
    float v = static_cast<float>(x[i]) * rms * static_cast<float>(weight[i]);
    out[i] = T(v);
  }
}

template <typename T>
void launch(const torch::Tensor& x, const torch::Tensor& weight, torch::Tensor& out, float eps) {
  const int64_t M = x.size(0), N = x.size(1);
  int block = (int)std::min<int64_t>(1024, ((N + 31) / 32) * 32);
  if (block < 32) block = 32;
  dim3 grid((unsigned)M);
  auto stream = at::cuda::getCurrentCUDAStream();
  rmsnormKernel<T><<<grid, block, 0, stream>>>(
      (T*)out.data_ptr(), (const T*)x.data_ptr(), (const T*)weight.data_ptr(), (int)N, eps);
}

}  // namespace

torch::Tensor rmsnorm(const torch::Tensor& x, const torch::Tensor& weight, double eps) {
  JO_CHECK_CUDA(x);
  JO_CHECK_CUDA(weight);
  TORCH_CHECK(x.dim() == 2, "x must be 2D [M, N]");
  TORCH_CHECK(x.stride(-1) == 1, "x last dim must be contiguous");
  TORCH_CHECK(weight.dim() == 1 && weight.numel() == x.size(1), "weight must be 1D [N]");
  TORCH_CHECK(x.dtype() == weight.dtype(), "x and weight dtype must match");

  const c10::cuda::CUDAGuard guard(x.device());
  auto out = torch::empty_like(x);
  if (x.dtype() == torch::kFloat32)
    launch<float>(x, weight, out, (float)eps);
  else if (x.dtype() == torch::kFloat16)
    launch<__half>(x, weight, out, (float)eps);
  else if (x.dtype() == torch::kBFloat16)
    launch<__nv_bfloat16>(x, weight, out, (float)eps);
  else
    TORCH_CHECK(false, "Unsupported dtype; use float32/float16/bfloat16");
  return out;
}

}  // namespace joyomni_ops
