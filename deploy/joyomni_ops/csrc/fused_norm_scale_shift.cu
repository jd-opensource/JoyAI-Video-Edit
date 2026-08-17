/*
 * Fused (Layer|RMS)Norm + adaLN scale/shift, bf16/fp16/fp32.
 *   out = Norm(x; gamma, beta) * (1 + scale) + shift
 *
 * Trimmed from the JoyOmni sgl-kernel fork (SGLang Team, Apache-2.0) to the
 * single [M, N] per-row scale/shift path the pipeline uses; cutlass dependency
 * removed (native __nv_bfloat16 instead of cutlass::bfloat16_t).
 *
 * x:            [M, N]  (row-contiguous)
 * gamma/beta:   None or [N]   (affine; beta only for LayerNorm)
 * scale/shift:  [M, N]  (per row) — one modulation vector per token row
 * norm_type:    0 = LayerNorm, 1 = RMSNorm
 */
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "joyomni_ops.h"

namespace joyomni_ops {
namespace {

enum NormType : int { kLayerNorm = 0, kRMSNorm = 1 };

template <typename T, int NumVals>
__device__ __forceinline__ void warpReduceSum(T (&vals)[NumVals]) {
  unsigned mask = 0xffffffffu;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
#pragma unroll
    for (int i = 0; i < NumVals; ++i) vals[i] += __shfl_down_sync(mask, vals[i], offset);
}

template <typename T, int NumVals>
__device__ __forceinline__ void blockReduceSum(T (&vals)[NumVals]) {
  __shared__ T shared[32][NumVals];
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;
  warpReduceSum<T, NumVals>(vals);
  if (lane == 0)
#pragma unroll
    for (int i = 0; i < NumVals; ++i) shared[wid][i] = vals[i];
  __syncthreads();
  if (wid == 0) {
    T acc[NumVals];
#pragma unroll
    for (int i = 0; i < NumVals; ++i) acc[i] = T(0);
    int num_warps = (blockDim.x + 31) / 32;
#pragma unroll
    for (int w = 0; w < 32; ++w)
      if (w < num_warps)
#pragma unroll
        for (int i = 0; i < NumVals; ++i) acc[i] += shared[w][i];
#pragma unroll
    for (int i = 0; i < NumVals; ++i) vals[i] = acc[i];
  }
  __syncthreads();
}

// Vec-of-4 element types.
struct alignas(8) bf16_4 { __nv_bfloat16 x, y, z, w; };
struct alignas(8) half4 { __half x, y, z, w; };

template <typename T4_, typename T_>
struct DTypeTag { using T4 = T4_; using T = T_; };

// One block per row (m_idx). scale/shift indexed per-row [M, N].
template <typename T4, typename T, int ITEM_PER_THREAD, int norm_type>
__global__ void normScaleShiftPerRow(
    T4* output, const T4* input, const T4* gamma, const T4* beta,
    const T4* scale, const T4* shift, const int n, bool affine, float eps) {
  const int m_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int bdimx = blockDim.x;
  __shared__ float s_mean, s_variance;
  float local_sums[1] = {0.0f};
  T4 local_val[ITEM_PER_THREAD];
  const int n_4 = n / 4;
  const int offset = m_idx * n_4;
  input += offset;
  output += offset;
  scale += offset;  // per-row
  shift += offset;

  const T4 zero = {T(0.0f), T(0.0f), T(0.0f), T(0.0f)};
#pragma unroll
  for (int i = 0; i < ITEM_PER_THREAD; ++i) {
    const int index = i * bdimx + tid;
    local_val[i] = index < n_4 ? input[index] : zero;
    if constexpr (norm_type == kLayerNorm) {
      local_sums[0] += float(local_val[i].x) + float(local_val[i].y) + float(local_val[i].z) + float(local_val[i].w);
    } else {
      local_sums[0] += float(local_val[i].x) * float(local_val[i].x) + float(local_val[i].y) * float(local_val[i].y) +
                       float(local_val[i].z) * float(local_val[i].z) + float(local_val[i].w) * float(local_val[i].w);
    }
  }
  if (blockDim.x <= 32) warpReduceSum<float, 1>(local_sums);
  else blockReduceSum<float, 1>(local_sums);
  if (tid == 0) s_mean = local_sums[0] / n;
  __syncthreads();

  if constexpr (norm_type == kLayerNorm) {
    local_sums[0] = 0.0f;
#pragma unroll
    for (int i = 0; i < ITEM_PER_THREAD; ++i) {
      const int index = i * bdimx + tid;
      if (index < n_4) {
        float4 t = {float(local_val[i].x) - s_mean, float(local_val[i].y) - s_mean,
                    float(local_val[i].z) - s_mean, float(local_val[i].w) - s_mean};
        local_sums[0] += t.x * t.x + t.y * t.y + t.z * t.z + t.w * t.w;
      }
    }
    if (blockDim.x <= 32) warpReduceSum<float, 1>(local_sums);
    else blockReduceSum<float, 1>(local_sums);
  }
  if (tid == 0) s_variance = rsqrtf(local_sums[0] / n + eps);  // rms: rsqrt(mean(x^2)+eps)
  __syncthreads();

#pragma unroll
  for (int i = 0; i < ITEM_PER_THREAD; ++i) {
    const int index = i * bdimx + tid;
    if (index >= n_4) continue;
    const T4 g = affine ? gamma[index] : T4{T(1.f), T(1.f), T(1.f), T(1.f)};
    const T4 sc = scale[index];
    const T4 sh = shift[index];
    T4 out;
    if constexpr (norm_type == kLayerNorm) {
      const T4 b = affine ? beta[index] : T4{T(0.f), T(0.f), T(0.f), T(0.f)};
      out.x = T(((float(local_val[i].x) - s_mean) * s_variance * float(g.x) + float(b.x)) * (1.f + float(sc.x)) + float(sh.x));
      out.y = T(((float(local_val[i].y) - s_mean) * s_variance * float(g.y) + float(b.y)) * (1.f + float(sc.y)) + float(sh.y));
      out.z = T(((float(local_val[i].z) - s_mean) * s_variance * float(g.z) + float(b.z)) * (1.f + float(sc.z)) + float(sh.z));
      out.w = T(((float(local_val[i].w) - s_mean) * s_variance * float(g.w) + float(b.w)) * (1.f + float(sc.w)) + float(sh.w));
    } else {
      out.x = T((float(local_val[i].x) * s_variance * float(g.x)) * (1.f + float(sc.x)) + float(sh.x));
      out.y = T((float(local_val[i].y) * s_variance * float(g.y)) * (1.f + float(sc.y)) + float(sh.y));
      out.z = T((float(local_val[i].z) * s_variance * float(g.z)) * (1.f + float(sc.z)) + float(sh.z));
      out.w = T((float(local_val[i].w) * s_variance * float(g.w)) * (1.f + float(sc.w)) + float(sh.w));
    }
    output[index] = out;
  }
}

template <typename DT>
void launch(const torch::Tensor& x, void* gamma_ptr, void* beta_ptr, const torch::Tensor& scale,
            const torch::Tensor& shift, torch::Tensor& y, int norm_type, bool affine, float eps) {
  using T4 = typename DT::T4;
  using T = typename DT::T;
  const int64_t M = x.size(0), N = x.size(1);
  dim3 grid((unsigned)M);
  dim3 block;
  auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH(IPT, NT)                                                                        \
  normScaleShiftPerRow<T4, T, IPT, NT><<<grid, block, 0, stream>>>(                            \
      (T4*)y.data_ptr(), (const T4*)x.data_ptr(), (const T4*)gamma_ptr, (const T4*)beta_ptr,   \
      (const T4*)scale.data_ptr(), (const T4*)shift.data_ptr(), (int)N, affine, eps)
  if (N <= 4096) {
    block.x = (unsigned)((N / 4 + 31) / 32 * 32);
    if (block.x > 1024) block.x = 1024;
    if (norm_type == kLayerNorm) { LAUNCH(1, kLayerNorm); } else { LAUNCH(1, kRMSNorm); }
  } else {
    block.x = (unsigned)(((N + 7) / 8 + 31) / 32 * 32);
    if (block.x > 1024) block.x = 1024;
    if (norm_type == kLayerNorm) { LAUNCH(8, kLayerNorm); } else { LAUNCH(8, kRMSNorm); }
  }
#undef LAUNCH
}

}  // namespace

torch::Tensor fused_norm_scale_shift(
    const torch::Tensor& x, const c10::optional<torch::Tensor>& gamma_opt,
    const c10::optional<torch::Tensor>& beta_opt, const torch::Tensor& scale,
    const torch::Tensor& shift, int64_t norm_type, double eps) {
  JO_CHECK_CUDA(x);
  JO_CHECK_CUDA(scale);
  JO_CHECK_CUDA(shift);
  TORCH_CHECK(x.dim() == 2, "x must be 2D [M, N]");
  TORCH_CHECK(x.stride(-1) == 1, "x last dim must be contiguous");
  const int64_t M = x.size(0), N = x.size(1);
  TORCH_CHECK((N % 4) == 0, "N must be divisible by 4");
  TORCH_CHECK(scale.dim() == 2 && shift.dim() == 2, "scale/shift must be 2D [M, N]");
  TORCH_CHECK(scale.size(0) == M && shift.size(0) == M, "scale/shift rows must equal M (per-row modulation)");
  TORCH_CHECK(scale.size(1) == N && shift.size(1) == N, "scale/shift last dim must be N");
  TORCH_CHECK(scale.stride(-1) == 1 && shift.stride(-1) == 1, "scale/shift last dim must be contiguous");
  TORCH_CHECK(x.dtype() == scale.dtype() && x.dtype() == shift.dtype(), "x/scale/shift dtype must match");
  TORCH_CHECK(norm_type == 0 || norm_type == 1, "norm_type must be 0 (layer) or 1 (rms)");

  bool has_gamma = gamma_opt.has_value() && gamma_opt->defined();
  bool has_beta = beta_opt.has_value() && beta_opt->defined();
  bool affine = has_gamma;
  void* gamma_ptr = has_gamma ? gamma_opt->data_ptr() : nullptr;
  void* beta_ptr = has_beta ? beta_opt->data_ptr() : nullptr;
  if (has_gamma) TORCH_CHECK(gamma_opt->numel() == N, "gamma must be length N");
  if (has_beta) TORCH_CHECK(beta_opt->numel() == N, "beta must be length N");

  const c10::cuda::CUDAGuard guard(x.device());
  auto y = torch::empty_like(x);
  if (x.dtype() == torch::kFloat32)
    launch<DTypeTag<float4, float>>(x, gamma_ptr, beta_ptr, scale, shift, y, (int)norm_type, affine, (float)eps);
  else if (x.dtype() == torch::kFloat16)
    launch<DTypeTag<half4, __half>>(x, gamma_ptr, beta_ptr, scale, shift, y, (int)norm_type, affine, (float)eps);
  else if (x.dtype() == torch::kBFloat16)
    launch<DTypeTag<bf16_4, __nv_bfloat16>>(x, gamma_ptr, beta_ptr, scale, shift, y, (int)norm_type, affine, (float)eps);
  else
    TORCH_CHECK(false, "Unsupported dtype; use float32/float16/bfloat16");
  return y;
}

}  // namespace joyomni_ops
