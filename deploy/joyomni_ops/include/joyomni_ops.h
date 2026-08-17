// Minimal shared utilities for joyomni_ops kernels.
// Self-contained: depends only on PyTorch + CUDA. No sgl-kernel / flashinfer.
#pragma once

#include <cuda_runtime.h>
#include <torch/all.h>

#define JO_CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define JO_CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define JO_CHECK_DTYPE(x, dt) \
  TORCH_CHECK((x).scalar_type() == (dt), #x " dtype is ", (x).scalar_type(), ", expected ", (dt))
#define JO_CHECK_INPUT(x, dt) \
  JO_CHECK_CUDA(x);           \
  JO_CHECK_CONTIGUOUS(x);     \
  JO_CHECK_DTYPE(x, dt)

// Compute capability (major*10 + minor) of the current CUDA device.
inline int getSMVersion() {
  int device{-1};
  cudaGetDevice(&device);
  int major = 0, minor = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
  return major * 10 + minor;
}
