// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/copy_impl.h"

#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace cuda {

namespace {
#ifdef USE_ROCM
constexpr int kElementsPerThread = 2;
constexpr int kThreadsPerBlock = 512;
#else
constexpr int kElementsPerThread = GridDim::maxElementsPerThread;
constexpr int kThreadsPerBlock = GridDim::maxThreadsPerBlock;
#endif
}  // namespace

template <typename T>
__global__ void StridedCopyKernel(int rank, const T* src_data, TArray<int64_t> src_strides, T* dst_data,
                                  TArray<int64_t> dst_strides, TArray<fast_divmod> dst_fdms, CUDA_LONG N) {
  CUDA_LONG start = kElementsPerThread * kThreadsPerBlock * blockIdx.x + threadIdx.x;
  T value[kElementsPerThread];
  CUDA_LONG dst_offsets[kElementsPerThread];

  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    if (id < N) {
      CUDA_LONG src_offset = 0;
      CUDA_LONG dst_offset = 0;
#pragma unroll
      for (int dim = 0; dim < dst_fdms.Capacity(); ++dim) {
        if (dim == rank) {
          break;
        }
        CUDA_LONG q, r = id;
        dst_fdms[dim].divmod(r, q, r);
        src_offset += src_strides[dim] * q;
        dst_offset += dst_strides[dim] * q;
      }

      value[i] = src_data[src_offset];
      dst_offsets[i] = dst_offset;
      id += kThreadsPerBlock;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    if (id < N) {
      dst_data[dst_offsets[i]] = value[i];
      id += kThreadsPerBlock;
    }
  }
}

template <typename T>
void StridedCopyImpl(cudaStream_t stream, size_t rank, const T* src_data, const TArray<int64_t>& src_strides,
                     T* dst_data, const TArray<int64_t>& dst_strides, const TArray<fast_divmod>& dst_fdms,
                     const size_t output_size) {
  if (output_size == 0) return;
  CUDA_LONG N = static_cast<CUDA_LONG>(output_size);
  int blocks_per_grid = static_cast<int>(CeilDiv(N, kElementsPerThread * kThreadsPerBlock));
  StridedCopyKernel<T><<<blocks_per_grid, kThreadsPerBlock, 0, stream>>>(static_cast<int>(rank), src_data, src_strides,
                                                                         dst_data, dst_strides, dst_fdms, N);
}

#define STRIDED_COPY_IMPL_SPECIALIZED(T)                                                                    \
  template void StridedCopyImpl<T>(                                                                         \
      cudaStream_t stream, size_t rank, const T* src_data, const TArray<int64_t>& src_strides, T* dst_data, \
      const TArray<int64_t>& dst_strides, const TArray<fast_divmod>& dst_fdms, const size_t output_size);

STRIDED_COPY_IMPL_SPECIALIZED(int8_t)
STRIDED_COPY_IMPL_SPECIALIZED(int16_t)
STRIDED_COPY_IMPL_SPECIALIZED(int32_t)
STRIDED_COPY_IMPL_SPECIALIZED(int64_t)

#undef STRIDED_COPY_IMPL_SPECIALIZED

}  // namespace cuda
}  // namespace onnxruntime
