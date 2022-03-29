// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "complex_mul.h"
#include "complex_mul_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cu_inc/binary_elementwise_impl.cuh"
#include "core/providers/cuda/math/binary_elementwise_ops.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
__device__ __inline__ void _ComplexMul(T a0, T a1, T b0, T b1, T* output_data, bool is_conj) {
  if (is_conj) {
    T out_real = a0 * b0 + a1 * b1;
    T out_imag = a1 * b0 - a0 * b1;
    output_data[0] = out_real;
    output_data[1] = out_imag;
  } else {
    T out_real = a0 * b0 - a1 * b1;
    T out_imag = a0 * b1 + a1 * b0;
    output_data[0] = out_real;
    output_data[1] = out_imag;
  }
};

template <typename T, typename OffsetCalcT>
__global__ void UnrolledBinaryElementwiseComplexKernel(const T* lhs_data, const T* rhs_data, T* output_data,
                                                       OffsetCalcT offset_calc, CUDA_LONG N, int64_t lhs_size,
                                                       int64_t rhs_size, bool is_conj) {
  CUDA_LONG start = elements_per_thread * threads_per_block * blockIdx.x + threadIdx.x;
  T a[elements_per_thread];
  T b[elements_per_thread];
  T c[elements_per_thread];
  T d[elements_per_thread];

  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < elements_per_thread; ++i) {
    if (id < N / 2) {
      TArray<int32_t, 2> offsets = offset_calc.get(id);
      a[i] = lhs_data[(2 * offsets[0]) % lhs_size];
      b[i] = lhs_data[(2 * offsets[0] + 1) % lhs_size];
      c[i] = rhs_data[(2 * offsets[1]) % rhs_size];
      d[i] = rhs_data[(2 * offsets[1] + 1) % rhs_size];
      id += threads_per_block;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < elements_per_thread; ++i) {
    if (id < N / 2) {
      _ComplexMul(a[i], b[i], c[i], d[i], &output_data[2 * id], is_conj);
      id += threads_per_block;
    }
  }
}

#define HANDLE_COMPLEX_RHS_INDEX_TYPE(LHS_INDEX_TYPE, RHS_INDEX_TYPE)                                                  \
  case RHS_INDEX_TYPE: {                                                                                               \
    auto offset_calc =                                                                                                 \
        BinaryOffsetCalculator<LHS_INDEX_TYPE, RHS_INDEX_TYPE>(rank, output_strides, lhs_strides, rhs_strides);        \
    UnrolledBinaryElementwiseComplexKernel<T, decltype(offset_calc)>                                                   \
        <<<blocks_per_grid, threads_per_block, 0, stream>>>(lhs_data, rhs_data, output_data, offset_calc, N, lhs_size, \
                                                            rhs_size, is_conj);                                        \
  } break

#define HANDLE_COMPLEX_LHS_INDEX_TYPE(LHS_INDEX_TYPE, RHS_INDEX_TYPE_VAL)             \
  case LHS_INDEX_TYPE: {                                                              \
    switch (RHS_INDEX_TYPE_VAL) {                                                     \
      HANDLE_COMPLEX_RHS_INDEX_TYPE(LHS_INDEX_TYPE, BroadcastIndexType::NoBroadcast); \
      HANDLE_COMPLEX_RHS_INDEX_TYPE(LHS_INDEX_TYPE, BroadcastIndexType::Scalar);      \
      HANDLE_COMPLEX_RHS_INDEX_TYPE(LHS_INDEX_TYPE, BroadcastIndexType::NeedCompute); \
    }                                                                                 \
  } break

template <typename T>
void ComplexMul_Impl(cudaStream_t stream, size_t rank, BroadcastIndexType lhs_index_type,
                     BroadcastIndexType rhs_index_type, gsl::span<const int64_t> lhs_strides,
                     gsl::span<const int64_t> rhs_strides, gsl::span<const int64_t> output_shapes,
                     gsl::span<const int64_t> output_strides, const T* lhs_data, const T* rhs_data, T* output_data,
                     size_t count, int64_t lhs_size, int64_t rhs_size, bool is_conj) {
  if (count == 0) return;
  int blocks_per_grid = static_cast<int>(CeilDiv(count, elements_per_thread * threads_per_block));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  switch (lhs_index_type) {
    HANDLE_COMPLEX_LHS_INDEX_TYPE(BroadcastIndexType::NoBroadcast, rhs_index_type);
    HANDLE_COMPLEX_LHS_INDEX_TYPE(BroadcastIndexType::Scalar, rhs_index_type);
    HANDLE_COMPLEX_LHS_INDEX_TYPE(BroadcastIndexType::NeedCompute, rhs_index_type);
  }
};

#define SPECIALIZE_STACKEDCOMPLEXMUL_IMPL(T)                                                                      \
  template void ComplexMul_Impl<T>(cudaStream_t stream, size_t rank, BroadcastIndexType lhs_index_type,           \
                                   BroadcastIndexType rhs_index_type, gsl::span<const int64_t> lhs_strides,       \
                                   gsl::span<const int64_t> rhs_strides, gsl::span<const int64_t> output_shapes,  \
                                   gsl::span<const int64_t> output_strides, const T* lhs_data, const T* rhs_data, \
                                   T* output_data, size_t count, int64_t lhs_size, int64_t rhs_size, bool is_conj);

SPECIALIZE_STACKEDCOMPLEXMUL_IMPL(float)
SPECIALIZE_STACKEDCOMPLEXMUL_IMPL(half)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
