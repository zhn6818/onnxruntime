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
  CUDA_LONG start = kElementsPerThread * kThreadsPerBlock * blockIdx.x + threadIdx.x;
  T a[kElementsPerThread];
  T b[kElementsPerThread];
  T c[kElementsPerThread];
  T d[kElementsPerThread];

  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    if (id < N / 2) {
      TArray<int32_t, 2> offsets = offset_calc.get(id);
      a[i] = lhs_data[(2 * offsets[0]) % lhs_size];
      b[i] = lhs_data[(2 * offsets[0] + 1) % lhs_size];
      c[i] = rhs_data[(2 * offsets[1]) % rhs_size];
      d[i] = rhs_data[(2 * offsets[1] + 1) % rhs_size];
      id += kThreadsPerBlock;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    if (id < N / 2) {
      _ComplexMul(a[i], b[i], c[i], d[i], &output_data[2 * id], is_conj);
      id += kThreadsPerBlock;
    }
  }
}

#define HANDLE_COMPLEX_RHS_INDEX_TYPE(lhs_index_type, rhs_index_type)                                                 \
  case rhs_index_type: {                                                                                              \
    auto offset_calc = BinaryOffsetCalculator<lhs_index_type, rhs_index_type>(                                        \
        static_cast<int>(args.rank), args.lhs_strides, args.rhs_strides, args.output_fdms);                           \
    UnrolledBinaryElementwiseComplexKernel<T, decltype(offset_calc)>                                                  \
        <<<blocks_per_grid, kThreadsPerBlock, 0, stream>>>(lhs_data, rhs_data, output_data, offset_calc, N, lhs_size, \
                                                           rhs_size, is_conj);                                        \
  } break

#define HANDLE_COMPLEX_LHS_INDEX_TYPE(lhs_index_type, rhs_index_type_val)             \
  case lhs_index_type: {                                                              \
    switch (rhs_index_type_val) {                                                     \
      HANDLE_COMPLEX_RHS_INDEX_TYPE(lhs_index_type, BroadcastIndexType::NoBroadcast); \
      HANDLE_COMPLEX_RHS_INDEX_TYPE(lhs_index_type, BroadcastIndexType::Scalar);      \
      HANDLE_COMPLEX_RHS_INDEX_TYPE(lhs_index_type, BroadcastIndexType::NeedCompute); \
    }                                                                                 \
  } break

template <typename T>
void ComplexMul_Impl(cudaStream_t stream, const T* lhs_data, const T* rhs_data, T* output_data,
                     const BinaryElementwiseArgs& args, int64_t lhs_size, int64_t rhs_size, bool is_conj) {
  if (args.output_size == 0) return;
  CUDA_LONG N = static_cast<CUDA_LONG>(args.output_size);
  int blocks_per_grid = static_cast<int>(CeilDiv(N, kElementsPerThread * kThreadsPerBlock));
  switch (args.lhs_index_type) {
    HANDLE_COMPLEX_LHS_INDEX_TYPE(BroadcastIndexType::NoBroadcast, args.rhs_index_type);
    HANDLE_COMPLEX_LHS_INDEX_TYPE(BroadcastIndexType::Scalar, args.rhs_index_type);
    HANDLE_COMPLEX_LHS_INDEX_TYPE(BroadcastIndexType::NeedCompute, args.rhs_index_type);
  }
};

#define SPECIALIZE_STACKEDCOMPLEXMUL_IMPL(T)                                                                  \
  template void ComplexMul_Impl<T>(cudaStream_t stream, const T* lhs_data, const T* rhs_data, T* output_data, \
                                   const BinaryElementwiseArgs& args, int64_t lhs_size, int64_t rhs_size,     \
                                   bool is_conj);

SPECIALIZE_STACKEDCOMPLEXMUL_IMPL(float)
SPECIALIZE_STACKEDCOMPLEXMUL_IMPL(half)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
