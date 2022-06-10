// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_runtime.h>
#include "core/providers/cuda/math/binary_elementwise_ops_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cu_inc/binary_elementwise_impl.cuh"
#include "core/framework/tensor_shape.h"

namespace onnxruntime {
namespace cuda {

// for now this operator classes are no different than a funciton.
// Eventually once multiple binary gradient ops are needed, we will pass
// its instance from API instead of direct function call.
template <class T>
struct OP_A_DivGrad {
  __device__ __inline__ T operator()(T dy, T b) const {
    return dy / b;
  }
};
template <class T>
struct OP_B_DivGrad {
  __device__ __inline__ T operator()(T dy, T a, T b) const {
    return -dy * a / (b * b);
  }
};

template <typename T, typename OffsetCalcT, bool require_da, bool require_db>
__global__ void UnrolledBinaryElementwiseDivGradKernel(const T* a_data, const T* b_data, const T* dy_data,
                                                       T* output_da_data, T* output_db_data, OffsetCalcT offset_calc, CUDA_LONG N) {
  CUDA_LONG start = elements_per_thread * threads_per_block * blockIdx.x + threadIdx.x;
  T avalue[elements_per_thread];
  T bvalue[elements_per_thread];

  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < elements_per_thread; ++i) {
    if (id < N) {
      TArray<int32_t, 2> offsets = offset_calc.get(id);
      avalue[i] = a_data[offsets[0]];
      bvalue[i] = b_data[offsets[1]];
      id += threads_per_block;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < elements_per_thread; ++i) {
    if (id < N) {
      if (require_da) output_da_data[id] = OP_A_DivGrad<T>()(dy_data[id], bvalue[i]);
      if (require_db) output_db_data[id] = OP_B_DivGrad<T>()(dy_data[id], avalue[i], bvalue[i]);
      id += threads_per_block;
    }
  }
}

#define HANDLE_DIVGRAD_REQUIREMENT()                                                                                 \
  if (da_output_data && db_output_data)                                                                              \
    UnrolledBinaryElementwiseDivGradKernel<T, decltype(offset_calc), true, true>                                     \
        <<<blocks_per_grid, threads_per_block, 0, stream>>>(a_data, b_data, dy_data, da_output_data, db_output_data, \
                                                            offset_calc, N);                                         \
  else if (da_output_data)                                                                                           \
    UnrolledBinaryElementwiseDivGradKernel<T, decltype(offset_calc), true, false>                                    \
        <<<blocks_per_grid, threads_per_block, 0, stream>>>(a_data, b_data, dy_data, da_output_data, db_output_data, \
                                                            offset_calc, N);                                         \
  else                                                                                                               \
    UnrolledBinaryElementwiseDivGradKernel<T, decltype(offset_calc), true, false>                                    \
        <<<blocks_per_grid, threads_per_block, 0, stream>>>(a_data, b_data, dy_data, da_output_data, db_output_data, \
                                                            offset_calc, N)

#define HANDLE_DIVGRAD_RHS_INDEX_TYPE(LHS_INDEX_TYPE, RHS_INDEX_TYPE)                                           \
  case RHS_INDEX_TYPE: {                                                                                        \
    auto offset_calc =                                                                                          \
        BinaryOffsetCalculator<LHS_INDEX_TYPE, RHS_INDEX_TYPE>(rank, output_strides, lhs_strides, rhs_strides); \
    HANDLE_DIVGRAD_REQUIREMENT();                                                                               \
  } break

#define HANDLE_DIVGRAD_LHS_INDEX_TYPE(LHS_INDEX_TYPE, RHS_INDEX_TYPE_VAL)             \
  case LHS_INDEX_TYPE: {                                                              \
    switch (RHS_INDEX_TYPE_VAL) {                                                     \
      HANDLE_DIVGRAD_RHS_INDEX_TYPE(LHS_INDEX_TYPE, BroadcastIndexType::NoBroadcast); \
      HANDLE_DIVGRAD_RHS_INDEX_TYPE(LHS_INDEX_TYPE, BroadcastIndexType::Scalar);      \
      HANDLE_DIVGRAD_RHS_INDEX_TYPE(LHS_INDEX_TYPE, BroadcastIndexType::NeedCompute); \
    }                                                                                 \
  } break

#define HANDLE_DIVGRAD_CHANNEL_BATCH(IS_RHS_NEED_COMPUTE, IS_BATCH_N)                    \
  auto offset_calc = BinaryBatchOffsetCalculator<IS_RHS_NEED_COMPUTE, IS_BATCH_N>(h, c); \
  HANDLE_DIVGRAD_REQUIREMENT()

template <typename T>
void ImplDivGrad(cudaStream_t stream, size_t rank, BroadcastIndexType lhs_index_type, BroadcastIndexType rhs_index_type,
                 gsl::span<const int64_t> lhs_strides, gsl::span<const int64_t> rhs_strides,
                 gsl::span<const int64_t> output_shapes, gsl::span<const int64_t> output_strides, const T* a_data,
                 const T* b_data, const T* dy_data, T* da_output_data, T* db_output_data, size_t count) {
  if (count == 0) return;
  int blocks_per_grid = static_cast<int>(CeilDiv(count, elements_per_thread * threads_per_block));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  int b, c, h;
  // a_data is needed for db_output_data only.
  BroadcastIndexType new_lhs_index_type = db_output_data ? lhs_index_type : BroadcastIndexType::NoBroadcast;
  if (new_lhs_index_type == BroadcastIndexType::NoBroadcast && rhs_index_type == BroadcastIndexType::NeedCompute &&
      TryGetChannelBatch(rank, rhs_strides, output_shapes, b, c, h)) {
    if (b == 1) {
      HANDLE_DIVGRAD_CHANNEL_BATCH(true, false);
    } else {
      HANDLE_DIVGRAD_CHANNEL_BATCH(true, true);
    }
  } else if (new_lhs_index_type == BroadcastIndexType::NeedCompute &&
             rhs_index_type == BroadcastIndexType::NoBroadcast &&
             TryGetChannelBatch(rank, lhs_strides, output_shapes, b, c, h)) {
    if (b == 1) {
      HANDLE_DIVGRAD_CHANNEL_BATCH(false, false);
    } else {
      HANDLE_DIVGRAD_CHANNEL_BATCH(false, true);
    }
  } else {
    switch (new_lhs_index_type) {
      HANDLE_DIVGRAD_LHS_INDEX_TYPE(BroadcastIndexType::NoBroadcast, rhs_index_type);
      HANDLE_DIVGRAD_LHS_INDEX_TYPE(BroadcastIndexType::Scalar, rhs_index_type);
      HANDLE_DIVGRAD_LHS_INDEX_TYPE(BroadcastIndexType::NeedCompute, rhs_index_type);
    }
  }
}

#define SPECIALIZED_DIV_GRAD_IMPL(T)                                                                         \
  template void ImplDivGrad<T>(cudaStream_t stream, size_t rank, BroadcastIndexType lhs_index_type,          \
                               BroadcastIndexType rhs_index_type, gsl::span<const int64_t> lhs_strides,      \
                               gsl::span<const int64_t> rhs_strides, gsl::span<const int64_t> output_shapes, \
                               gsl::span<const int64_t> output_strides, const T* a_data, const T* b_data,    \
                               const T* dy_data, T* da_output_data, T* db_output_data, size_t count);

SPECIALIZED_DIV_GRAD_IMPL(half)
SPECIALIZED_DIV_GRAD_IMPL(float)
SPECIALIZED_DIV_GRAD_IMPL(double)

}  // namespace cuda
}  // namespace onnxruntime
