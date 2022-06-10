// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace cuda {

namespace {
#ifdef USE_ROCM
constexpr int elements_per_thread = 2;
constexpr int threads_per_block = 512;
#else
constexpr int elements_per_thread = GridDim::maxElementsPerThread;
constexpr int threads_per_block = GridDim::maxThreadsPerBlock;
#endif
}  // namespace

template <BroadcastIndexType LhsIndexType, BroadcastIndexType RhsIndexType>
struct BinaryOffsetCalculator {
  BinaryOffsetCalculator(int rank, gsl::span<const int64_t> output_strides, gsl::span<const int64_t> lhs_strides,
                         gsl::span<const int64_t> rhs_strides)
      : rank_(rank) {
    if (LhsIndexType == BroadcastIndexType::NeedCompute || RhsIndexType == BroadcastIndexType::NeedCompute) {
      fast_divmods_.SetSize(rank);
      strides_.SetSize(rank);
      for (int dim = 0; dim < rank; ++dim) {
        fast_divmods_[dim] = fast_divmod(gsl::narrow_cast<int>(output_strides[dim]));
        strides_[dim].SetSize(2);
        if (LhsIndexType == BroadcastIndexType::NeedCompute)
          strides_[dim][0] = static_cast<CUDA_LONG>(lhs_strides[dim]);
        if (RhsIndexType == BroadcastIndexType::NeedCompute)
          strides_[dim][1] = static_cast<CUDA_LONG>(rhs_strides[dim]);
      }
    }
  }

  __device__ __forceinline__ TArray<CUDA_LONG, 2> get(CUDA_LONG linear_idx) const {
    TArray<CUDA_LONG, 2> offsets;
    offsets[0] = LhsIndexType == BroadcastIndexType::NoBroadcast ? linear_idx : 0;
    offsets[1] = RhsIndexType == BroadcastIndexType::NoBroadcast ? linear_idx : 0;

    if (LhsIndexType == BroadcastIndexType::NeedCompute || RhsIndexType == BroadcastIndexType::NeedCompute) {
#pragma unroll
      for (int dim = 0; dim < fast_divmods_.Capacity(); ++dim) {
        if (dim == rank_) {
          break;
        }
        CUDA_LONG q, r;
        fast_divmods_[dim].divmod(linear_idx, q, r);
        linear_idx = r;
        if (LhsIndexType == BroadcastIndexType::NeedCompute) offsets[0] += strides_[dim][0] * q;
        if (RhsIndexType == BroadcastIndexType::NeedCompute) offsets[1] += strides_[dim][1] * q;
      }
    }

    return offsets;
  }

  int rank_;
  TArray<fast_divmod> fast_divmods_;
  TArray<TArray<CUDA_LONG, 2>> strides_;
};

template <bool IsRhsNeedCompute, bool IsBatchN>
struct BinaryBatchOffsetCalculator {
  BinaryBatchOffsetCalculator(int h, int c) {
    fast_divmod_h_ = fast_divmod(h);
    if (IsBatchN) fast_divmod_c_ = fast_divmod(c);
  }

  __device__ __forceinline__ TArray<CUDA_LONG, 2> get(CUDA_LONG linear_idx) const {
    TArray<CUDA_LONG, 2> offsets;
    CUDA_LONG offset = fast_divmod_h_.div(linear_idx);
    if (IsBatchN) {
      CUDA_LONG q, r;
      fast_divmod_c_.divmod(offset, q, r);
      offset = r;
    }
    offsets[0] = IsRhsNeedCompute ? linear_idx : offset;
    offsets[1] = IsRhsNeedCompute ? offset : linear_idx;
    return offsets;
  }

  fast_divmod fast_divmod_h_;
  fast_divmod fast_divmod_c_;
};

static bool TryGetChannelBatch(size_t rank, gsl::span<const int64_t> input_strides,
                               gsl::span<const int64_t> output_shapes, int& b, int& c, int& h) {
  size_t count = 0;
  size_t dim_c = 0;
  for (size_t dim = 0; dim < rank; ++dim) {
    if (output_shapes[dim] != 1 && input_strides[dim] != 0) {
      if (count > 0) return false;
      ++count;
      c = output_shapes[dim];
      dim_c = dim;
    }
  }
  if (count == 0) return false;
  b = 1;
  for (size_t dim = 0; dim < dim_c; ++dim) {
    b *= static_cast<int>(output_shapes[dim]);
  }
  h = 1;
  for (size_t dim = dim_c + 1; dim < rank; ++dim) {
    h *= static_cast<int>(output_shapes[dim]);
  }
  return true;
}

// for scalar broadcast or non-broadcast case
template <bool IncL, bool IncR, typename T, typename T1, typename T2, typename FuncT>
__global__ void _BinaryElementWiseSimple(const T1* lhs_data, const T2* rhs_data, T* output_data, const FuncT& func,
                                         CUDA_LONG N) {
  CUDA_LONG start = elements_per_thread * threads_per_block * blockIdx.x + threadIdx.x;
  T1 lvalue[elements_per_thread];
  T2 rvalue[elements_per_thread];

  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < elements_per_thread; i++) {
    if (id < N) {
      lvalue[i] = lhs_data[IncL ? id : 0];
      rvalue[i] = rhs_data[IncR ? id : 0];

      id += threads_per_block;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < elements_per_thread; i++) {
    if (id < N) {
      output_data[id] = func(lvalue[i], rvalue[i]);

      id += threads_per_block;
    }
  }
}

template <typename T, typename T1, typename T2, typename FuncT>
void BinaryElementWiseNoBroadcastImpl(cudaStream_t stream, const T1* lhs_data, const T2* rhs_data, T* output_data,
                                      const FuncT& func, size_t count) {
  if (count == 0)  // special case where there's a dim value of 0 in the output shape
    return;

  int blocksPerGrid = static_cast<int>(CeilDiv(count, threads_per_block * elements_per_thread));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  _BinaryElementWiseSimple<true, true, T, T1, T2, FuncT>
      <<<blocksPerGrid, threads_per_block, 0, stream>>>(lhs_data, rhs_data, output_data, func, N);
}

template <typename T, typename T1, typename T2, typename FuncT, typename OffsetCalcT>
__global__ void UnrolledBinaryElementwiseKernel(const T1* lhs_data, const T2* rhs_data, T* output_data, FuncT functor,
                                                OffsetCalcT offset_calc, CUDA_LONG N) {
  CUDA_LONG start = elements_per_thread * threads_per_block * blockIdx.x + threadIdx.x;
  T1 lvalue[elements_per_thread];
  T2 rvalue[elements_per_thread];

  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < elements_per_thread; ++i) {
    if (id < N) {
      TArray<int32_t, 2> offsets = offset_calc.get(id);
      lvalue[i] = lhs_data[offsets[0]];
      rvalue[i] = rhs_data[offsets[1]];
      id += threads_per_block;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < elements_per_thread; ++i) {
    if (id < N) {
      output_data[id] = functor(lvalue[i], rvalue[i]);
      id += threads_per_block;
    }
  }
}

#define HANDLE_RHS_INDEX_TYPE(LHS_INDEX_TYPE, RHS_INDEX_TYPE)                                                       \
  case RHS_INDEX_TYPE: {                                                                                            \
    auto offset_calc =                                                                                              \
        BinaryOffsetCalculator<LHS_INDEX_TYPE, RHS_INDEX_TYPE>(rank, output_strides, lhs_strides, rhs_strides);     \
    UnrolledBinaryElementwiseKernel<T, T1, T2, FuncT, decltype(offset_calc)>                                        \
        <<<blocks_per_grid, threads_per_block, 0, stream>>>(lhs_data, rhs_data, output_data, func, offset_calc, N); \
  } break

#define HANDLE_LHS_INDEX_TYPE(LHS_INDEX_TYPE, RHS_INDEX_TYPE_VAL)             \
  case LHS_INDEX_TYPE: {                                                      \
    switch (RHS_INDEX_TYPE_VAL) {                                             \
      HANDLE_RHS_INDEX_TYPE(LHS_INDEX_TYPE, BroadcastIndexType::NoBroadcast); \
      HANDLE_RHS_INDEX_TYPE(LHS_INDEX_TYPE, BroadcastIndexType::Scalar);      \
      HANDLE_RHS_INDEX_TYPE(LHS_INDEX_TYPE, BroadcastIndexType::NeedCompute); \
    }                                                                         \
  } break

#define HANDLE_CHANNEL_BATCH(IS_RHS_NEED_COMPUTE, IS_BATCH_N)                            \
  auto offset_calc = BinaryBatchOffsetCalculator<IS_RHS_NEED_COMPUTE, IS_BATCH_N>(h, c); \
  UnrolledBinaryElementwiseKernel<T, T1, T2, FuncT, decltype(offset_calc)>               \
      <<<blocks_per_grid, threads_per_block, 0, stream>>>(lhs_data, rhs_data, output_data, func, offset_calc, N)

template <typename T, typename T1, typename T2, typename FuncT>
void BinaryElementWiseImpl(cudaStream_t stream, size_t rank, BroadcastIndexType lhs_index_type,
                           BroadcastIndexType rhs_index_type, gsl::span<const int64_t> lhs_strides,
                           gsl::span<const int64_t> rhs_strides, gsl::span<const int64_t> output_shapes,
                           gsl::span<const int64_t> output_strides, const T1* lhs_data, const T2* rhs_data,
                           T* output_data, const FuncT& func, size_t count) {
  if (count == 0) return;
  int blocks_per_grid = static_cast<int>(CeilDiv(count, elements_per_thread * threads_per_block));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  int b, c, h;
  if (lhs_index_type == BroadcastIndexType::NoBroadcast && rhs_index_type == BroadcastIndexType::NeedCompute &&
      TryGetChannelBatch(rank, rhs_strides, output_shapes, b, c, h)) {
    if (b == 1) {
      HANDLE_CHANNEL_BATCH(true, false);
    } else {
      HANDLE_CHANNEL_BATCH(true, true);
    }
  } else if (lhs_index_type == BroadcastIndexType::NeedCompute && rhs_index_type == BroadcastIndexType::NoBroadcast &&
             TryGetChannelBatch(rank, lhs_strides, output_shapes, b, c, h)) {
    if (b == 1) {
      HANDLE_CHANNEL_BATCH(false, false);
    } else {
      HANDLE_CHANNEL_BATCH(false, true);
    }
  } else {
    switch (lhs_index_type) {
      HANDLE_LHS_INDEX_TYPE(BroadcastIndexType::NoBroadcast, rhs_index_type);
      HANDLE_LHS_INDEX_TYPE(BroadcastIndexType::Scalar, rhs_index_type);
      HANDLE_LHS_INDEX_TYPE(BroadcastIndexType::NeedCompute, rhs_index_type);
    }
  }
}

}  // namespace cuda
}  // namespace onnxruntime
