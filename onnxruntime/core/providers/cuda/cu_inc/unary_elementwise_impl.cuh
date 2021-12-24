// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace cuda {

template <typename InT, typename OutT, typename FuncT, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _UnaryElementWise(
    const InT* input_data,
    OutT* output_data,
    const FuncT functor,
    CUDA_LONG N) {
  CUDA_LONG start = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;
  InT value[NumElementsPerThread];

  CUDA_LONG id = start;
  #pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      value[i] = input_data[id];
      id += NumThreadsPerBlock;
    }
  }

  id = start;
  #pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      output_data[id] = functor(value[i]);
      id += NumThreadsPerBlock;
    }
  }
}

template <typename InT, typename OutT, typename FuncT, int NumElementsPerThread>
__global__ void _VectorizedUnaryElementWise(const InT* input_data, OutT* output_data, const FuncT& functor,
                                            CUDA_LONG N) {
  CUDA_LONG id = (blockDim.x * blockIdx.x + threadIdx.x) * NumElementsPerThread;
  if (id >= N) return;
  using LoadInT = aligned_vector<InT, NumElementsPerThread>;
  using LoadOutT = aligned_vector<OutT, NumElementsPerThread>;

  // Vectorized load into storage.
  InT input_vec[NumElementsPerThread];
  LoadInT* input_value = reinterpret_cast<LoadInT*>(&input_vec);
  *input_value = *reinterpret_cast<const LoadInT*>(&input_data[id]);

  OutT results[NumElementsPerThread];
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; ++i) {
    results[i] = functor(input_vec[i]);
  }

  // Vectorized writes.
  *(reinterpret_cast<LoadOutT*>(&output_data[id])) = *reinterpret_cast<LoadOutT*>(&results[0]);
}

template <typename InT, typename OutT, typename FuncT>
void UnaryElementWiseImpl(
    cudaStream_t stream,
    const InT* input_data,
    OutT* output_data,
    const FuncT& func,
    size_t count) {
  if (count == 0)  // special case where there's a dim value of 0 in the shape
    return;

  int blocksPerGrid = static_cast<int>(CeilDiv(count, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  if (N % GridDim::maxElementsPerThread == 0) {
    _VectorizedUnaryElementWise<InT, OutT, FuncT, GridDim::maxElementsPerThread>
        <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(input_data, output_data, func, N);
  } else {
    _UnaryElementWise<InT, OutT, FuncT, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>
        <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(input_data, output_data, func, N);
  }
}

}  // namespace cuda
}  // namespace onnxruntime
