// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/nn/softmax_dropout_grad_impl.h"

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/math/softmax_warpwise_impl.cuh"
#include "core/providers/cuda/shared_inc/accumulation_type.h"

namespace onnxruntime {
namespace cuda {

namespace {
constexpr int kNumUnroll = 4;
constexpr uint32_t FULL_MASK = 0xffffffff;
}  // namespace

template <typename T, typename AccT, int Log2Elements>
__global__ void SoftmaxDropoutGradVectorizedKernel(T* input_grad_data, const T* output_grad_data, const bool* mask_data,
                                                   const T* softmax_output_data, int element_count, int batch_count,
                                                   int batch_stride, const float scale) {
  constexpr int kNextPowerOfTwo = 1 << Log2Elements;
  constexpr int kWarpSize = (kNextPowerOfTwo < GPU_WARP_SIZE) ? kNextPowerOfTwo : GPU_WARP_SIZE;
  constexpr int kWarpIterations = kNextPowerOfTwo / kWarpSize;

  int batch = blockDim.y * blockIdx.x + threadIdx.y;
  int local_idx = threadIdx.x;
  int thread_offset = batch * batch_stride + kNumUnroll * local_idx;

  using TVec4 = aligned_vector<T, kNumUnroll>;
  using MaskVec4 = aligned_vector<bool, kNumUnroll>;

  // load data from global memory
  T output_grads_t[kWarpIterations];
  T softmax_outputs_t[kWarpIterations];
  T input_grads_t[kNumUnroll];
  bool masks[kWarpIterations];
  AccT grads_acct[kWarpIterations];
  AccT softmax_outputs_acct[kWarpIterations];

  for (int it = 0; it < kWarpIterations; it += kNumUnroll) {
    int element_index = kNumUnroll * local_idx + it * kWarpSize;
    if (element_index < element_count) {
      int itr_jmp = it * kWarpSize;
      *(reinterpret_cast<TVec4*>(output_grads_t + it)) =
          *(reinterpret_cast<const TVec4*>(output_grad_data + thread_offset + itr_jmp));
      *(reinterpret_cast<TVec4*>(softmax_outputs_t + it)) =
          *(reinterpret_cast<const TVec4*>(softmax_output_data + thread_offset + itr_jmp));
      *(reinterpret_cast<MaskVec4*>(masks + it)) =
          *(reinterpret_cast<const MaskVec4*>(mask_data + thread_offset + itr_jmp));
#pragma unroll
      for (int element = 0; element < kNumUnroll; ++element) {
        softmax_outputs_acct[it + element] = static_cast<AccT>(softmax_outputs_t[it + element]);
        grads_acct[it + element] = (AccT)(output_grads_t[it + element]) * masks[it + element] * scale *
                                   (AccT)(softmax_outputs_t[it + element]);
      }
    } else {
#pragma unroll
      for (int element = 0; element < kNumUnroll; ++element) {
        softmax_outputs_acct[it + element] = static_cast<AccT>(0.f);
        grads_acct[it + element] = static_cast<AccT>(0.f);
      }
    }
  }

  AccT sum;
  sum = grads_acct[0];
#pragma unroll
  for (int it = 1; it < kWarpIterations; ++it) {
    sum += grads_acct[it];
  }

// reduction sum
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    sum += __shfl_xor_sync(FULL_MASK, sum, offset, kWarpSize);
  }

// store result
#pragma unroll
  for (int it = 0; it < kWarpIterations; it += kNumUnroll) {
    int element_index = kNumUnroll * local_idx + it * kWarpSize;
    if (element_index < element_count) {
#pragma unroll
      for (int element = 0; element < kNumUnroll; ++element) {
        input_grads_t[element] = grads_acct[it + element] - sum * softmax_outputs_acct[it + element];
      }
      *(reinterpret_cast<TVec4*>(input_grad_data + thread_offset + it * kWarpSize)) =
          *(reinterpret_cast<TVec4*>(input_grads_t));
    }
  }
}

template <typename T>
void SoftmaxDropoutGradImpl(cudaStream_t stream, T* input_grad_data, const T* output_grad_data, const bool* mask_data,
                            const T* softmax_output_data, int element_count, int batch_count, int batch_stride,
                            const float ratio) {
  if (element_count == 0) return;

  typedef AccumulationType_t<T> AccT;
  const float scale = 1.f / (1.f - ratio);

  int log2_elements = log2_ceil(element_count);
  const int next_power_of_two = 1 << log2_elements;

  // This value must match the WARP_SIZE constexpr value computed inside softmax_warp_forward.
  int warp_size = std::min(next_power_of_two, GPU_WARP_SIZE);

  // This value must match the WARP_BATCH constexpr value computed inside softmax_warp_forward.
  int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

  // use 128 threads per block to maximize gpu utilization
  constexpr int threads_per_block = 128;

  int warps_per_block = (threads_per_block / warp_size);
  int batches_per_block = warps_per_block * batches_per_warp;
  int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
  dim3 threads(warp_size, warps_per_block, 1);

  switch (log2_elements) {
#define CASE_LOG2_ELEMENTS(value)                                                                           \
  case value: {                                                                                             \
    SoftmaxDropoutGradVectorizedKernel<T, AccT, value>                                                      \
        <<<blocks, threads, 0, stream>>>(input_grad_data, output_grad_data, mask_data, softmax_output_data, \
                                         element_count, batch_count, batch_stride, ratio);                  \
  } break
    CASE_LOG2_ELEMENTS(0);   // 1
    CASE_LOG2_ELEMENTS(1);   // 2
    CASE_LOG2_ELEMENTS(2);   // 4
    CASE_LOG2_ELEMENTS(3);   // 8
    CASE_LOG2_ELEMENTS(4);   // 16
    CASE_LOG2_ELEMENTS(5);   // 32
    CASE_LOG2_ELEMENTS(6);   // 64
    CASE_LOG2_ELEMENTS(7);   // 128
    CASE_LOG2_ELEMENTS(8);   // 256
    CASE_LOG2_ELEMENTS(9);   // 512
    CASE_LOG2_ELEMENTS(10);  // 1024
#undef CASE_LOG2_ELEMENTS
  }
}

#define SPECIALIZED_SOFTMAX_DROPOUT_GRAD_IMPL(T)                                                                  \
  template void SoftmaxDropoutGradImpl<T>(cudaStream_t stream, T * input_grad_data, const T* output_grad_data,    \
                                          const bool* mask_data, const T* softmax_output_data, int element_count, \
                                          int batch_count, int batch_stride, const float ratio);

SPECIALIZED_SOFTMAX_DROPOUT_GRAD_IMPL(double)
SPECIALIZED_SOFTMAX_DROPOUT_GRAD_IMPL(float)
SPECIALIZED_SOFTMAX_DROPOUT_GRAD_IMPL(half)

}  // namespace cuda
}  // namespace onnxruntime
