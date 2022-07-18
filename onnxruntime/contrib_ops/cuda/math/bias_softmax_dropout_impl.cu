// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/bias_softmax_dropout_impl.h"

#include <curand_kernel.h>
#include <algorithm>
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/math/softmax_warpwise_impl.cuh"
#include "core/providers/cuda/shared_inc/accumulation_type.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

namespace {
constexpr int kNumUnroll = 4;
constexpr uint32_t FULL_MASK = 0xffffffff;
}  // namespace

template <typename T, typename AccT, int Log2Elements>
__global__ void BiasSoftmaxDropoutVectorizedKernel(T* output_data, bool* mask_data, T* softmax_output_data,
                                                   const T* input_data, const T* bias_data, int element_count,
                                                   int batch_count, int batch_stride,
                                                   int bias_broadcast_count_per_batch, const float ratio,
                                                   const std::pair<uint64_t, uint64_t> seeds) {
  constexpr int kNextPowOfTwo = 1 << Log2Elements;
  constexpr int kWarpSize = kNextPowOfTwo < GPU_WARP_SIZE ? kNextPowOfTwo : GPU_WARP_SIZE;
  constexpr int kWarpIterations = kNextPowOfTwo / kWarpSize;

  int batch = blockDim.y * blockIdx.x + threadIdx.y;
  int tid = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
  int local_idx = threadIdx.x;

  T inputs[kWarpIterations];
  T biases[kWarpIterations];

  using TVec4 = aligned_vector<T, kNumUnroll>;
  using MaskVec4 = aligned_vector<bool, kNumUnroll>;

  int thread_offset = batch * batch_stride + kNumUnroll * local_idx;
  int bias_offset = (batch / bias_broadcast_count_per_batch) * batch_stride + kNumUnroll * local_idx;

#pragma unroll
  for (int it = 0; it < kWarpIterations; it += kNumUnroll) {
    int element_index = kNumUnroll * local_idx + it * kWarpSize;
    if (element_index < element_count) {
      int itr_jmp = it * kWarpSize;
      *(reinterpret_cast<TVec4*>(inputs + it)) =
          *(reinterpret_cast<const TVec4*>(input_data + thread_offset + itr_jmp));
      *(reinterpret_cast<TVec4*>(biases + it)) = *(reinterpret_cast<const TVec4*>(bias_data + bias_offset + itr_jmp));
    } else {
#pragma unroll
      for (int element = 0; element < kNumUnroll; ++element) {
        inputs[it + element] = T(-10000.f);
        biases[it + element] = T(0.f);
      }
    }
  }

  AccT inputs_acct[kWarpIterations];
#pragma unroll
  for (int it = 0; it < kWarpIterations; ++it) {
    inputs_acct[it] = static_cast<AccT>(inputs[it] + biases[it]);
  }

  // compute local max_value
  AccT max_value = inputs_acct[0];

#pragma unroll
  for (int it = 1; it < kWarpIterations; ++it) {
    max_value = (max_value > inputs_acct[it]) ? max_value : inputs_acct[it];
  }

// reduction max_value
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    float val = __shfl_xor_sync(FULL_MASK, max_value, offset, kWarpSize);
  }

  // compute local sum
  AccT sum = 0.0f;

#pragma unroll
  for (int it = 0; it < kWarpIterations; ++it) {
    inputs_acct[it] = std::exp(inputs_acct[it] - max_value);
    sum += inputs_acct[it];
  }

// reduction sum
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    sum += __shfl_xor_sync(FULL_MASK, sum, offset, kWarpSize);
  }

  T outputs[kWarpIterations];
  bool masks[kWarpIterations];
  T softmax_outputs[kWarpIterations];

  const float p = 1.0f - ratio;
  const float scale = 1.0f / p;
  curandStatePhilox4_32_10_t state;
  curand_init(seeds.first, tid, seeds.second, &state);
  float4 rand;

#pragma unroll
  for (int it = 0; it < kWarpIterations; it += kNumUnroll) {
    int element_index = kNumUnroll * local_idx + it * kWarpSize;
    if (element_index < element_count) {
      rand = curand_uniform4(&state);
      masks[it] = rand.x < p;
      masks[it + 1] = rand.y < p;
      masks[it + 2] = rand.z < p;
      masks[it + 3] = rand.w < p;
#pragma unroll
      for (int element = 0; element < kNumUnroll; ++element) {
        softmax_outputs[it + element] = static_cast<T>(inputs_acct[it + element]);
        outputs[it + element] = static_cast<T>(masks[it + element] * inputs_acct[it + element] * scale);
      }
    }
  }

// store result
#pragma unroll
  for (int it = 0; it < kWarpIterations; it += kNumUnroll) {
    int element_index = kNumUnroll * local_idx + it * kWarpSize;
    if (element_index < element_count) {
      int itr_jmp = thread_offset + it * kWarpSize;
      *(reinterpret_cast<TVec4*>(output_data + itr_jmp)) = *(reinterpret_cast<TVec4*>(outputs + it));
      *(reinterpret_cast<MaskVec4*>(mask_data + itr_jmp)) = *(reinterpret_cast<MaskVec4*>(masks + it));
      *(reinterpret_cast<TVec4*>(softmax_output_data + itr_jmp)) = *(reinterpret_cast<TVec4*>(softmax_outputs + it));
    }
  }
}

template <typename T>
void BiasSoftmaxDropoutImpl(cudaStream_t stream, T* output_data, bool* mask_data, T* softmax_output_data,
                            const T* input_data, const T* bias_data, int element_count, int batch_count,
                            int batch_stride, int bias_broadcast_count_per_batch, const float ratio,
                            PhiloxGenerator& generator) {
  if (element_count == 0) return;

  typedef AccumulationType_t<T> AccT;

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

  auto seeds = generator.NextPhiloxSeeds(static_cast<uint64_t>((next_power_of_two / warp_size) * kNumUnroll));

  switch (log2_elements) {
#define CASE_LOG2_ELEMENTS(value)                                                                                     \
  case value: {                                                                                                       \
    BiasSoftmaxDropoutVectorizedKernel<T, AccT, value><<<blocks, threads, 0, stream>>>(                               \
        output_data, mask_data, softmax_output_data, input_data, bias_data, element_count, batch_count, batch_stride, \
        bias_broadcast_count_per_batch, ratio, seeds);                                                                \
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

#define SPECIALIZED_BIAS_SOFTMAX_DROPOUT_IMPL(T)                                                                    \
  template void BiasSoftmaxDropoutImpl<T>(                                                                          \
      cudaStream_t stream, T * output_data, bool* mask_data, T* softmax_output_data, const T* input_data,           \
      const T* bias_data, int element_count, int batch_count, int batch_stride, int bias_broadcast_count_per_batch, \
      const float ratio, PhiloxGenerator& generator);

SPECIALIZED_BIAS_SOFTMAX_DROPOUT_IMPL(double)
SPECIALIZED_BIAS_SOFTMAX_DROPOUT_IMPL(float)
SPECIALIZED_BIAS_SOFTMAX_DROPOUT_IMPL(half)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
