// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

enum class PerChannelType : int32_t {
  None = (int32_t)0,
  LhsNeedCompute = (int32_t)1,
  RhsNeedCompute = (int32_t)2,
};

struct BinaryElementwiseArgs {
  size_t rank = 0;
  BroadcastIndexType lhs_index_type;
  BroadcastIndexType rhs_index_type;
  TArray<int64_t> lhs_strides;
  TArray<int64_t> rhs_strides;
  TArray<fast_divmod> output_fdms;
  size_t output_size;

  // Optimization for case Op([N,C,H],[C,1]).
  PerChannelType per_channel_type = PerChannelType::None;
  int batch;
  int channel;
  int height;
};

}  // namespace cuda
}  // namespace onnxruntime
