// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
void ImplDivGrad(cudaStream_t stream, size_t rank, BroadcastIndexType lhs_index_type, BroadcastIndexType rhs_index_type,
                 gsl::span<const int64_t> lhs_strides, gsl::span<const int64_t> rhs_strides,
                 gsl::span<const int64_t> output_shapes, gsl::span<const int64_t> output_strides, const T* a_data,
                 const T* b_data, const T* dy_data, T* da_output_data, T* db_output_data, size_t count);

}  // namespace cuda
}  // namespace onnxruntime
