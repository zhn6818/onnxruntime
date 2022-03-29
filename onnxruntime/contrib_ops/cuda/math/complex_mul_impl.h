// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "complex_mul.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace ::onnxruntime::cuda;

template <typename T>
void ComplexMul_Impl(cudaStream_t stream, size_t rank, BroadcastIndexType lhs_index_type,
                     BroadcastIndexType rhs_index_type, gsl::span<const int64_t> lhs_strides,
                     gsl::span<const int64_t> rhs_strides, gsl::span<const int64_t> output_shapes,
                     gsl::span<const int64_t> output_strides, const T* lhs_data, const T* rhs_data, T* output_data,
                     size_t count, int64_t lhs_size, int64_t rhs_size, bool is_conj);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
