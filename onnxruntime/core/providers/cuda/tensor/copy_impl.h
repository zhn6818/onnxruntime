// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
void StridedCopyImpl(cudaStream_t stream, size_t rank, const T* src_data, const TArray<int64_t>& src_strides,
                     T* dst_data, const TArray<int64_t>& dst_strides, const TArray<fast_divmod>& dst_fdms,
                     const size_t output_size);

}  // namespace cuda
}  // namespace onnxruntime
