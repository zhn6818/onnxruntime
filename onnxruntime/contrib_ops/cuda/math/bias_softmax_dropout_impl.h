// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/random_generator.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
void BiasSoftmaxDropoutImpl(cudaStream_t stream, T* output_data, bool* mask_data, T* softmax_output_data,
                            const T* input_data, const T* bias_data, int element_count, int batch_count,
                            int batch_stride, int bias_broadcast_count_per_batch, const float ratio,
                            PhiloxGenerator& generator);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
