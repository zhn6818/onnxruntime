// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace cuda {

template <typename T>
void SoftmaxDropoutGradImpl(cudaStream_t stream, T* input_grad_data, const T* output_grad_data, const bool* mask_data,
                            const T* softmax_output_data, int element_count, int batch_count, int batch_stride,
                            const float ratio);

}  // namespace cuda
}  // namespace onnxruntime
