// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/providers/cuda/cuda_pch.h"

namespace onnxruntime {

class Tensor;

namespace cuda {

Status StridedCopyTensor(cudaStream_t stream, const Tensor& src, Tensor& dst);

}  // namespace cuda
}  // namespace onnxruntime
