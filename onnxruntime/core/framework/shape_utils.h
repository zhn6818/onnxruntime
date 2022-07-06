// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <initializer_list>
#include <functional>

#include "core/framework/tensor_shape.h"

namespace onnxruntime {

class Tensor;

/**
 * @brief Coalesce contiguous dimensions in the tensors. Operates inplace on the function arguments.
 *
 * @param tensors_strides Strides of tensors.
 * @param shape  Output tensor shape.
 */
void CoalesceDimensions(std::initializer_list<std::reference_wrapper<TensorShapeVector>>&& tensors_strides,
                        TensorShapeVector& shape);

TensorShapeVector StridesForTensor(const Tensor& tensor);

}  // namespace onnxruntime
