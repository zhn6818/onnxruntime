//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/shape_utils.h"
#include "core/framework/tensor.h"

namespace onnxruntime {

void CoalesceDimensions(std::initializer_list<std::reference_wrapper<TensorShapeVector>>&& tensors_strides,
                        TensorShapeVector& shape) {
  const std::size_t dims = shape.size();
  if (dims <= 1) return;

  // We can coalesce two adjacent dimensions if either dim has size 1 or if:
  // shape[src] * stride[src] == shape[dst].
  auto CanCoalesce = [&](int dst, int src) {
    auto shape_dst = shape[dst];
    auto shape_src = shape[src];
    if (shape_dst == 1 || shape_src == 1) {
      return true;
    }
    for (const auto& r_tensor_strides : tensors_strides) {
      auto& tensor_strides = r_tensor_strides.get();
      if (shape_src * tensor_strides[src] != tensor_strides[dst]) {
        return false;
      }
    }
    return true;
  };

  // replace each operands stride at dst with its stride at src
  auto ReplaceStride = [&](int dst, int src) {
    for (const auto& r_tensor_strides : tensors_strides) {
      auto& tensor_strides = r_tensor_strides.get();
      tensor_strides[dst] = tensor_strides[src];
    }
  };

  // the current dimension is the one we are attempting to "coalesce onto"
  std::size_t current_dim = 0;
  for (std::size_t dim = 1; dim < dims; dim++) {
    // check if dim can be coalesced with current_dim
    if (CanCoalesce(current_dim, dim)) {
      if (shape[dim] != 1) {
        ReplaceStride(current_dim, dim);
      }
      shape[current_dim] *= shape[dim];
    } else {
      current_dim++;
      if (current_dim != dim) {
        // we have coalesced at least one value before this: bump forward the values into the correct place
        ReplaceStride(current_dim, dim);
        shape[current_dim] = shape[dim];
      }
    }
  }

  shape.resize(current_dim + 1);
  for (const auto& r_tensor_strides : tensors_strides) {
    auto& tensor_strides = r_tensor_strides.get();
    tensor_strides.resize(current_dim + 1);
  }
}

TensorShapeVector StridesForTensor(const Tensor& tensor) {
#ifdef ENABLE_TRAINING
  return ToShapeVector(tensor.Strides());
#else
  const auto& shape = tensor.Shape();
  TensorShapeVector strides(shape.NumDimensions());
  int64_t running_size = 1;
  for (auto i = shape.NumDimensions(); i > 0; i--) {
    strides[i - 1] = running_size;
    running_size *= shape[i - 1];
  }
  return strides;
#endif
}

}  // namespace onnxruntime
