// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <functional>
#include "core/framework/tensor.h"

// Example usage:
//  #include "core/framework/data_types_internal.h"
//
//  void DumpCpuTensor(const onnxruntime::Tensor& tensor, int threshold = 512, int edge_items = 6) {
//     DispatchOnTensorType(tensor.DataType(), PrintCpuTensor, tensor, threshold, edge_items);
//  }

namespace onnxruntime {
namespace utils {

// Skip non edge items in last dimension
#define SKIP_NON_EDGE_ITEMS_LAST_DIM(dim_size, index, edge_items)                          \
  if (dim_size > 2 * edge_items && index >= edge_items && index + edge_items < dim_size) { \
    if (index == edge_items) {                                                             \
      std::cout << ", ... ";                                                               \
    }                                                                                      \
    continue;                                                                              \
  }

// Skip non edge items in other dimensions except the last dimension
#define SKIP_NON_EDGE_ITEMS(dim_size, index, edge_items)                                   \
  if (dim_size > 2 * edge_items && index >= edge_items && index + edge_items < dim_size) { \
    if (index == edge_items) {                                                             \
      std::cout << "..." << std::endl;                                                     \
    }                                                                                      \
    continue;                                                                              \
  }

// Print 2D tensor snippet
template <typename T>
void PrintCpuTensorSnippet(const T* tensor, int64_t dim0, int64_t dim1, int64_t edge_items, const std::function<void(const T&)>& print) {
  for (int64_t i = 0; i < dim0; i++) {
    SKIP_NON_EDGE_ITEMS(dim0, i, edge_items);
    print(tensor[i * dim1]);
    for (int64_t j = 1; j < dim1; j++) {
      SKIP_NON_EDGE_ITEMS_LAST_DIM(dim1, j, edge_items);
      std::cout << ", ";
      print(tensor[i * dim1 + j]);
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

// Print 3D tensor
template <typename T>
void PrintCpuTensorSnippet(const T* tensor, int64_t dim0, int64_t dim1, int64_t dim2, int64_t edge_items, const std::function<void(const T&)>& print) {
  for (int64_t i = 0; i < dim0; i++) {
    SKIP_NON_EDGE_ITEMS(dim0, i, edge_items);
    for (int64_t j = 0; j < dim1; j++) {
      SKIP_NON_EDGE_ITEMS(dim1, j, edge_items);
      print(tensor[i * dim1 * dim2 + j * dim2]);
      for (int64_t k = 0; k < dim2; k++) {
        SKIP_NON_EDGE_ITEMS_LAST_DIM(dim2, k, edge_items);
        std::cout << ", ";
        print(tensor[i * dim1 * dim2 + j * dim2 + k]);
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

// Print 2D tensor
template <typename T>
void PrintCpuTensor(const T* tensor, int64_t dim0, int64_t dim1, const std::function<void(const T&)>& print) {
  for (int64_t i = 0; i < dim0; i++) {
    print(tensor[i * dim1]);
    for (int64_t j = 1; j < dim1; j++) {
      std::cout << ", ";
      print(tensor[i * dim1 + j]);
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

// Print 3D tensor
template <typename T>
void PrintCpuTensor(const T* tensor, int64_t dim0, int64_t dim1, int64_t dim2, const std::function<void(const T&)>& print) {
  for (int64_t i = 0; i < dim0; i++) {
    for (int64_t j = 0; j < dim1; j++) {
      print(tensor[i * dim1 * dim2 + j * dim2]);
      for (int64_t k = 0; k < dim2; k++) {
        std::cout << ", ";
        print(tensor[i * dim1 * dim2 + j * dim2 + k]);
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

template <typename T>
void PrintCpuTensor(const onnxruntime::Tensor& tensor, int threshold, int edge_items) {
  const auto& shape = tensor.Shape();
  auto num_items = shape.Size();

  if (num_items == 0) {
    std::cout << "no data";
    return;
  }

  size_t num_dims = shape.NumDimensions();
  size_t num_rows = 1;
  if (num_dims > 1) {
    num_rows = static_cast<size_t>(shape[0]);
  }

  size_t row_size = num_items / num_rows;

  auto data = tensor.Data<T>();

  auto print_val = [](const T& value) {
    if (std::is_floating_point<T>::value)
      std::cout << std::setprecision(8) << value;
    else
      std::cout << value;
  };

  if (threshold > 0 && static_cast<int64_t>(threshold) < num_items) {
    if (num_dims >= 3) {
      PrintCpuTensorSnippet<T>(data, shape.SizeToDimension(num_dims - 2), shape[num_dims - 2], shape[num_dims - 1], edge_items, print_val);
    } else {
      PrintCpuTensorSnippet<T>(data, num_rows, row_size, edge_items, print_val);
    }
  } else {
    if (num_dims >= 3) {
      PrintCpuTensor<T>(data, shape.SizeToDimension(num_dims - 2), shape[num_dims - 2], shape[num_dims - 1], print_val);
    } else {
      PrintCpuTensor<T>(data, num_rows, row_size, print_val);
    }
  }

  return;
}

}  // namespace utils
}  // namespace onnxruntime
