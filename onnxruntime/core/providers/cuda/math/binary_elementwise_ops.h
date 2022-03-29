// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/shared_inc/fast_divmod.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace cuda {

struct BinaryElementwisePreparation {
  const Tensor* lhs_tensor = nullptr;
  const Tensor* rhs_tensor = nullptr;
  Tensor* output_tensor = nullptr;
  TensorShapeVector output_dims;
  size_t rank;
  BroadcastIndexType lhs_index_type;
  BroadcastIndexType rhs_index_type;
  TensorShapeVector lhs_strides;
  TensorShapeVector rhs_strides;
  TensorShapeVector output_strides;

  BinaryElementwisePreparation() {}

  Status BinaryElementwiseBroadcastPrepareHelper(const TensorShape& lhs_shape, const TensorShape& rhs_shape,
                                                 const TensorShape& output_shape) {
    bool lhs_is_contiguous = true;
    bool rhs_is_contiguous = true;
    TensorShapeVector lhs_original_strides = TensorPitches(lhs_shape);
    TensorShapeVector rhs_original_strides = TensorPitches(rhs_shape);
#ifdef ENABLE_TRAINING
    if (lhs_tensor) {
      lhs_is_contiguous = lhs_tensor->IsContiguous();
      if (!lhs_is_contiguous) {
        ORT_ENFORCE(lhs_tensor->Shape() == lhs_shape);
        lhs_original_strides = ToShapeVector(lhs_tensor->Strides());
      }
    }
    if (rhs_tensor) {
      rhs_is_contiguous = rhs_tensor->IsContiguous();
      if (!rhs_is_contiguous) {
        ORT_ENFORCE(rhs_tensor->Shape() == rhs_shape);
        rhs_original_strides = ToShapeVector(rhs_tensor->Strides());
      }
    }
#endif
    lhs_index_type = lhs_shape.Size() == 1                            ? BroadcastIndexType::Scalar
                     : lhs_is_contiguous && lhs_shape == output_shape ? BroadcastIndexType::NoBroadcast
                                                                      : BroadcastIndexType::NeedCompute;
    rhs_index_type = rhs_shape.Size() == 1                            ? BroadcastIndexType::Scalar
                     : rhs_is_contiguous && rhs_shape == output_shape ? BroadcastIndexType::NoBroadcast
                                                                      : BroadcastIndexType::NeedCompute;
    rank = output_shape.NumDimensions();
    output_dims = output_shape.AsShapeVector();

    // Strides not needed if none of sizes needs compute.
    if (lhs_index_type != BroadcastIndexType::NeedCompute && rhs_index_type != BroadcastIndexType::NeedCompute) {
      return Status::OK();
    }

    size_t lhs_offset = rank - lhs_original_strides.size();
    size_t rhs_offset = rank - rhs_original_strides.size();
    lhs_strides.resize(rank, 0);
    for (size_t i = 0; i < lhs_original_strides.size(); ++i) {
      lhs_strides[lhs_offset + i] =
          (lhs_shape[i] == 1 && output_dims[lhs_offset + i] != 1) ? 0 : lhs_original_strides[i];
    }
    rhs_strides.resize(rank, 0);
    for (size_t i = 0; i < rhs_original_strides.size(); ++i) {
      rhs_strides[rhs_offset + i] =
          (rhs_shape[i] == 1 && output_dims[rhs_offset + i] != 1) ? 0 : rhs_original_strides[i];
    }

    // Coalesce the dimensions.
    if (rank > 1) {
      // Reverse the shape and strides for better computation.
      TensorShapeVector reversed_shape(rank);
      TensorShapeVector lhs_reversed_strides(rank);
      TensorShapeVector rhs_reversed_strides(rank);
      for (size_t dim = 0; dim < rank; ++dim) {
        reversed_shape[dim] = output_dims[rank - 1 - dim];
        lhs_reversed_strides[dim] = lhs_strides[rank - 1 - dim];
        rhs_reversed_strides[dim] = rhs_strides[rank - 1 - dim];
      }

      // We can coalesce two adjacent dimensions if either dim has size 1 or if:
      // shape[n] * stride[n] == shape[n + 1].
      auto CanCoalesce = [&](size_t dim0, size_t dim1) {
        auto shape0 = reversed_shape[dim0];
        auto shape1 = reversed_shape[dim1];
        if (shape0 == 1 || shape1 == 1) {
          return true;
        }
        return shape0 * lhs_reversed_strides[dim0] == lhs_reversed_strides[dim1] &&
               shape0 * rhs_reversed_strides[dim0] == rhs_reversed_strides[dim1];
      };

      // Replace each operands stride at dim0 with its stride at dim1.
      auto ReplaceStride = [&](size_t dim0, size_t dim1) {
        lhs_reversed_strides[dim0] = lhs_reversed_strides[dim1];
        rhs_reversed_strides[dim0] = rhs_reversed_strides[dim1];
      };

      size_t prev_dim = 0;
      for (size_t dim = 1; dim < rank; ++dim) {
        if (CanCoalesce(prev_dim, dim)) {
          if (reversed_shape[prev_dim] == 1) {
            ReplaceStride(prev_dim, dim);
          }
          reversed_shape[prev_dim] *= reversed_shape[dim];
        } else {
          prev_dim++;
          if (prev_dim != dim) {
            ReplaceStride(prev_dim, dim);
            reversed_shape[prev_dim] = reversed_shape[dim];
          }
        }
      }

      rank = prev_dim + 1;
      reversed_shape.resize(rank);
      lhs_reversed_strides.resize(rank);
      rhs_reversed_strides.resize(rank);

      // Reverse the shape and strides back.
      output_dims.resize(rank);
      lhs_strides.resize(rank);
      rhs_strides.resize(rank);
      for (size_t dim = 0; dim < rank; ++dim) {
        output_dims[dim] = reversed_shape[rank - 1 - dim];
        lhs_strides[dim] = lhs_reversed_strides[rank - 1 - dim];
        rhs_strides[dim] = rhs_reversed_strides[rank - 1 - dim];
      }
    }

    output_strides.resize(rank);
    int64_t running_size = 1;
    for (size_t dim = 0; dim < rank; ++dim) {
      output_strides[rank - 1 - dim] = running_size;
      running_size *= output_dims[rank - 1 - dim];
    }

    return Status::OK();
  }
};

Status ComputeOutputShape(
    const std::string& node_name,
    const TensorShape& lhs_shape,
    const TensorShape& rhs_shape,
    TensorShape& out_shape);

Status BinaryElementwiseBroadcastPrepare(
    const Tensor* lhs_tensor,
    const Tensor* rhs_tensor,
    Tensor* output_tensor,
    BinaryElementwisePreparation* p,
    const TensorShape* override_lhs_shape = nullptr,
    const TensorShape* override_rhs_shape = nullptr);

// trait classes to indicate if the kernel supports broadcast
class ShouldBroadcast {
};

class ShouldNotBroadcast {
};

template <typename BroadcastTrait>
class BinaryElementwise : public CudaKernel {
 protected:
  typedef BroadcastTrait broadcast_type;

  BinaryElementwise(const OpKernelInfo& info) : CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext*) const override {
    return Status(common::ONNXRUNTIME, common::FAIL);  // should not reach here
  }
  Status Prepare(OpKernelContext* context, BinaryElementwisePreparation* p) const;
};

template <typename T>
class Add final : public BinaryElementwise<ShouldBroadcast> {
 public:
  Add(const OpKernelInfo& info) : BinaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Sub final : public BinaryElementwise<ShouldBroadcast> {
 public:
  Sub(const OpKernelInfo& info) : BinaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Mul final : public BinaryElementwise<ShouldBroadcast> {
 public:
  Mul(const OpKernelInfo& info) : BinaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Div final : public BinaryElementwise<ShouldBroadcast> {
 public:
  Div(const OpKernelInfo& info) : BinaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Pow_7 final : public BinaryElementwise<ShouldBroadcast> {
 public:
  Pow_7(const OpKernelInfo& info) : BinaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

// Since version 12
class Pow final : public BinaryElementwise<ShouldBroadcast> {
 public:
  Pow(const OpKernelInfo& info) : BinaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class And final : public BinaryElementwise<ShouldBroadcast> {
 public:
  And(const OpKernelInfo& info) : BinaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Or final : public BinaryElementwise<ShouldBroadcast> {
 public:
  Or(const OpKernelInfo& info) : BinaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Xor final : public BinaryElementwise<ShouldBroadcast> {
 public:
  Xor(const OpKernelInfo& info) : BinaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

// PRelu is activation function, but it's closer to binary elementwise ops in implementation
template <typename T>
class PRelu final : public BinaryElementwise<ShouldBroadcast> {
 public:
  PRelu(const OpKernelInfo& info) : BinaryElementwise(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T, typename CudaT>
class CompareFunction : public BinaryElementwise<ShouldBroadcast> {
 public:
  CompareFunction(const OpKernelInfo& info) : BinaryElementwise(info) {}

  typedef void (*ImplCompare)(cudaStream_t stream, size_t rank, BroadcastIndexType lhs_index_type,
                              BroadcastIndexType rhs_index_type, gsl::span<const int64_t> lhs_strides,
                              gsl::span<const int64_t> rhs_strides, gsl::span<const int64_t> output_shapes,
                              gsl::span<const int64_t> output_strides, const CudaT* lhs_data, const CudaT* rhs_data,
                              bool* output_data, size_t count);

  Status CompareMethod(OpKernelContext* context, ImplCompare Impl_Compare) const;
};

template <typename T>
class Greater final : public CompareFunction<T, typename ToCudaType<T>::MappedType> {
 public:
  Greater(const OpKernelInfo& info) : CompareFunction<T, typename ToCudaType<T>::MappedType>(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Equal final : public CompareFunction<T, typename ToCudaType<T>::MappedType> {
 public:
  Equal(const OpKernelInfo& info) : CompareFunction<T, typename ToCudaType<T>::MappedType>(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Less final : public CompareFunction<T, typename ToCudaType<T>::MappedType> {
 public:
  Less(const OpKernelInfo& info) : CompareFunction<T, typename ToCudaType<T>::MappedType>(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class GreaterOrEqual final : public CompareFunction<T, typename ToCudaType<T>::MappedType> {
 public:
  GreaterOrEqual(const OpKernelInfo& info) : CompareFunction<T, typename ToCudaType<T>::MappedType>(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class LessOrEqual final : public CompareFunction<T, typename ToCudaType<T>::MappedType> {
 public:
  LessOrEqual(const OpKernelInfo& info) : CompareFunction<T, typename ToCudaType<T>::MappedType>(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
