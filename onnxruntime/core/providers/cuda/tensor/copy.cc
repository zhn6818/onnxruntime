// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/copy.h"

#include "core/framework/shape_utils.h"
#include "core/providers/cuda/tensor/copy_impl.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace cuda {

Status StridedCopyTensor(cudaStream_t stream, const Tensor& src, Tensor& dst) {
  auto src_strides = ToShapeVector(src.Strides());
  auto dst_strides = ToShapeVector(dst.Strides());
  auto dst_shape = dst.Shape().AsShapeVector();
  CoalesceDimensions({src_strides, dst_strides}, dst_shape);
  const void* src_raw = src.DataRaw();
  void* dst_raw = dst.MutableDataRaw();
  size_t rank = dst_shape.size();
  if (rank == 1) {
    // Contiguous case.
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_raw, src_raw, src.SizeInBytes(), cudaMemcpyDeviceToDevice, stream));
  } else {
    TArray<int64_t> src_strides_array(src_strides);
    TArray<int64_t> dst_strides_array(dst_strides);
    TensorPitches dst_strides_full(dst_shape);
    TArray<fast_divmod> dst_fdms(static_cast<int>(rank));
    for (int i = 0; i < static_cast<int>(rank); ++i) {
      dst_fdms[i] = fast_divmod(static_cast<int>(dst_strides_full[i]));
    }
    size_t output_size = static_cast<size_t>(dst.Shape().Size());
    switch (src.DataType()->Size()) {
#define CASE_DATATYPE(type)                                                                                  \
  case sizeof(type): {                                                                                       \
    typedef typename ToCudaType<type>::MappedType CudaT;                                                     \
    const CudaT* src_data = reinterpret_cast<const CudaT*>(src_raw);                                         \
    CudaT* dst_data = reinterpret_cast<CudaT*>(dst_raw);                                                     \
    StridedCopyImpl<CudaT>(stream, rank, src_data, src_strides_array, dst_data, dst_strides_array, dst_fdms, \
                           output_size);                                                                     \
  } break
      CASE_DATATYPE(int8_t);
      CASE_DATATYPE(int16_t);
      CASE_DATATYPE(int32_t);
      CASE_DATATYPE(int64_t);
      default:
        ORT_THROW("Unsupported element size by StridedCopyTensor.");
#undef CASE_DATATYPE
    }
  }
  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
