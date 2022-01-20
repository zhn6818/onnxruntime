// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_execution_provider.h"
#include "contrib_ops/cuda/transformers/beam_search.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    BeamSearch,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                              DataTypeImpl::GetTensorType<MLFloat16>()}),
    BeamSearch);

BeamSearch::BeamSearch(const OpKernelInfo& info)
    : onnxruntime::contrib::transformers::BeamSearch(info) {
  SetComputeStream(static_cast<void*>(info.GetExecutionProvider()->GetComputeStream()));
}

Status BeamSearch::ComputeInternal(OpKernelContext* context) const{
  return onnxruntime::contrib::transformers::BeamSearch::Compute(context);
}

Status BeamSearch::Compute(OpKernelContext* context) const {
  auto s = ComputeInternal(context);

  if (s.IsOK()) {
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "CUDA error ", cudaGetErrorName(err), ":", cudaGetErrorString(err));
    }
  }

  return s;
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime