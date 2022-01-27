// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_execution_provider.h"
#include "contrib_ops/cuda/transformers/beam_search.h"
#include "beam_search_device_helper.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    BeamSearch,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)  // 'max_length' needs to be on CPU
        .InputMemoryType(OrtMemTypeCPUInput, 2)  // 'min_length' needs to be on CPU
        .InputMemoryType(OrtMemTypeCPUInput, 3)  // 'num_beams' needs to be on CPU
        .InputMemoryType(OrtMemTypeCPUInput, 4)  // 'num_return_sequences' needs to be on CPU
        .InputMemoryType(OrtMemTypeCPUInput, 5)  // 'temperature' needs to be on CPU
        .InputMemoryType(OrtMemTypeCPUInput, 6)  // 'length_penalty' needs to be on CPU
        .InputMemoryType(OrtMemTypeCPUInput, 7)  // 'repetition_penalty' needs to be on CPU
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                              DataTypeImpl::GetTensorType<MLFloat16>()}),
    BeamSearch);

BeamSearch::BeamSearch(const OpKernelInfo& info)
    : onnxruntime::contrib::transformers::BeamSearch(info) {
  SetComputeStream(static_cast<void*>(info.GetExecutionProvider()->GetComputeStream()));
  SetDeviceHelpers(BeamSearchCudaDeviceHelper::CreateInputs, BeamSearchCudaDeviceHelper::TopK);
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