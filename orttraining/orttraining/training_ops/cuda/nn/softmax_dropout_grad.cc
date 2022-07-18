// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/nn/softmax_dropout_grad.h"

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/common.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "orttraining/training_ops/cuda/nn/softmax_dropout_grad_impl.h"

namespace onnxruntime {
namespace cuda {

namespace {

template <typename T>
struct GetRatioDataImpl {
  void operator()(const Tensor* ratio, float& ratio_data) const {
    ratio_data = static_cast<float>(*(ratio->template Data<T>()));
    ORT_ENFORCE(ratio_data >= 0.0f && ratio_data < 1.0f, "ratio_data is outside range [0, 1)");
  }
};

template <typename T>
struct DispatchSoftmaxDropoutGradImpl {
  void operator()(cudaStream_t stream, Tensor* dX, const Tensor* dY, const Tensor* mask, const Tensor* softmax_Y,
                  int element_count, int batch_count, int batch_stride, const float ratio) {
    typedef typename ToCudaType<T>::MappedType CudaT;
    CudaT* input_grad_data = reinterpret_cast<CudaT*>(dX->template MutableData<T>());
    const CudaT* output_grad_data = reinterpret_cast<const CudaT*>(dY->template Data<T>());
    const bool* mask_data = reinterpret_cast<const bool*>(mask->template Data<bool>());
    const CudaT* softmax_output_data = reinterpret_cast<const CudaT*>(softmax_Y->template Data<T>());
    SoftmaxDropoutGradImpl(stream, input_grad_data, output_grad_data, mask_data, softmax_output_data, element_count,
                           batch_count, batch_stride, ratio);
  }
};

}  // namespace

ONNX_OPERATOR_KERNEL_EX(SoftmaxDropoutGrad, kMSDomain, 1, kCudaExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .TypeConstraint("T", BuildKernelDefConstraints<MLFloat16, float, double>())
                            .TypeConstraint("T1", BuildKernelDefConstraints<MLFloat16, float, double>())
                            .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>())
                            .InputMemoryType(OrtMemTypeCPUInput, 3),
                        SoftmaxDropoutGrad);

Status SoftmaxDropoutGrad::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* dY = ctx->Input<Tensor>(0);
  const TensorShape& input_shape = dY->Shape();
  const Tensor* mask = ctx->Input<Tensor>(1);
  const Tensor* softmax_Y = ctx->Input<Tensor>(2);

  const int axis = static_cast<int>(HandleNegativeAxis(axis_, input_shape.NumDimensions()));
  const int N = static_cast<int>(input_shape.SizeToDimension(axis));
  const int D = static_cast<int>(input_shape.SizeFromDimension(axis));

  // Get the ratio_data
  float ratio_data = default_ratio_;
  auto ratio = ctx->Input<Tensor>(3);
  if (ratio) {
    utils::MLTypeCallDispatcher<float, MLFloat16, double> ratio_t_disp(ratio->GetElementType());
    ratio_t_disp.Invoke<GetRatioDataImpl>(ratio, ratio_data);
  }

  Tensor* dX = ctx->Output(0, input_shape);
  utils::MLTypeCallDispatcher<double, float, MLFloat16> t_disp(dY->GetElementType());
  t_disp.Invoke<DispatchSoftmaxDropoutGradImpl>(Stream(), dX, dY, mask, softmax_Y, D, N, D, ratio_data);
  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
