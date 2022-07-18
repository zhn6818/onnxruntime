// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/bias_softmax_dropout.h"

#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/math/bias_softmax_dropout_impl.h"

using namespace onnxruntime;
using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
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
struct DispatchBiasSoftmaxDropoutImpl {
  void operator()(cudaStream_t stream, Tensor* Y, Tensor* mask, Tensor* softmax_Y, const Tensor* X, const Tensor* B,
                  int element_count, int batch_count, int batch_stride, int bias_broadcast_count_per_batch,
                  const float ratio, PhiloxGenerator& generator) {
    typedef typename ToCudaType<T>::MappedType CudaT;
    CudaT* output_data = reinterpret_cast<CudaT*>(Y->template MutableData<T>());
    bool* mask_data = reinterpret_cast<bool*>(mask->template MutableData<bool>());
    CudaT* softmax_output_data = reinterpret_cast<CudaT*>(softmax_Y->template MutableData<T>());
    const CudaT* input_data = reinterpret_cast<const CudaT*>(X->template Data<T>());
    const CudaT* bias_data = reinterpret_cast<const CudaT*>(B->template Data<T>());
    BiasSoftmaxDropoutImpl(stream, output_data, mask_data, softmax_output_data, input_data, bias_data, element_count,
                           batch_count, batch_stride, bias_broadcast_count_per_batch, ratio, generator);
  }
};

}  // namespace

ONNX_OPERATOR_KERNEL_EX(BiasSoftmaxDropout, kMSDomain, 1, kCudaExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .TypeConstraint("T", DataTypeImpl::AllIEEEFloatTensorTypes())
                            .TypeConstraint("T1", DataTypeImpl::AllIEEEFloatTensorTypes())
                            .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>())
                            .InputMemoryType(OrtMemTypeCPUInput, 2),
                        BiasSoftmaxDropout);

Status BiasSoftmaxDropout::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* X = ctx->Input<Tensor>(0);
  const Tensor* B = ctx->Input<Tensor>(1);
  const TensorShape& X_shape = X->Shape();

  const int softmax_axis = static_cast<int>(HandleNegativeAxis(softmax_axis_, X_shape.NumDimensions()));
  const int N = static_cast<int>(X_shape.SizeToDimension(softmax_axis));
  const int D = static_cast<int>(X_shape.SizeFromDimension(softmax_axis));

  const int broadcast_axis = static_cast<int>(HandleNegativeAxis(broadcast_axis_, X_shape.NumDimensions()));
  const int broadcast_size = N / static_cast<int>(X_shape.SizeToDimension(broadcast_axis));

  // Get the ratio_data
  float ratio_data = default_ratio_;
  auto ratio = ctx->Input<Tensor>(2);
  if (ratio) {
    utils::MLTypeCallDispatcher<float, MLFloat16, double> ratio_t_disp(ratio->GetElementType());
    ratio_t_disp.Invoke<GetRatioDataImpl>(ratio, ratio_data);
  }

  Tensor* Y = ctx->Output(0, X_shape);
  Tensor* mask = ctx->Output(1, X_shape);
  Tensor* softmax_Y = ctx->Output(2, X_shape);

  PhiloxGenerator& generator = generator_ ? *generator_ : PhiloxGenerator::Default();
  utils::MLTypeCallDispatcher<double, float, MLFloat16> t_disp(X->GetElementType());
  t_disp.Invoke<DispatchBiasSoftmaxDropoutImpl>(Stream(), Y, mask, softmax_Y, X, B, D, N, D, broadcast_size, ratio_data,
                                                generator);
  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
