# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(PROJECT_ROOT ${PROJECT_SOURCE_DIR}/..)

function(hipify cuda_src_files rocm_src_files) 
  set(${rocm_src_files} PARENT_SCOPE)
  foreach(cuda_src_relative_path ${cuda_src_files})
    string(REPLACE "cuda" "rocm" rocm_src_relative_path ${cuda_src_relative_path})
    set(cuda_src_absolute_path ${ONNXRUNTIME_ROOT}/${cuda_src_relative_path})
    set(rocm_src_absolute_path ${CMAKE_CURRENT_BINARY_DIR}/amdgpu/${rocm_src_relative_path})
    message(STATUS "hipify ${rocm_src_absolute_path}")
    add_custom_command(
            OUTPUT ${rocm_src_absolute_path}
            COMMAND tools/ci_build/amd_hipify_file.py ${cuda_src_absolute_path} ${rocm_src_absolute_path}
            DEPENDS ${cuda_src_absolute_path}
            VERBATIM
    )
    list(APPEND ${rocm_src_files} ${rocm_src_absolute_path})
  endforeach()
  set(${rocm_src_files} ${${rocm_src_files}} PARENT_SCOPE)
endfunction()

# CUDA provider files
file(GLOB_RECURSE _amd_onnxruntime_providers_cuda_srcs RELATIVE ${PROJECT_ROOT} CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/core/providers/cuda/*"
)

list(REMOVE_ITEM _amd_onnxruntime_providers_cuda_srcs
  "onnxruntime/core/providers/atomic/common.cuh"
  "onnxruntime/core/providers/controlflow/if.cc"
  "onnxruntime/core/providers/controlflow/if.h"
  "onnxruntime/core/providers/controlflow/loop.cc"
  "onnxruntime/core/providers/controlflow/loop.h"
  "onnxruntime/core/providers/controlflow/scan.cc"
  "onnxruntime/core/providers/controlflow/scan.h"
  "onnxruntime/core/providers/cu_inc/common.cuh"
  "onnxruntime/core/providers/math/einsum_utils/einsum_auxiliary_ops.cc"
  "onnxruntime/core/providers/math/einsum_utils/einsum_auxiliary_ops.h"
  "onnxruntime/core/providers/math/einsum_utils/einsum_auxiliary_ops_diagonal.cu"
  "onnxruntime/core/providers/math/einsum_utils/einsum_auxiliary_ops_diagonal.h"
  "onnxruntime/core/providers/math/einsum.cc"
  "onnxruntime/core/providers/math/einsum.h"
  "onnxruntime/core/providers/math/gemm.cc"
  "onnxruntime/core/providers/math/matmul.cc"
  "onnxruntime/core/providers/math/matmul_integer.cc"
  "onnxruntime/core/providers/math/matmul_integer.cu"
  "onnxruntime/core/providers/math/matmul_integer.cuh"
  "onnxruntime/core/providers/math/matmul_integer.h"
  "onnxruntime/core/providers/math/softmax_impl.cu"
  "onnxruntime/core/providers/math/softmax.cc"
  "onnxruntime/core/providers/math/topk.cc"
  "onnxruntime/core/providers/math/topk.h"
  "onnxruntime/core/providers/math/topk_impl.cu"
  "onnxruntime/core/providers/math/topk_impl.h"
  "onnxruntime/core/providers/nn/batch_norm.cc"
  "onnxruntime/core/providers/nn/batch_norm.h"
  "onnxruntime/core/providers/nn/conv.cc"
  "onnxruntime/core/providers/nn/conv.h"
  "onnxruntime/core/providers/nn/conv_transpose.cc"
  "onnxruntime/core/providers/nn/conv_transpose.h"
  "onnxruntime/core/providers/nn/instance_norm.cc"
  "onnxruntime/core/providers/nn/instance_norm.h"
  "onnxruntime/core/providers/nn/instance_norm_impl.cu"
  "onnxruntime/core/providers/nn/instance_norm_impl.h"
  "onnxruntime/core/providers/nn/lrn.cc"
  "onnxruntime/core/providers/nn/lrn.h"
  "onnxruntime/core/providers/nn/max_pool_with_index.cu"
  "onnxruntime/core/providers/nn/max_pool_with_index.h"
  "onnxruntime/core/providers/nn/pool.cc"
  "onnxruntime/core/providers/nn/pool.h"
  "onnxruntime/core/providers/object_detection/non_max_suppression.cc"
  "onnxruntime/core/providers/object_detection/non_max_suppression.h"
  "onnxruntime/core/providers/object_detection/non_max_suppression_impl.cu"
  "onnxruntime/core/providers/object_detection/non_max_suppression_impl.h"
  "onnxruntime/core/providers/object_detection/roialign.cc"
  "onnxruntime/core/providers/object_detection/roialign.h"
  "onnxruntime/core/providers/object_detection/roialign_impl.cu"
  "onnxruntime/core/providers/object_detection/roialign_impl.h"
)

hipify("${_amd_onnxruntime_providers_cuda_srcs}" onnxruntime_providers_rocm_generated_srcs)

set(onnxruntime_providers_rocm_generated_cc_srcs "${onnxruntime_providers_rocm_generated_srcs}")
list(FILTER onnxruntime_providers_rocm_generated_cc_srcs INCLUDE REGEX ".*\.cc$") 

set(onnxruntime_providers_rocm_generated_cu_srcs "${onnxruntime_providers_rocm_generated_srcs}")
list(FILTER onnxruntime_providers_rocm_generated_cu_srcs INCLUDE REGEX ".*\.cu$") 

# CUDA contrib ops
file(GLOB_RECURSE _amd_onnxruntime_cuda_contrib_ops_cc_srcs RELATIVE ${PROJECT_ROOT} CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/*"
)

list(REMOVE_ITEM _amd_onnxruntime_cuda_contrib_ops_cc_srcs
  "onnxruntime/contrib_ops/cuda/bert/attention.cc"
  "onnxruntime/contrib_ops/cuda/bert/attention.h"
  "onnxruntime/contrib_ops/cuda/bert/attention_impl.cu"
  "onnxruntime/contrib_ops/cuda/bert/attention_impl.h"
  "onnxruntime/contrib_ops/cuda/bert/attention_transpose.cu"
  "onnxruntime/contrib_ops/cuda/bert/attention_past.cu"
  "onnxruntime/contrib_ops/cuda/bert/embed_layer_norm.cc"
  "onnxruntime/contrib_ops/cuda/bert/embed_layer_norm.h"
  "onnxruntime/contrib_ops/cuda/bert/embed_layer_norm_impl.cu"
  "onnxruntime/contrib_ops/cuda/bert/embed_layer_norm_impl.h"
  "onnxruntime/contrib_ops/cuda/bert/fast_gelu_impl.cu"
  "onnxruntime/contrib_ops/cuda/bert/layer_norm.cuh"
  "onnxruntime/contrib_ops/cuda/bert/longformer_attention.cc"
  "onnxruntime/contrib_ops/cuda/bert/longformer_attention.h"
  "onnxruntime/contrib_ops/cuda/bert/longformer_attention_softmax.cu"
  "onnxruntime/contrib_ops/cuda/bert/longformer_attention_softmax.h"
  "onnxruntime/contrib_ops/cuda/bert/longformer_attention_impl.cu"
  "onnxruntime/contrib_ops/cuda/bert/longformer_attention_impl.h"
  "onnxruntime/contrib_ops/cuda/bert/longformer_global_impl.cu"
  "onnxruntime/contrib_ops/cuda/bert/longformer_global_impl.h"
  "onnxruntime/contrib_ops/cuda/math/bias_softmax.cc"
  "onnxruntime/contrib_ops/cuda/math/bias_softmax.h"
  "onnxruntime/contrib_ops/cuda/math/bias_softmax_impl.cu"
  "onnxruntime/contrib_ops/cuda/math/complex_mul.cc"
  "onnxruntime/contrib_ops/cuda/math/complex_mul.h"
  "onnxruntime/contrib_ops/cuda/math/complex_mul_impl.cu"
  "onnxruntime/contrib_ops/cuda/math/complex_mul_impl.h"
  "onnxruntime/contrib_ops/cuda/math/cufft_plan_cache.h"
  "onnxruntime/contrib_ops/cuda/math/fft_ops.cc"
  "onnxruntime/contrib_ops/cuda/math/fft_ops.h"
  "onnxruntime/contrib_ops/cuda/math/fft_ops_impl.cu"
  "onnxruntime/contrib_ops/cuda/math/fft_ops_impl.h"
  "onnxruntime/contrib_ops/cuda/quantization/attention_quantization.cc"
  "onnxruntime/contrib_ops/cuda/quantization/attention_quantization.h"
  "onnxruntime/contrib_ops/cuda/quantization/attention_quantization_impl.cu"
  "onnxruntime/contrib_ops/cuda/quantization/attention_quantization_impl.cuh"
  "onnxruntime/contrib_ops/cuda/quantization/quantize_dequantize_linear.cc"
  "onnxruntime/contrib_ops/cuda/tensor/crop.cc"
  "onnxruntime/contrib_ops/cuda/tensor/crop.h"
  "onnxruntime/contrib_ops/cuda/tensor/crop_impl.cu"
  "onnxruntime/contrib_ops/cuda/tensor/crop_impl.h"
  "onnxruntime/contrib_ops/cuda/tensor/dynamicslice.cc"
  "onnxruntime/contrib_ops/cuda/tensor/image_scaler.cc"
  "onnxruntime/contrib_ops/cuda/tensor/image_scaler.h"
  "onnxruntime/contrib_ops/cuda/tensor/image_scaler_impl.cu"
  "onnxruntime/contrib_ops/cuda/tensor/image_scaler_impl.h"
  "onnxruntime/contrib_ops/cuda/conv_transpose_with_dynamic_pads.cc"
  "onnxruntime/contrib_ops/cuda/conv_transpose_with_dynamic_pads.h"
  "onnxruntime/contrib_ops/cuda/cuda_contrib_kernels.cc"
  "onnxruntime/contrib_ops/cuda/cuda_contrib_kernels.h"
  "onnxruntime/contrib_ops/cuda/inverse.cc"
  "onnxruntime/contrib_ops/cuda/fused_conv.cc"
)

hipify("${_amd_onnxruntime_cuda_contrib_ops_srcs}" onnxruntime_rocm_generated_contrib_ops_srcs)

set(onnxruntime_rocm_generated_contrib_ops_cc_srcs "${onnxruntime_rocm_generated_contrib_ops_srcs}")
list(FILTER onnxruntime_rocm_generated_contrib_ops_cc_srcs INCLUDE REGEX ".*\.cc$") 

set(onnxruntime_rocm_generated_contrib_ops_cu_srcs "${onnxruntime_rocm_generated_contrib_ops_srcs}")
list(FILTER onnxruntime_rocm_generated_contrib_ops_cu_srcs INCLUDE REGEX ".*\.cu$") 


# CUDA training ops
file(GLOB_RECURSE _amd_onnxruntime_cuda_training_ops_srcs RELATIVE ${PROJECT_ROOT} CONFIGURE_DEPENDS
  "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/*"
)


list(REMOVE_ITEM _amd_onnxruntime_cuda_training_ops_srcs
  "orttraining/orttraining/training_ops/activation/gelu_grad_impl_common.cuh"
  "orttraining/orttraining/training_ops/collective/adasum_kernels.cc"
  "orttraining/orttraining/training_ops/collective/adasum_kernels.h"
  "orttraining/orttraining/training_ops/collective/nccl_common.cc"
  "orttraining/orttraining/training_ops/collective/ready_event.cc"
  "orttraining/orttraining/training_ops/collective/ready_event.h"
  "orttraining/orttraining/training_ops/communication/common.h"
  "orttraining/orttraining/training_ops/communication/nccl_service.cc"
  "orttraining/orttraining/training_ops/communication/nccl_service.h"
  "orttraining/orttraining/training_ops/communication/recv.cc"
  "orttraining/orttraining/training_ops/communication/recv.h"
  "orttraining/orttraining/training_ops/communication/send.cc"
  "orttraining/orttraining/training_ops/communication/send.h"
  "orttraining/orttraining/training_ops/controlflow/record.cc"
  "orttraining/orttraining/training_ops/controlflow/record.h"
  "orttraining/orttraining/training_ops/controlflow/wait.cc"
  "orttraining/orttraining/training_ops/controlflow/wait.h"
  "orttraining/orttraining/training_ops/math/div_grad.cc"
  "orttraining/orttraining/training_ops/math/softmax_grad_impl.cu"
  "orttraining/orttraining/training_ops/math/softmax_grad.cc"
  "orttraining/orttraining/training_ops/nn/batch_norm_grad.cc"
  "orttraining/orttraining/training_ops/nn/batch_norm_grad.h"
  "orttraining/orttraining/training_ops/nn/batch_norm_internal.cc"
  "orttraining/orttraining/training_ops/nn/batch_norm_internal.h"
  "orttraining/orttraining/training_ops/nn/conv_grad.cc"
  "orttraining/orttraining/training_ops/nn/conv_grad.h"
  "orttraining/orttraining/training_ops/reduction/reduction_all.cc"
  "orttraining/orttraining/training_ops/reduction/reduction_ops.cc"
  "orttraining/orttraining/training_ops/tensor/gather_nd_grad_impl.cu"
  "orttraining/orttraining/training_ops/cuda_training_kernels.cc"
  "orttraining/orttraining/training_ops/cuda_training_kernels.h"
)

hipify("${_amd_onnxruntime_cuda_training_ops_srcs}" onnxruntime_rocm_generated_training_ops_cc_srcs)

set(onnxruntime_rocm_generated_training_ops_cc_srcs "${_amd_onnxruntime_cuda_training_ops_srcs}")
list(FILTER onnxruntime_rocm_generated_training_ops_cc_srcs INCLUDE REGEX ".*\.cc$") 

set(onnxruntime_rocm_generated_training_ops_cu_srcs "${_amd_onnxruntime_cuda_training_ops_srcs}")
list(FILTER onnxruntime_rocm_generated_training_ops_cu_srcs INCLUDE REGEX ".*\.cu$") 


