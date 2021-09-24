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
  "onnxruntime/core/providers/cuda/bert/attention.cc"
  "onnxruntime/core/providers/cuda/bert/attention.h"
  "onnxruntime/core/providers/cuda/bert/attention_impl.cu"
  "onnxruntime/core/providers/cuda/bert/attention_impl.h"
  "onnxruntime/core/providers/cuda/bert/attention_transpose.cu"
  "onnxruntime/core/providers/cuda/bert/attention_past.cu"
  "onnxruntime/core/providers/cuda/bert/embed_layer_norm.cc"
  "onnxruntime/core/providers/cuda/bert/embed_layer_norm.h"
  "onnxruntime/core/providers/cuda/bert/embed_layer_norm_impl.cu"
  "onnxruntime/core/providers/cuda/bert/embed_layer_norm_impl.h"
  "onnxruntime/core/providers/cuda/bert/fast_gelu_impl.cu"
  "onnxruntime/core/providers/cuda/bert/layer_norm.cuh"
  "onnxruntime/core/providers/cuda/bert/longformer_attention.cc"
  "onnxruntime/core/providers/cuda/bert/longformer_attention.h"
  "onnxruntime/core/providers/cuda/bert/longformer_attention_softmax.cu"
  "onnxruntime/core/providers/cuda/bert/longformer_attention_softmax.h"
  "onnxruntime/core/providers/cuda/bert/longformer_attention_impl.cu"
  "onnxruntime/core/providers/cuda/bert/longformer_attention_impl.h"
  "onnxruntime/core/providers/cuda/bert/longformer_global_impl.cu"
  "onnxruntime/core/providers/cuda/bert/longformer_global_impl.h"
  "onnxruntime/core/providers/cuda/math/bias_softmax.cc"
  "onnxruntime/core/providers/cuda/math/bias_softmax.h"
  "onnxruntime/core/providers/cuda/math/bias_softmax_impl.cu"
  "onnxruntime/core/providers/cuda/math/complex_mul.cc"
  "onnxruntime/core/providers/cuda/math/complex_mul.h"
  "onnxruntime/core/providers/cuda/math/complex_mul_impl.cu"
  "onnxruntime/core/providers/cuda/math/complex_mul_impl.h"
  "onnxruntime/core/providers/cuda/math/cufft_plan_cache.h"
  "onnxruntime/core/providers/cuda/math/fft_ops.cc"
  "onnxruntime/core/providers/cuda/math/fft_ops.h"
  "onnxruntime/core/providers/cuda/math/fft_ops_impl.cu"
  "onnxruntime/core/providers/cuda/math/fft_ops_impl.h"
  "onnxruntime/core/providers/cuda/quantization/attention_quantization.cc"
  "onnxruntime/core/providers/cuda/quantization/attention_quantization.h"
  "onnxruntime/core/providers/cuda/quantization/attention_quantization_impl.cu"
  "onnxruntime/core/providers/cuda/quantization/attention_quantization_impl.cuh"
  "onnxruntime/core/providers/cuda/quantization/quantize_dequantize_linear.cc"
  "onnxruntime/core/providers/cuda/tensor/crop.cc"
  "onnxruntime/core/providers/cuda/tensor/crop.h"
  "onnxruntime/core/providers/cuda/tensor/crop_impl.cu"
  "onnxruntime/core/providers/cuda/tensor/crop_impl.h"
  "onnxruntime/core/providers/cuda/tensor/dynamicslice.cc"
  "onnxruntime/core/providers/cuda/tensor/image_scaler.cc"
  "onnxruntime/core/providers/cuda/tensor/image_scaler.h"
  "onnxruntime/core/providers/cuda/tensor/image_scaler_impl.cu"
  "onnxruntime/core/providers/cuda/tensor/image_scaler_impl.h"
  "onnxruntime/core/providers/cuda/conv_transpose_with_dynamic_pads.cc"
  "onnxruntime/core/providers/cuda/conv_transpose_with_dynamic_pads.h"
  "onnxruntime/core/providers/cuda/cuda_contrib_kernels.cc"
  "onnxruntime/core/providers/cuda/cuda_contrib_kernels.h"
  "onnxruntime/core/providers/cuda/inverse.cc"
  "onnxruntime/core/providers/cuda/fused_conv.cc"
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

hipify("${_amd_onnxruntime_cuda_contrib_ops_srcs}" onnxruntime_rocm_generated_contrib_ops_srcs)

set(onnxruntime_rocm_generated_contrib_ops_cc_srcs "${onnxruntime_rocm_generated_contrib_ops_srcs}")
list(FILTER onnxruntime_rocm_generated_contrib_ops_cc_srcs INCLUDE REGEX ".*\.cc$") 

set(onnxruntime_rocm_generated_contrib_ops_cu_srcs "${onnxruntime_rocm_generated_contrib_ops_srcs}")
list(FILTER onnxruntime_rocm_generated_contrib_ops_cu_srcs INCLUDE REGEX ".*\.cu$") 


# CUDA training ops
file(GLOB_RECURSE _amd_onnxruntime_cuda_training_ops_srcs RELATIVE ${PROJECT_ROOT} CONFIGURE_DEPENDS
  "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/*"
)

hipify("${_amd_onnxruntime_cuda_training_ops_srcs}" onnxruntime_rocm_generated_training_ops_cc_srcs)

set(onnxruntime_rocm_generated_training_ops_cc_srcs "${_amd_onnxruntime_cuda_training_ops_srcs}")
list(FILTER onnxruntime_rocm_generated_training_ops_cc_srcs INCLUDE REGEX ".*\.cc$") 

set(onnxruntime_rocm_generated_training_ops_cu_srcs "${_amd_onnxruntime_cuda_training_ops_srcs}")
list(FILTER onnxruntime_rocm_generated_training_ops_cu_srcs INCLUDE REGEX ".*\.cu$") 


