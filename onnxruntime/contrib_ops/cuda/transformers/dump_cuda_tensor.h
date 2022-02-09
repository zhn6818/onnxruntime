// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/tensorprotoutils.h"
#include "core/framework/ort_value.h"
#include <string>
#include "contrib_ops/cpu/transformers/beam_search_shared.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace transformers {
/*
template <typename T>
void Dump2DTensor(cudaStream_t stream, const T* tensor, int dim0, int dim1, char title, char subtitle);

template <typename T>
void Dump3DTensor(cudaStream_t stream, const T* tensor, int dim0, int dim1, int dim2, char title, char subtitle);

template <typename T>
void Dump4DTensor(cudaStream_t stream, const T* tensor, int dim0, int dim1, int dim2, int dim3, char title, char subtitle);
*/

void DumpTensor(const char* name, const Tensor& tensor);

void DumpTensor(const char* name, const Tensor& tensor, int dim0, int dim1);

void DumpTensor(const char* name, const Tensor& tensor, int dim0, int dim1, int dim2);

template <typename T>
void DumpTensor(const char* name, const T* tensor, int dim0, int dim1);

template <typename T>
void DumpTensor(const char* name, const T* tensor, int dim0, int dim1, int dim2);


void DumpOrtValue(const char* name, const OrtValue& value);

void DisableTensorDump();

class CudaTensorConsoleDumper: public onnxruntime::contrib::transformers::IConsoleDumper {
public:
  CudaTensorConsoleDumper();
  //CudaTensorConsoleDumper(cudaStream_t stream);
  virtual ~CudaTensorConsoleDumper() {}
  void Disable() const override;
  bool IsEnabled() const  override;
  void Print(const char* name, const float* tensor, int dim0, int dim1) const  override;
  void Print(const char* name, const int64_t* tensor, int dim0, int dim1) const  override;
  void Print(const char* name, const float* tensor, int dim0, int dim1, int dim2) const  override;
  void Print(const char* name, const int64_t* tensor, int dim0, int dim1, int dim2) const  override;
  void Print(const char* name, const Tensor& value) const  override;
  void Print(const char* name, const OrtValue& value) const  override;
  void Print(const char* name, int index, bool end_line) const  override;
  void Print(const char* name, const std::string& value, bool end_line) const  override;

private:
  cudaStream_t stream_;
  static bool s_is_enabled_;
};


}  // namespace transformers
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime