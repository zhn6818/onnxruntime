// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "dump_tensor.h"
#include "core/platform/env.h"
#include "core/platform/env_var_utils.h"

namespace onnxruntime {
namespace contrib {
namespace transformers {

namespace dump_tensor_env_vars {
constexpr const char* kDumpBeamSearch = "ORT_DUMP_BEAM_SEARCH";
}

#ifdef NDEBUG
bool g_enable_tensor_dump = false;
#else
bool g_enable_tensor_dump = true;
#endif

void ConfigureTensorDump() {
  const auto parsed = ParseEnvironmentVariable<bool>(dump_tensor_env_vars::kDumpBeamSearch);
  if (parsed.has_value()) {
    g_enable_tensor_dump = *parsed;
  }
}

void DisableTensorDump() {
  g_enable_tensor_dump = false;
}

void DumpString(const char* name, int index, bool end_line) {
  if (!g_enable_tensor_dump)
    return;
  std::cout << std::string(name) << "[" << index << "]";

  if (end_line) {
    std::cout << std::endl;
  }
}

void DumpString(const char* name, const std::string& value, bool end_line) {
  if (!g_enable_tensor_dump)
    return;

  std::cout << std::string(name) << "=" << value;

  if (end_line) {
    std::cout << std::endl;
  }
}

void DumpTensor(const char* name, const Tensor& tensor, int dim0, int dim1, int dim2) {
  MLDataType dataType = tensor.DataType();
  if (dataType == DataTypeImpl::GetType<float>()) {
    DumpTensor<float>(name, tensor.Data<float>(), dim0, dim1, dim2);
  } else if (dataType == DataTypeImpl::GetType<int32_t>()) {
    DumpTensor<int32_t>(name, tensor.Data<int32_t>(), dim0, dim1, dim2);
  } else if (dataType == DataTypeImpl::GetType<int64_t>()) {
    DumpTensor<int64_t>(name, tensor.Data<int64_t>(), dim0, dim1, dim2);
  } else {
    assert(0);
  }
}

void DumpTensor(const char* name, const Tensor& tensor, int dim0, int dim1) {
  MLDataType dataType = tensor.DataType();
  if (dataType == DataTypeImpl::GetType<float>()) {
    DumpTensor<float>(name, tensor.Data<float>(), dim0, dim1);
  } else if (dataType == DataTypeImpl::GetType<int32_t>()) {
    DumpTensor<int32_t>(name, tensor.Data<int32_t>(), dim0, dim1);
  } else if (dataType == DataTypeImpl::GetType<int64_t>()) {
    DumpTensor<int64_t>(name, tensor.Data<int64_t>(), dim0, dim1);
  } else {
    assert(0);
  }
}

void DumpTensor(const char* name, const Tensor& tensor) {
  const auto& shape = tensor.Shape();
  auto num_items = shape.Size();
  size_t num_dims = shape.NumDimensions();

  if (num_dims == 3) {
    DumpTensor(name, tensor, static_cast<int>(shape[0]), static_cast<int>(shape[1]), static_cast<int>(shape[2]));
    return;
  }

  size_t num_rows = 1;
  if (num_dims > 1) {
    num_rows = static_cast<size_t>(shape[0]);
  }
  size_t row_size = num_items / num_rows;
  DumpTensor(name, tensor, static_cast<int>(num_rows), static_cast<int>(row_size));
}

void DumpOrtValue(const char* name, const OrtValue& value) {
  const Tensor& tensor = value.Get<Tensor>();
  DumpTensor(name, tensor);
}

void CpuTensorConsoleDumper::Disable() const {
  DisableTensorDump();
}

bool CpuTensorConsoleDumper::IsEnabled() const {
  return g_enable_tensor_dump;
}

#ifdef DEBUG_BEAM_SEARCH
void CpuTensorConsoleDumper::Print(const char* name, const float* tensor, int dim0, int dim1) const {
  onnxruntime::contrib::transformers::DumpTensor<float>(name, tensor, dim0, dim1);
}

void CpuTensorConsoleDumper::Print(const char* name, const int64_t* tensor, int dim0, int dim1) const {
  onnxruntime::contrib::transformers::DumpTensor<int64_t>(name, tensor, dim0, dim1);
}

void CpuTensorConsoleDumper::Print(const char* name, const float* tensor, int dim0, int dim1, int dim2) const {
  onnxruntime::contrib::transformers::DumpTensor<float>(name, tensor, dim0, dim1, dim2);
}

void CpuTensorConsoleDumper::Print(const char* name, const int64_t* tensor, int dim0, int dim1, int dim2) const {
  onnxruntime::contrib::transformers::DumpTensor<int64_t>(name, tensor, dim0, dim1, dim2);
}

void CpuTensorConsoleDumper::Print(const char* name, const Tensor& tensor) const {
  onnxruntime::contrib::transformers::DumpTensor(name, tensor);
}

void CpuTensorConsoleDumper::Print(const char* name, const OrtValue& value) const {
  const Tensor& tensor = value.Get<Tensor>();
  onnxruntime::contrib::transformers::DumpTensor(name, tensor);
}

void CpuTensorConsoleDumper::Print(const char* name, int index, bool end_line) const {
  onnxruntime::contrib::transformers::DumpString(name, index, end_line);
}

void CpuTensorConsoleDumper::Print(const char* name, const std::string& value, bool end_line) const {
  onnxruntime::contrib::transformers::DumpString(name, value, end_line);
}

#else
void CpuTensorConsoleDumper::Print(const char*, const float*, int, int) const {
}

void CpuTensorConsoleDumper::Print(const char*, const int64_t*, int, int) const {
}

void CpuTensorConsoleDumper::Print(const char*, const float*, int, int, int) const {
}

void CpuTensorConsoleDumper::Print(const char*, const int64_t*, int, int, int) const {
}

void CpuTensorConsoleDumper::Print(const char*, const Tensor&) const {
}

void CpuTensorConsoleDumper::Print(const char*, const OrtValue&) const {
}

void CpuTensorConsoleDumper::Print(const char*, int, bool) const {
}

void CpuTensorConsoleDumper::Print(const char*, const std::string&, bool) const {
}

#endif

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime