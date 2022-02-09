// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <iomanip>
#include <cuda_runtime_api.h>
#include <memory>
#include <cuda_fp16.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "dump_cuda_tensor.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace transformers {

#ifdef NDEBUG
bool g_enable_tensor_dump = false;
#else
bool g_enable_tensor_dump = true;
#endif

#define MAX_ROW_OR_COLUMN 8

#define SKIP_IF_MORE_THAN(row_or_column_size, i, max_n, new_line)                           \
  if (row_or_column_size > max_n && i >= max_n / 2 && i + max_n / 2 < row_or_column_size) { \
    if (i == max_n / 2) {                                                                   \
      printf(", ...");                                                                      \
      if (new_line)                                                                         \
        printf("\n");                                                                       \
    }                                                                                       \
    continue;                                                                               \
  }                                                                                         \
                                                                                            \
  if (i > 0 && !new_line)                                                                   \
    printf(", ");

#define SKIP_IF_TOO_MANY(row_or_column_size, i, new_line) SKIP_IF_MORE_THAN(row_or_column_size, i, MAX_ROW_OR_COLUMN, new_line)

template <typename T>
class PinnedHostBuffer {
 public:
  typedef std::shared_ptr<PinnedHostBuffer<T>> ptr;

  PinnedHostBuffer(size_t elementCount)
      : mBuffer(nullptr) {
    cudaHostAlloc(&mBuffer, elementCount * sizeof(T), cudaHostAllocDefault);
  }

  virtual ~PinnedHostBuffer() {
    if (mBuffer) {
      cudaFreeHost(mBuffer);
    }
  }

  operator T*() {
    return mBuffer;
  }

  operator const T*() const {
    return mBuffer;
  }

 protected:
  T* mBuffer;
};

template <typename T>
__host__ __device__  void PrintValue(T value) {
  if (std::is_same<T, half>::value) {
    printf("%.8f", __half2float(value));
  } else if (std::is_integral<T>::value) {
    printf("%d", (int)value);
  } else {
    printf("%.8f", (float)value);
  }
}

/*
template <typename T>
__global__ void Print2DTensor(const T* tensor, int dim0, int dim1, char title, char subtitle) {
  if (threadIdx.x == 0) {
    for (int i = 0; i < dim0; i++) {
      SKIP_IF_TOO_MANY(dim0, i, true);
      printf("%c%c[%d]:", title, subtitle, i);
      for (int j = 0; j < dim1; j++) {
        SKIP_IF_TOO_MANY(dim1, j, false);
        T value = tensor[i * dim1 + j];
        PrintValue<T>(value);
      }
      printf("\n");
    }
  }
}

template <typename T>
__global__ void Print3DTensor(const T* tensor, int dim0, int dim1, int dim2, char title, char subtitle) {
  if (threadIdx.x == 0) {
    for (int i = 0; i < dim0; i++) {
      SKIP_IF_TOO_MANY(dim0, i, true);
      for (int j = 0; j < dim1; j++) {
        printf("%c%c[%d][%d]:", title, subtitle, i, j);
        for (int k = 0; k < dim2; k++) {
          SKIP_IF_TOO_MANY(dim1, j, false);
          T value = tensor[i * dim1 * dim2 + j * dim2 + k];
          PrintValue<T>(value);
        }
        printf("\n");
      }
      printf("\n");
    }
  }
}

template <typename T>
__global__ void Print4DTensor(const T* tensor, int dim0, int dim1, int dim2, int dim3, char title, char subtitle) {
  if (threadIdx.x == 0) {
    for (int i = 0; i < dim0; i++) {
      SKIP_IF_TOO_MANY(dim0, i, true);
      for (int j = 0; j < dim1; j++) {
        SKIP_IF_TOO_MANY(dim1, j, true);
        for (int k = 0; k < dim2; k++) {
          SKIP_IF_TOO_MANY(dim2, k, true);
          printf("%c%c[%d][%d][%d]:", title, subtitle, i, j, k);
          for (int x = 0; x < dim3; x++) {
            SKIP_IF_TOO_MANY(dim3, x, false);
            T value = tensor[i * dim1 * dim2 * dim3 + j * dim2 * dim3 + k * dim3 + x];
            PrintValue<T>(value);
          }
          printf("\n");
        }
        printf("\n");
      }
      printf("\n");
    }
  }
}

template <typename T>
__global__ void Print4DTensorSub2(const T* tensor, int dim0, int dim1, int dim2, int dim3, char title, char subtitle) {
  if (threadIdx.x == 0) {
    int i = dim0;
    int j = dim1;
    for (int k = 0; k < dim2; k++) {
      SKIP_IF_TOO_MANY(dim2, k, true);
      printf("%c%c[%d][%d][%d]:", title, subtitle, i, j, k);
      for (int x = 0; x < dim3; x++) {
        SKIP_IF_TOO_MANY(dim3, x, false);
        T value = tensor[i * dim1 * dim2 * dim3 + j * dim2 * dim3 + k * dim3 + x];
        PrintValue<T>(value);
      }
      printf("\n");
    }
  }
}

template <typename T>
__global__ void Print4DTensorSub3(const T* tensor, int dim0, int dim1, int dim2, int dim3, char title, char subtitle) {
  if (threadIdx.x == 0) {
    int i = dim0;
    for (int j = 0; j < dim1; j++) {
      SKIP_IF_TOO_MANY(dim1, j, true);
      for (int k = 0; k < dim2; k++) {
        SKIP_IF_TOO_MANY(dim2, k, true);
        printf("%c%c[%d][%d][%d]:", title, subtitle, i, j, k);
        for (int x = 0; x < dim3; x++) {
          SKIP_IF_TOO_MANY(dim3, x, false);
          T value = tensor[i * dim1 * dim2 * dim3 + j * dim2 * dim3 + k * dim3 + x];
          PrintValue<T>(value);
        }
        printf("\n");
      }
      printf("\n");
    }
    printf("\n");
  }
}

template <typename T>
void DumpTensor4D(cudaStream_t stream, const T* tensor, int dim0, int dim1, int dim2, int dim3, const std::string& title) {
  int elements = dim0 * dim1 * dim2 * dim3;

  auto data = std::make_shared<PinnedHostBuffer<T>>(elements);
  cudaMemcpyAsync(*data, tensor, elements * sizeof(T), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  const T* pinned_data = *data;
  for (int i = 0; i < dim0; i++) {
    SKIP_IF_TOO_MANY(dim0, i, true);
    for (int j = 0; j < dim1; j++) {
      SKIP_IF_TOO_MANY(dim1, j, true);
      for (int k = 0; k < dim2; k++) {
        SKIP_IF_TOO_MANY(dim2, k, true);
        printf("%s[%d][%d][%d]:", title.c_str(), i, j, k);
        for (int x = 0; x < dim3; x++) {
          SKIP_IF_TOO_MANY(dim3, x, false);
          T value = pinned_data[i * dim1 * dim2 * dim3 + j * dim2 * dim3 + k * dim3 + x];
          PrintValue<T>(value);
        }
        printf("\n");
      }
      printf("\n");
    }
    printf("\n");
  }
}

template <typename T>
void DumpTensor2D(cudaStream_t stream, const T* tensor, int dim0, int dim1, const std::string& title) {
  int elements = dim0 * dim1;

  auto data = std::make_shared<PinnedHostBuffer<T>>(elements);
  cudaMemcpyAsync(*data, tensor, elements * sizeof(T), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  const T* pinned_data = *data;

  for (int i = 0; i < dim0; i++) {
    SKIP_IF_TOO_MANY(dim0, i, true);
    printf("%s[%d]:", title.c_str(), i);
    for (int j = 0; j < dim1; j++) {
      SKIP_IF_TOO_MANY(dim1, j, true);
      T value = pinned_data[i * dim1 + j];
      PrintValue<T>(value);
    }
    printf("\n");
  }
}


template <typename T>
void Dump2DTensor(cudaStream_t stream, const T* tensor, int dim0, int dim1, char title, char subtitle) {
  cudaDeviceSynchronize();
  Print2DTensor<T><<<1, 1, 0, stream>>>(tensor, dim0, dim1, title, subtitle);
  cudaDeviceSynchronize();
}

// template instantiation
template void Dump2DTensor<float>(cudaStream_t, const float*, int, int, char, char);
template void Dump2DTensor<half>(cudaStream_t, const half*, int, int, char, char);
template void Dump2DTensor<int8_t>(cudaStream_t, const int8_t*, int, int, char, char);
template void Dump2DTensor<int32_t>(cudaStream_t, const int32_t*, int, int, char, char);
template void Dump2DTensor<int64_t>(cudaStream_t, const int64_t*, int, int, char, char);

template <typename T>
void Dump3DTensor(cudaStream_t stream, const T* tensor, int dim0, int dim1, int dim2, char title, char subtitle) {
  cudaDeviceSynchronize();
  Print3DTensor<T><<<1, 1, 0, stream>>>(tensor, dim0, dim1, dim2, title, subtitle);
  cudaDeviceSynchronize();
}

// template instantiation
template void Dump3DTensor<float>(cudaStream_t, const float*, int, int, int, char, char);
template void Dump3DTensor<half>(cudaStream_t, const half*, int, int, int, char, char);
template void Dump3DTensor<int8_t>(cudaStream_t, const int8_t*, int, int, int, char, char);
template void Dump3DTensor<int32_t>(cudaStream_t, const int32_t*, int, int, int, char, char);
template void Dump3DTensor<int64_t>(cudaStream_t, const int64_t*, int, int, int, char, char);

template <typename T>
void Dump4DTensor(cudaStream_t stream, const T* tensor, int dim0, int dim1, int dim2, int dim3, char title, char subtitle) {
  cudaDeviceSynchronize();
  Print4DTensor<T><<<1, 1, 0, stream>>>(tensor, dim0, dim1, dim2, dim3, title, subtitle);
  cudaDeviceSynchronize();
}

// template instantiation
template void Dump4DTensor<float>(cudaStream_t, const float*, int, int, int, int, char, char);
template void Dump4DTensor<half>(cudaStream_t, const half*, int, int, int, int, char, char);

void Dump4DTensor2(bool is_float16, cudaStream_t stream, const void* tensor, int dim0, int dim1, int dim2, int dim3, char title, char subtitle) {
  if (is_float16) {
    Print4DTensorSub2<<<1, 1, 0, stream>>>(reinterpret_cast<const float*>(tensor), dim0, dim1, dim2, dim3, title, subtitle);
  } else {
    Print4DTensorSub2<<<1, 1, 0, stream>>>(reinterpret_cast<const half*>(tensor), dim0, dim1, dim2, dim3, title, subtitle);
  }
}

void Dump4DTensor3(bool is_float16, cudaStream_t stream, const void* tensor, int dim0, int dim1, int dim2, int dim3, char title, char subtitle) {
  if (is_float16) {
    Print4DTensorSub3<<<1, 1, 0, stream>>>(reinterpret_cast<const float*>(tensor), dim0, dim1, dim2, dim3, title, subtitle);
  } else {
    Print4DTensorSub3<<<1, 1, 0, stream>>>(reinterpret_cast<const half*>(tensor), dim0, dim1, dim2, dim3, title, subtitle);
  }
}

void Dump4DTensor4(bool is_float16, cudaStream_t stream, const void* tensor, int dim0, int dim1, int dim2, int dim3, char title, char subtitle) {
  if (is_float16) {
    Dump4DTensor(stream, reinterpret_cast<const float*>(tensor), dim0, dim1, dim2, dim3, title, subtitle);
  } else {
    Dump4DTensor(stream, reinterpret_cast<const half*>(tensor), dim0, dim1, dim2, dim3, title, subtitle);
  }
}
*/

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




template <typename T>
void DumpTensor(const char* name, const T* tensor, int dim0, int dim1, int dim2) {
  if (!g_enable_tensor_dump)
    return;

  int num_items = dim0 * dim1 * dim2;
  auto data = std::make_shared<PinnedHostBuffer<T>>(num_items);
  cudaDeviceSynchronize();
  cudaMemcpy(*data, tensor, num_items * sizeof(T), cudaMemcpyDeviceToHost);

  if (nullptr != name) {
    std::cout << std::string(name) << std::endl;
  }

  const T* pinned_data = *data;
  for (int i = 0; i < dim0; i++) {
    SKIP_IF_TOO_MANY(dim0, i, true);
    for (int j = 0; j < dim1; j++) {
      SKIP_IF_TOO_MANY(dim1, j, true);
      std::cout << "[" << i << "][" << j << "]:";
      for (int k = 0; k < dim2; k++) {
        SKIP_IF_TOO_MANY(dim2, k, false);
        T value = pinned_data[i * dim1 * dim2 + j * dim2 + k];
        PrintValue<T>(value);
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

template <typename T>
void DumpTensor(const char* name, const T* tensor, int dim0, int dim1) {
  if (!g_enable_tensor_dump)
    return;

  int num_items = dim0 * dim1;
  auto data = std::make_shared<PinnedHostBuffer<T>>(num_items);
  cudaDeviceSynchronize();
  cudaMemcpy(*data, tensor, num_items * sizeof(T), cudaMemcpyDeviceToHost);

  if (nullptr != name) {
    std::cout << std::string(name) << std::endl;
  }

  const T* pinned_data = *data;
  for (int i = 0; i < dim0; i++) {
    SKIP_IF_TOO_MANY(dim0, i, true);
    std::cout << "[" << i << "]:";
    for (int j = 0; j < dim1; j++) {
      SKIP_IF_TOO_MANY(dim1, j, false);
      T value = pinned_data[i * dim1 + j];
      PrintValue<T>(value);
    }
    std::cout << std::endl;
  }
}


void DumpOrtValue(const char* name, const OrtValue& value) {
  const Tensor& tensor = value.Get<Tensor>();
  DumpTensor(name, tensor);
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

void DumpString(const char* name, std::string value, bool end_line) {
  if (!g_enable_tensor_dump)
    return;

  std::cout << std::string(name) << "=" << value;

  if (end_line) {
    std::cout << std::endl;
  }
}


CudaTensorConsoleDumper::CudaTensorConsoleDumper():stream_(cudaStreamLegacy){
}

//CudaTensorConsoleDumper::CudaTensorConsoleDumper(cudaStream_t stream):stream_(stream){
//}

void CudaTensorConsoleDumper::Disable() const{
  DisableTensorDump();
}

bool CudaTensorConsoleDumper::IsEnabled() const {
  return g_enable_tensor_dump;
}

#ifdef DEBUG_BEAM_SEARCH
void CudaTensorConsoleDumper::Print(const char* name, const float* tensor, int dim0, int dim1) const {
  return onnxruntime::contrib::cuda::transformers::DumpTensor<float>(name, tensor, dim0, dim1);
}

void CudaTensorConsoleDumper::Print(const char* name, const int64_t* tensor, int dim0, int dim1) const {
  return onnxruntime::contrib::cuda::transformers::DumpTensor<int64_t>(name, tensor, dim0, dim1);
}

void CudaTensorConsoleDumper::Print(const char* name, const float* tensor, int dim0, int dim1, int dim2) const {
  return onnxruntime::contrib::cuda::transformers::DumpTensor<float>(name, tensor, dim0, dim1, dim2);
}

void CudaTensorConsoleDumper::Print(const char* name, const int64_t* tensor, int dim0, int dim1, int dim2) const {
  return onnxruntime::contrib::cuda::transformers::DumpTensor<int64_t>(name, tensor, dim0, dim1, dim2);
}

void CudaTensorConsoleDumper::Print(const char* name, const Tensor& tensor) const {
  onnxruntime::contrib::cuda::transformers::DumpTensor(name, tensor);
}

void CudaTensorConsoleDumper::Print(const char* name, const OrtValue& value) const {
  onnxruntime::contrib::cuda::transformers::DumpOrtValue(name, value);
}

void CudaTensorConsoleDumper::Print(const char* name, int index, bool end_line) const {
  onnxruntime::contrib::cuda::transformers::DumpString(name, index, end_line);
}

void CudaTensorConsoleDumper::Print(const char* name, const std::string& value, bool end_line) const {
  onnxruntime::contrib::cuda::transformers::DumpString(name, value, end_line);
}

#else
void CudaTensorConsoleDumper::Print(const char*, const float*, int, int) const {
}

void CudaTensorConsoleDumper::Print(const char*, const int64_t*, int, int) const {
}

void CudaTensorConsoleDumper::Print(const char*, const float*, int, int, int) const {
}

void CudaTensorConsoleDumper::Print(const char*, const int64_t*, int, int, int) const {
}

void CudaTensorConsoleDumper::Print(const char*, const Tensor&) const {
}

void CudaTensorConsoleDumper::Print(const char*, const OrtValue&) const {
}

void CudaTensorConsoleDumper::Print(const char*, int, bool) const {
}

void CudaTensorConsoleDumper::Print(const char*, const std::string&, bool) const {
}
#endif


}
}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime