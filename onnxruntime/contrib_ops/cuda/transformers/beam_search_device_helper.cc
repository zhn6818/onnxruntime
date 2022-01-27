#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/math/topk_impl.h"
#include "core/framework/ort_value.h"

namespace onnxruntime {
namespace concurrency {
class ThreadPool;
}
}  // namespace onnxruntime

#include "beam_search_device_helper.h"

namespace onnxruntime {
namespace contrib {
namespace BeamSearchCudaDeviceHelper {

Status TopK(const Tensor* input, const int axis, const unsigned k, bool largest, bool sorted,
            AllocatorPtr allocator,
            void* stream,
            onnxruntime::concurrency::ThreadPool* /*threadpool*/,
            std::unique_ptr<Tensor>& output_values,
            std::unique_ptr<Tensor>& output_indices) {
  ORT_ENFORCE(nullptr != input);
  int32_t rank = static_cast<int32_t>(input->Shape().NumDimensions());

  ORT_ENFORCE(axis >= 0 && axis < rank);
  ORT_ENFORCE(k > 0 && k <= input->Shape().GetDims()[axis]);

  auto output_shape = input->Shape();
  output_shape[axis] = k;

  auto elem_nums = input->Shape().GetDimsAsVector();
  int64_t dimension = elem_nums[axis];
  for (auto i = static_cast<int32_t>(elem_nums.size()) - 2; i >= 0; --i) {
    elem_nums[i] *= elem_nums[i + 1];
  }

  int64_t N = elem_nums[0] / dimension;
  TArray<int64_t> elem_nums_cuda(elem_nums);

  output_values = Tensor::Create(input->DataType(), output_shape, allocator);
  output_indices = Tensor::Create(DataTypeImpl::GetType<int64_t>(), output_shape, allocator);

  if (input->IsDataType<float>()) {
    return TopKImpl<float>(nullptr, // We limit number of beams in BeamSearchParameters, so that K <= 256 and kernel is not needed
                           reinterpret_cast<cudaStream_t>(stream),
                           input->Data<float>(),
                           static_cast<float*>(output_values->MutableDataRaw()),
                           static_cast<int64_t*>(output_indices->MutableDataRaw()),
                           elem_nums_cuda,
                           elem_nums.size(),
                           static_cast<int32_t>(axis),
                           static_cast<int64_t>(k),
                           static_cast<int64_t>(largest),
                           static_cast<int64_t>(sorted),
                           N,
                           dimension);
  } else if (input->IsDataType<MLFloat16>()) {
    return TopKImpl<MLFloat16>(nullptr,
                               reinterpret_cast<cudaStream_t>(stream),
                               input->Data<MLFloat16>(),
                               static_cast<MLFloat16*>(output_values->MutableDataRaw()),
                               static_cast<int64_t*>(output_indices->MutableDataRaw()),
                               elem_nums_cuda,
                               elem_nums.size(),
                               static_cast<int32_t>(axis),
                               static_cast<int64_t>(k),
                               static_cast<int64_t>(largest),
                               static_cast<int64_t>(sorted),
                               N,
                               dimension);
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                         "BeamSearch op: An implementation for the input type ",
                         input->DataType(), " is not supported yet");
}

OrtValue ExpandInputs(const OrtValue& input, int num_beams, AllocatorPtr allocator) {
  // Input shape (batch_size, sequence_length)
  // Output shape (batch_size * num_beams, sequence_length)
  if (num_beams == 1)
    return input;

  const TensorShape& input_shape = input.Get<Tensor>().Shape();
  const int64_t& batch_size = input_shape[0];
  const int64_t& sequence_length = input_shape[1];

  int64_t dims[] = {batch_size * num_beams, sequence_length};
  TensorShape expanded_shape(&dims[0], 2);

  OrtValue expanded;
  MLDataType element_type = input.Get<Tensor>().DataType();
  Tensor::InitOrtValue(element_type, expanded_shape, allocator, expanded);

  if (element_type == DataTypeImpl::GetType<int64_t>()) {
    const int64_t* input_data = input.Get<Tensor>().Data<int64_t>();
    int64_t* expanded_data = expanded.GetMutable<Tensor>()->MutableData<int64_t>();
    int64_t* target = expanded_data;
    for (int i = 0; i < batch_size; i++) {
      for (int j = 0; j < num_beams; j++) {
        memcpy(target, input_data + i * sequence_length, sizeof(int64_t) * sequence_length);
        target += sequence_length;
      }
    }
  } else if (element_type == DataTypeImpl::GetType<float>()) {
    const float* input_data = input.Get<Tensor>().Data<float>();
    float* expanded_data = expanded.GetMutable<Tensor>()->MutableData<float>();
    float* target = expanded_data;
    for (int i = 0; i < batch_size; i++) {
      for (int j = 0; j < num_beams; j++) {
        memcpy(target, input_data + i * sequence_length, sizeof(float) * sequence_length);
        target += sequence_length;
      }
    }
  }

  return expanded;
}

void CreateInputs(
    const Tensor* original_input_ids,
    int num_beams,
    int pad_token_id,
    gsl::span<int64_t>& next_positions,
    AllocatorPtr alloactor,
    std::vector<OrtValue>& feeds) {
  const TensorShape& input_ids_shape = original_input_ids->Shape();
  ORT_ENFORCE(input_ids_shape.NumDimensions() == 2);
  const int64_t& batch_size = input_ids_shape[0];
  const int64_t& sequence_length = input_ids_shape[1];

  // Allocate position_ids and attention_mask based on shape of input_ids
  auto element_type = DataTypeImpl::GetType<int64_t>();

  // input_ids for subgraph is int64, so we need Cast input_ids from int32 to int64.
  // Current shape is (batch_size, sequence_length)
  // Note that we will expand it to (batch_size * num_beams, sequence_length) later.
  OrtValue input_ids;
  Tensor::InitOrtValue(element_type, input_ids_shape, alloactor, input_ids);

  const int32_t* source = original_input_ids->Data<int32_t>();
  int64_t* target = input_ids.GetMutable<Tensor>()->MutableData<int64_t>();
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < sequence_length; j++, source++, target++) {
      *target = static_cast<int64_t>(*source);
    }
  }

  OrtValue position_ids;
  Tensor::InitOrtValue(element_type, input_ids_shape, alloactor, position_ids);

  OrtValue attention_mask;
  auto mask_type = DataTypeImpl::GetType<float>();
  Tensor::InitOrtValue(mask_type, input_ids_shape, alloactor, attention_mask);

  // Set attention mask to be 0 for pad tokens, and 1 for all other tokens.
  // Set position id to be 0 for pad tokens, and accumulated sum of mask in a batch for other tokens
  float* mask_data = attention_mask.GetMutable<Tensor>()->MutableData<float>();
  int64_t* position_data = position_ids.GetMutable<Tensor>()->MutableData<int64_t>();
  source = original_input_ids->Data<int32_t>();
  float* mask = mask_data;
  int64_t* position = position_data;
  for (int i = 0; i < batch_size; i++) {
    int64_t abs_position = 0;
    for (int j = 0; j < sequence_length; j++, source++, mask++, position++) {
      if (*source == pad_token_id) {
        *mask = 0.0f;
        *position = 0;
      } else {
        *mask = 1.0f;
        *position = abs_position;
        abs_position++;
      }
    }
    for (int k = 0; k < num_beams; k++) {
      next_positions[i * num_beams + k] = abs_position;
    }
  }

  // Expand (batch_size, sequence_length) to (batch_size * num_beams, sequence_length) for input_ids, position_ids and attention_mask
  // TODO: Try expand inputs/outputs after first subgraph call instead. That may get better performance, but more complex to implement.
  OrtValue expanded_input_ids = ExpandInputs(input_ids, num_beams, alloactor);
  OrtValue expanded_position_ids = ExpandInputs(position_ids, num_beams, alloactor);
  OrtValue expanded_attention_mask = ExpandInputs(attention_mask, num_beams, alloactor);

  feeds.push_back(expanded_input_ids);
  feeds.push_back(expanded_position_ids);
  feeds.push_back(expanded_attention_mask);
}

}  // namespace BeamSearchCpuDeviceHelper
}  // namespace contrib
}  // namespace onnxruntime