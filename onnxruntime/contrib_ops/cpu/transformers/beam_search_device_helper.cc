#include "beam_search_device_helper.h"
#include "core/providers/cpu/math/top_k.h"

namespace onnxruntime {
namespace contrib {
namespace BeamSearchCpuDeviceHelper {

Status TopK(const Tensor* input, const int axis, const unsigned k, bool largest, bool sorted,
            AllocatorPtr allocator,
            onnxruntime::concurrency::ThreadPool* threadpool,
            std::unique_ptr<Tensor>& output_values,
            std::unique_ptr<Tensor>& output_indices) {
  if (input->IsDataType<float>()) {
    return GetTopK<float>(input, axis, k, largest, sorted, allocator, threadpool, output_values, output_indices);
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