#pragma once

#ifndef SHARED_PROVIDER
#include "core/common/common.h"
#include "core/framework/tensor.h"
#include "core/framework/allocator.h"
//#include "core/framework/threadpool.h"
#endif

#include "gsl/gsl"

namespace onnxruntime {
namespace concurrency {
class ThreadPool;
}
}  // namespace onnxruntime

namespace onnxruntime {
namespace contrib {

namespace BeamSearchDeviceHelper{
  using TopkFunc = std::function<Status(const Tensor* input, const int axis, const unsigned k, bool largest, bool sorted,
                                        AllocatorPtr allocator,
                                        onnxruntime::concurrency::ThreadPool* threadpool,
                                        std::unique_ptr<Tensor>& output_values,
                                        std::unique_ptr<Tensor>& output_indices)>;

  // Create subgraph inputs: input_ids, position_ids and attention_mask
  using CreateInputsFunc = std::function<void(const Tensor* original_input_ids,
                                              int num_beams,
                                              int pad_token_id,
                                              gsl::span<int64_t>& next_positions,
                                              AllocatorPtr alloactor,
                                              std::vector<OrtValue>& feeds
                                              )>;
}

// These are CPU specific device helper implementations
namespace BeamSearchCpuDeviceHelper {
Status TopK(const Tensor* input, const int axis, const unsigned k, bool largest, bool sorted,
            AllocatorPtr allocator,
            onnxruntime::concurrency::ThreadPool* threadpool,
            std::unique_ptr<Tensor>& output_values,
            std::unique_ptr<Tensor>& output_indices);

void CreateInputs(
    const Tensor* original_input_ids,
    int num_beams,
    int pad_token_id,
    gsl::span<int64_t>& next_positions,
    AllocatorPtr alloactor,
    std::vector<OrtValue>& feeds);

}  // namespace BeamSearchCpuDeviceHelper
}  // namespace contrib
}  // namespace onnxruntime