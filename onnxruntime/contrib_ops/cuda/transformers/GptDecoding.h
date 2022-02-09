/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstddef>
#include <vector>

#include "contrib_ops/cuda/transformers/fastertransformer/layers/DynamicDecodeBaseLayer.h"

namespace fastertransformer {

template<typename T>
class GptDecoding: public BaseLayer {
private:
    // buffer handling
    size_t max_batch_size_ = 0;
    size_t max_seq_len_ = 0;
    size_t max_input_len_ = 0;
    
    // meta data
    size_t beam_width_;
    size_t head_num_;
    size_t size_per_head_;
    size_t inter_size_;
    size_t num_layer_;
    size_t vocab_size_;

    int start_id_;
    int end_id_;
    float beam_search_diversity_rate_;
    size_t hidden_units_;
    size_t top_k_;
    float top_p_;
    unsigned long long random_seed_;
    float temperature_;
    float len_penalty_;
    float repetition_penalty_;
    size_t vocab_size_padded_;

    const bool is_context_qk_buf_float_ = true;

    DynamicDecodeBaseLayer* dynamic_decode_;

    void allocateBuffer() override;
    void freeBuffer() override;

    void checkBuffer(size_t batch_size, size_t input_seq_len, size_t output_seq_len);

    void initialize();

protected:
    T* logits_buf_;
    float* cum_log_probs_;
    bool* finished_buf_;
    bool* h_finished_buf_;

    int* output_ids_buf_;
    int* parent_ids_buf_;
    int* input_length_buf_;

public:
    GptDecoding(
        size_t max_batch_size,
        size_t max_seq_len,
        size_t max_input_len,
        size_t beam_width,
        size_t head_num,
        size_t size_per_head,
        size_t inter_size,
        size_t num_layer,
        size_t vocab_size,
        int start_id,
        int end_id,
        float beam_search_diversity_rate,
        size_t top_k,
        float top_p,
        unsigned long long random_seed,
        float temperature,
        float len_penalty,
        float repetition_penalty,
        cudaStream_t stream,
        cublasMMWrapper* cublas_wrapper,
        IAllocator* allocator,
        bool is_free_buffer_after_forward,
        cudaDeviceProp* cuda_device_prop,
        bool sparse = false);

    GptDecoding(GptDecoding<T> const& gpt);

    ~GptDecoding();

    void forward(std::vector<Tensor>* output_tensors,
                 const std::vector<Tensor>* input_tensors);
};

}  // namespace fastertransformer
