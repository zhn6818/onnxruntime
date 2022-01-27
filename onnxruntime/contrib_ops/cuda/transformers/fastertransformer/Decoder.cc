/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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

#include "Decoder.h"

namespace fastertransformer {

template<typename T>
void Decoder<T>::initialize()
{
}

template<typename T>
void Decoder<T>::allocateBuffer()
{
    if (is_allocate_buffer_ == false) {
        decoder_normed_input_ =
            reinterpret_cast<T*>(allocator_->malloc(sizeof(T) * max_batch_size_ * hidden_units_, false));
        self_attn_output_ =
            reinterpret_cast<T*>(allocator_->malloc(sizeof(T) * max_batch_size_ * hidden_units_, false));
        normed_self_attn_output_ =
            reinterpret_cast<T*>(allocator_->malloc(sizeof(T) * max_batch_size_ * hidden_units_, false));
        cross_attn_output_ =
            reinterpret_cast<T*>(allocator_->malloc(sizeof(T) * max_batch_size_ * hidden_units_, false));
        normed_cross_attn_output_ =
            reinterpret_cast<T*>(allocator_->malloc(sizeof(T) * max_batch_size_ * hidden_units_, false));
        decoder_layer_output_ =
            reinterpret_cast<T*>(allocator_->malloc(sizeof(T) * max_batch_size_ * hidden_units_, false));
        is_allocate_buffer_ = true;
    }
}

template<typename T>
void Decoder<T>::freeBuffer()
{
    if (is_allocate_buffer_ == true) {
        allocator_->free(decoder_normed_input_);
        allocator_->free(self_attn_output_);
        allocator_->free(normed_self_attn_output_);
        allocator_->free(cross_attn_output_);
        allocator_->free(normed_cross_attn_output_);
        allocator_->free(decoder_layer_output_);
        is_allocate_buffer_ = false;
    }
}

template<typename T>
bool Decoder<T>::isValidBatchSize(size_t batch_size)
{
    if (max_batch_size_ == 0) {
        max_batch_size_ = batch_size;
        return true;
    }
    else {
        return batch_size <= max_batch_size_;
    }
}

template<typename T>
Decoder<T>::Decoder(size_t max_batch_size,
                    size_t head_num,
                    size_t size_per_head,
                    size_t inter_size,
                    size_t num_layer,
                    cudaStream_t stream,
                    cublasMMWrapper* cublas_wrapper,
                    IAllocator* allocator,
                    bool is_free_buffer_after_forward):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    max_batch_size_(max_batch_size),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layer_(num_layer),
    hidden_units_(head_num_ * size_per_head)
{
    initialize();
}

template<typename T>
Decoder<T>::Decoder(Decoder<T> const& decoder):
    BaseLayer(decoder.stream_, decoder.cublas_wrapper_, decoder.allocator_, decoder.is_free_buffer_after_forward_),
    max_batch_size_(decoder.max_batch_size_),
    head_num_(decoder.head_num_),
    size_per_head_(decoder.size_per_head_),
    inter_size_(decoder.inter_size_),
    num_layer_(decoder.num_layer_),
    hidden_units_(decoder.hidden_units_)
{
    initialize();
}

template<typename T>
Decoder<T>::~Decoder()
{
    freeBuffer();
}

template<typename T>
void Decoder<T>::forward(std::vector<Tensor>* output_tensors,
                         const std::vector<Tensor>* input_tensors)
{
    // input tensors:
    //      decoder_input [batch_size, hidden_dimension],
    //      encoder_output [batch_size, mem_max_seq_len, memory_hidden_dimension],
    //      encoder_sequence_length [batch_size],
    //      finished [batch_size],
    //      step [1] on cpu
    //      sequence_lengths [batch_size]

    // output tensors:
    //      decoder_output [batch_size, hidden_dimension],
    //      key_cache [num_layer, batch, head_num, size_per_head // x, max_seq_len, x]
    //      value_cache [num_layer, batch, head_num, max_seq_len, size_per_head]
    //      key_mem_cache [num_layer, batch_size, mem_max_seq_len, hidden_dimension],
    //      value_mem_cache [num_layer, batch_size, mem_max_seq_len, hidden_dimension]

    FT_CHECK(input_tensors->size() == 6);
    FT_CHECK(output_tensors->size() == 5);
    isValidBatchSize(input_tensors->at(0).shape[0]);
    allocateBuffer();

    const size_t batch_size = (size_t)input_tensors->at(0).shape[0];
    const size_t mem_max_seq_len = (size_t)input_tensors->at(1).shape[1];
    const DataType data_type = getTensorType<T>();

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template class Decoder<float>;
template class Decoder<half>;

}  // namespace fastertransformer