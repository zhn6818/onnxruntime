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

#include "GptDecoding.h"
#include "contrib_ops/cuda/transformers/fastertransformer/kernels/gpt_kernels.h"
#include "contrib_ops/cuda/transformers/fastertransformer/kernels/decoding_kernels.h"
#include "contrib_ops/cuda/transformers/fastertransformer/layers/beam_search_layers/BeamSearchLayer.h"
#include "contrib_ops/cuda/transformers/fastertransformer/layers/beam_search_layers/OnlineBeamSearchLayer.h"
#include "contrib_ops/cuda/transformers/fastertransformer/utils/memory_utils.h"
// #include "contrib_ops/cuda/transformers/fastertransformer/layers/sampling_layers/TopKSamplingLayer.h"
// #include "contrib_ops/cuda/transformers/fastertransformer/layers/sampling_layers/TopKTopPSamplingLayer.h"
// #include "contrib_ops/cuda/transformers/fastertransformer/layers/sampling_layers/TopPSamplingLayer.h"

namespace fastertransformer {

template<typename T>
void GptDecoding<T>::initialize()
{
    if (beam_width_ > 1) {
        if (beam_width_ < 16) {
            dynamic_decode_ = new OnlineBeamSearchLayer<T>(max_batch_size_,
                                                           head_num_,
                                                           size_per_head_,
                                                           beam_width_,
                                                           vocab_size_,
                                                           vocab_size_padded_,
                                                           end_id_,
                                                           beam_search_diversity_rate_,
                                                           temperature_,
                                                           len_penalty_,
                                                           repetition_penalty_,
                                                           stream_,
                                                           cublas_wrapper_,
                                                           allocator_,
                                                           is_free_buffer_after_forward_);
        }
        else {
            dynamic_decode_ = new BeamSearchLayer<T>(max_batch_size_,
                                                     head_num_,
                                                     size_per_head_,
                                                     beam_width_,
                                                     vocab_size_,
                                                     vocab_size_padded_,
                                                     end_id_,
                                                     beam_search_diversity_rate_,
                                                     temperature_,
                                                     len_penalty_,
                                                     repetition_penalty_,
                                                     stream_,
                                                     cublas_wrapper_,
                                                     allocator_,
                                                     is_free_buffer_after_forward_);
        }
    }
    // else if (top_p_ == 0 && top_k_ != 0) {
    //     // we sugguest set the is_free_buffer_after_forward_ of sampling to false
    //     // since we need to initialize some buffers if we allocate buffer
    //     // every time.
    //     dynamic_decode_ = new TopKSamplingLayer<T>(max_batch_size_,
    //                                                vocab_size_,
    //                                                vocab_size_padded_,
    //                                                end_id_,
    //                                                top_k_,
    //                                                random_seed_,
    //                                                temperature_,
    //                                                len_penalty_,
    //                                                repetition_penalty_,
    //                                                stream_,
    //                                                cublas_wrapper_,
    //                                                allocator_,
    //                                                false);
    // }
    // else if (top_k_ == 0 && top_p_ != 0.0f) {
    //     // we sugguest set the is_free_buffer_after_forward_ of sampling to false
    //     // since we need to initialize some buffers if we allocate buffer
    //     // every time.
    //     dynamic_decode_ = new TopPSamplingLayer<T>(max_batch_size_,
    //                                                vocab_size_,
    //                                                vocab_size_padded_,
    //                                                end_id_,
    //                                                top_p_,
    //                                                random_seed_,
    //                                                temperature_,
    //                                                len_penalty_,
    //                                                repetition_penalty_,
    //                                                stream_,
    //                                                cublas_wrapper_,
    //                                                allocator_,
    //                                                false,
    //                                                cuda_device_prop_);
    // }
    // else {
    //     // we sugguest set the is_free_buffer_after_forward_ of sampling to false
    //     // since we need to initialize some buffers if we allocate buffer
    //     // every time.
    //     dynamic_decode_ = new TopKTopPSamplingLayer<T>(max_batch_size_,
    //                                                    vocab_size_,
    //                                                    vocab_size_padded_,
    //                                                    end_id_,
    //                                                    top_k_,
    //                                                    top_p_,
    //                                                    random_seed_,
    //                                                    temperature_,
    //                                                    len_penalty_,
    //                                                    repetition_penalty_,
    //                                                    stream_,
    //                                                    cublas_wrapper_,
    //                                                    allocator_,
    //                                                    false);
    // }
}

template<typename T>
void GptDecoding<T>::allocateBuffer()
{
    if (is_allocate_buffer_ == false) {
        const size_t batch_x_beam = max_batch_size_ * beam_width_;
        logits_buf_ = (T*)(allocator_->malloc(sizeof(T) * batch_x_beam * vocab_size_padded_, false));
        cum_log_probs_ = (float*)(allocator_->malloc(sizeof(float) * batch_x_beam, false));
        finished_buf_ = (bool*)(allocator_->malloc(sizeof(bool) * batch_x_beam, false));
        h_finished_buf_ = new bool[batch_x_beam];
        output_ids_buf_ = (int*)(allocator_->malloc(sizeof(int) * batch_x_beam * max_seq_len_, true));
        parent_ids_buf_ = (int*)(allocator_->malloc(sizeof(int) * batch_x_beam * max_seq_len_, true));
        input_length_buf_ = (int*)(allocator_->malloc(sizeof(int) * batch_x_beam, false));
        is_allocate_buffer_ = true;
    }
}

template<typename T>
void GptDecoding<T>::freeBuffer()
{
    if (is_allocate_buffer_ == true) {
        allocator_->free(logits_buf_);
        allocator_->free(cum_log_probs_);
        allocator_->free(finished_buf_);
        delete[] h_finished_buf_;
        allocator_->free(output_ids_buf_);
        allocator_->free(parent_ids_buf_);
        is_allocate_buffer_ = false;
    }
}

template<typename T>
void GptDecoding<T>::checkBuffer(size_t batch_size, size_t input_seq_len, size_t output_seq_len)
{
    bool is_enough = true;
    if (batch_size > max_batch_size_) {
        max_batch_size_ = static_cast<size_t>(batch_size * 1.2f);
        is_enough = false;
    }

    if (output_seq_len + 1 > max_seq_len_) {
        max_seq_len_ = static_cast<size_t>((output_seq_len + 1) * 1.2);
        is_enough = false;
    }

    if (input_seq_len > max_input_len_) {
        max_input_len_ = static_cast<size_t>(input_seq_len * 1.2);
        is_enough = false;
    }

    if (!is_enough){
        freeBuffer();
    }
}


template <typename T>
GptDecoding<T>::GptDecoding(
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
    bool sparse) : BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, cuda_device_prop, sparse),
                   max_batch_size_(max_batch_size),
                   max_seq_len_(max_seq_len + 1),
                   max_input_len_(max_input_len),
                   beam_width_(beam_width),
                   head_num_(head_num),
                   size_per_head_(size_per_head),
                   inter_size_(inter_size),
                   num_layer_(num_layer),
                   vocab_size_(vocab_size),
                   start_id_(start_id),
                   end_id_(end_id),
                   beam_search_diversity_rate_(beam_search_diversity_rate),
                   hidden_units_(head_num_ * size_per_head),
                   top_k_(top_k),
                   top_p_(top_p),
                   random_seed_(random_seed),
                   temperature_(temperature),
                   len_penalty_(len_penalty),
                   repetition_penalty_(repetition_penalty) {
  vocab_size_padded_ = vocab_size_;
  if (std::is_same<half, T>::value) {
    vocab_size_padded_ = ((size_t)ceil(vocab_size_padded_ / 8.) * 8);
  }

  initialize();
}

template<typename T>
GptDecoding<T>::GptDecoding(GptDecoding<T> const& gpt):
    BaseLayer(gpt),
    max_batch_size_(gpt.max_batch_size_),
    max_seq_len_(gpt.max_seq_len_),
    max_input_len_(gpt.max_input_len_),
    beam_width_(gpt.beam_width_),
    head_num_(gpt.head_num_),
    size_per_head_(gpt.size_per_head_),
    inter_size_(gpt.inter_size_),
    num_layer_(gpt.num_layer_),
    vocab_size_(gpt.vocab_size_),
    start_id_(gpt.start_id_),
    end_id_(gpt.end_id_),
    beam_search_diversity_rate_(gpt.beam_search_diversity_rate_),
    hidden_units_(gpt.hidden_units_),
    top_k_(gpt.top_k_),
    top_p_(gpt.top_p_),
    random_seed_(gpt.random_seed_),
    temperature_(gpt.temperature_),
    len_penalty_(gpt.len_penalty_),
    repetition_penalty_(gpt.repetition_penalty_),
    vocab_size_padded_(gpt.vocab_size_padded_)
{
    initialize();
}

template<typename T>
GptDecoding<T>::~GptDecoding()
{
    delete dynamic_decode_;
    freeBuffer();
}

template<typename T>
void GptDecoding<T>::forward(std::vector<Tensor>* output_tensors,
                             const std::vector<Tensor>* input_tensors)
{
    // input_tensors:
    //      input_ids [batch_size * beam, max_input_length]
    //      input_lengths [batch_size * beam]
    //      max_output_seq_len [1] on cpu

    // output_tensors:
    //      output_ids [batch_size, beam, max_output_seq_len]
    //      parent_ids [max_output_seq_len, batch_size, beam]
    //      sequence_length [batch_size * beam]
    //      output_cum_log_probs [request_output_seq_len, batch_size, beam], must be float*.
    //          It leads to additional computing cost. If we don't need this result, please put nullptr

    // Step is from max_input_length ~ max_output_seq_len,
    // When step = k,  we put output ids and caches at step k, and the sequence_length would be k - 1 before
    // complete this step.
    // When there is no input_ids, put the start token at step 0 of output_ids_buf_. After forward, only copy
    // the step 1 ~ max_output_seq_len of output_ids_buf_ to output_tensors->at(0).data

    FT_CHECK(input_tensors->size() == 3);
    FT_CHECK(output_tensors->size() == 4);

    size_t max_input_length = input_tensors->at(0).shape[1];
    const int* input_length_ptr = (const int*)(input_tensors->at(1).data);
    const size_t max_output_seq_len = (size_t)(*(int*)input_tensors->at(2).data);

    const size_t batch_size = output_tensors->at(0).shape[0];
    int* sequence_lengths = (int*)(output_tensors->at(2).data);
    const DataType data_type = getTensorType<T>();

    float* output_cum_log_probs = (float*)(output_tensors->at(3).data);

    checkBuffer(batch_size, static_cast<size_t>(max_input_length), max_output_seq_len);

    // initialize the output ids and parent ids
    cudaMemsetAsync(output_ids_buf_, 0, sizeof(int) * batch_size * beam_width_ * max_seq_len_, stream_);
    cudaMemsetAsync(parent_ids_buf_, 0, sizeof(int) * batch_size * beam_width_ * max_seq_len_, stream_);

    
    invokeDecodingInitialize(finished_buf_,
                             sequence_lengths,
                             nullptr,
                             cum_log_probs_,
                             start_id_,
                             static_cast<int>(batch_size),
                             static_cast<int>(beam_width_),
                             static_cast<int>(max_input_length - 1),
                             stream_);
    sync_check_cuda_error();

    cudaMemcpyAsync(output_ids_buf_,
                    (int*)input_tensors->at(0).data,
                    sizeof(int) * batch_size * beam_width_,
                    cudaMemcpyDeviceToDevice,
                    stream_);

    const std::vector<size_t> self_k_cache_size = {num_layer_,
                                                   batch_size * beam_width_,
                                                   head_num_,
                                                   size_per_head_ / (16 / sizeof(T)),
                                                   max_output_seq_len,
                                                   16 / sizeof(T)};
    const std::vector<size_t> self_v_cache_size = {
        num_layer_, batch_size * beam_width_, head_num_, max_output_seq_len, size_per_head_};

    for (size_t step = max_input_length; step < max_output_seq_len; step++) {
        cudaMemcpy(h_finished_buf_,
                   finished_buf_,
                   batch_size * beam_width_,
                   cudaMemcpyDeviceToHost);
        //cudaD2Hcpy(h_finished_buf_, finished_buf_, batch_size * beam_width_);

        size_t sum = 0;
        for (size_t i = 0; i < batch_size * beam_width_; i++) {
            sum += static_cast<size_t>(h_finished_buf_[i]);
        }
        if (sum == batch_size * beam_width_) {
            break;
        }

        //TODO: call subgraph
        std::vector<Tensor>* dynamic_decode_input_tensors;
        std::vector<Tensor>* dynamic_decode_output_tensors;

        const int tmp_ite = 0;
        // if (beam_width_ > 1)
        {
            Tensor empty_tensor(MEMORY_GPU, data_type, {}, nullptr);

            dynamic_decode_input_tensors = new std::vector<Tensor>{
                Tensor{MEMORY_GPU, data_type, {batch_size, beam_width_, vocab_size_padded_}, logits_buf_},
                empty_tensor, // Tensor{MEMORY_GPU, data_type, {vocab_size_padded_}, nullptr},
                Tensor{MEMORY_CPU, TYPE_INT32, {1}, &step},
                empty_tensor, // Tensor{MEMORY_GPU, data_type, self_k_cache_size, key_caches_[src_cache_id]},
                empty_tensor, // Tensor{MEMORY_GPU, data_type, self_v_cache_size, value_caches_[src_cache_id]},
                empty_tensor, // Tensor{MEMORY_CPU, TYPE_INT32, {1}, &max_input_length},
                empty_tensor, // Tensor{MEMORY_GPU, TYPE_INT32, {batch_size, beam_width_}, input_length_ptr},
                Tensor{MEMORY_CPU, TYPE_INT32, {1}, &tmp_ite}};

            dynamic_decode_output_tensors = new std::vector<Tensor>{
                Tensor{MEMORY_GPU, TYPE_INT32, {max_output_seq_len, batch_size, beam_width_}, output_ids_buf_},
                Tensor{MEMORY_GPU, TYPE_BOOL, {batch_size * beam_width_}, finished_buf_},
                Tensor{MEMORY_GPU, TYPE_FP32, {batch_size * beam_width_}, cum_log_probs_},
                Tensor{MEMORY_GPU, TYPE_INT32, {max_output_seq_len, batch_size, beam_width_}, parent_ids_buf_},
                output_tensors->at(2),
                empty_tensor, // Tensor{MEMORY_GPU, data_type, self_k_cache_size, key_caches_[tgt_cache_id]},
                empty_tensor //Tensor{MEMORY_GPU, data_type, self_v_cache_size, value_caches_[tgt_cache_id]}
                };
        }
        // else {
        //     dynamic_decode_input_tensors = new std::vector<Tensor>{
        //         Tensor{MEMORY_GPU, data_type, {batch_size, beam_width_, vocab_size_padded_}, logits_buf_},
        //         Tensor{MEMORY_GPU, data_type, {vocab_size_padded_}, nullptr},
        //         Tensor{MEMORY_CPU, TYPE_INT32, {1}, &step},
        //         Tensor{MEMORY_CPU, TYPE_INT32, {1}, &max_input_length},
        //         Tensor{MEMORY_GPU, TYPE_INT32, {batch_size, beam_width_}, input_length_ptr},
        //         Tensor{MEMORY_CPU, TYPE_INT32, {1}, &tmp_ite}};

        //     dynamic_decode_output_tensors = new std::vector<Tensor>{
        //         Tensor{MEMORY_GPU, TYPE_INT32, {max_output_seq_len, batch_size, beam_width_}, output_ids_buf_},
        //         Tensor{MEMORY_GPU, TYPE_BOOL, {batch_size * beam_width_}, finished_buf_},
        //         output_tensors->at(2),
        //         Tensor{MEMORY_GPU,
        //                TYPE_FP32,
        //                {max_seq_len_, batch_size, beam_width_},
        //                output_cum_log_probs == nullptr ?
        //                    nullptr :
        //                    output_cum_log_probs + (step - max_input_length) * batch_size * beam_width_}};
        // }

        dynamic_decode_->forward(dynamic_decode_output_tensors, dynamic_decode_input_tensors);

        delete dynamic_decode_input_tensors;
        delete dynamic_decode_output_tensors;
    }

    {
        if (beam_width_ > 1) {
            // For beam search, do gather_tree
            invokeGatherTree((int*)output_tensors->at(1).data,
                             (int*)output_tensors->at(2).data,
                             static_cast<int>(max_output_seq_len),
                             static_cast<int>(batch_size),
                             static_cast<int>(beam_width_),
                             output_ids_buf_,
                             parent_ids_buf_,
                             -1,
                             stream_);

            //transpose and take output_parent_ids as inter buffer
            invokeTransposeAxis01((int*)output_tensors->at(0).data,
                                  (int*)output_tensors->at(1).data,
                                  static_cast<int>(max_output_seq_len), static_cast<int>(batch_size * beam_width_), 1, stream_);

            cudaD2Dcpy((int*)output_tensors->at(1).data,
                       parent_ids_buf_,
                       static_cast<int>(batch_size * beam_width_ * max_output_seq_len));

        }
        else {
            // For sampling, only transpose the results to output_tensor
            invokeTransposeAxis01((int*)output_tensors->at(0).data,
                                  output_ids_buf_,
                                  static_cast<int>(max_output_seq_len), static_cast<int>(batch_size * beam_width_), 1, stream_);
        }
    }
}

template class GptDecoding<float>;
template class GptDecoding<half>;

}  // namespace fastertransformer
