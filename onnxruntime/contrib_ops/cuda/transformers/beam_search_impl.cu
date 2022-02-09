// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "beam_search_impl.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

__global__
void NextTokenKernel(const int64_t* next_token_indices,
                     int64_t* next_indices,
                     int64_t* next_tokens,
                     int vocab_size,
                     int total_elements)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < total_elements)
  {
      next_indices[index] = next_token_indices[index] / vocab_size;
      next_tokens[index] = next_token_indices[index] % vocab_size;
  }
}

/* NextToken kernel is corresponding to CPU logic like the following:
      int offset = 0;
      for (int i = 0; i < batch_size; i++) {
        for (unsigned int j = 0; j < top_k; j++, offset++) {
          next_indices[offset] = next_token_indices[offset] / vocab_size;
          next_tokens[offset] = next_token_indices[offset] % vocab_size;
        }
      }
*/
void LaunchNextTokenKernel(const int64_t* next_token_indices,
                          int64_t* next_indices,
                          int64_t* next_tokens,
                          int batch_size,
                          int top_k,
                          int vocab_size,
                          cudaStream_t stream){
  int total_elements = batch_size * top_k;
  constexpr int blockSize = 256;
  const int gridSize = (total_elements + blockSize - 1) / blockSize;
  NextTokenKernel<<<gridSize, blockSize, 0, stream>>>(next_token_indices, next_indices, next_tokens, vocab_size, total_elements);
}

__global__
void InitKernel(float* beam_scores, 
              int num_beams, 
              int total_elements)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < total_elements)
  {
    int beam_index = index % num_beams;
    beam_scores[index] = beam_index > 0 ? static_cast<float>(-1e9): 0.0f; // This value exceeds limit of MLFloat16 so it is for float only.
  }
}

void LaunchInitKernel(
  float* beam_scores,
  int batch_size,
  int num_beams,
  cudaStream_t stream){
    int total_elements = batch_size * num_beams;
    constexpr int blockSize = 256;
    const int gridSize = (total_elements + blockSize - 1) / blockSize;
    InitKernel<<<gridSize, blockSize, 0, stream>>>(beam_scores, num_beams, total_elements);
  }

template<typename T>
__global__
void VocabMaskKernel(T* log_probs, 
                      const int* vocab_mask, 
                      int vocab_size, 
                      int total_elements)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int word_id = index % vocab_size;

  if (index < total_elements && vocab_mask[word_id] == 0) {
      log_probs[index] = std::numeric_limits<T>::lowest();
  }
}


template <typename T>
void LaunchVocabMaskKernel(
  T* log_probs,
  const int* vocab_mask,
  int batch_size,
  int num_beams,
  int vocab_size,
  cudaStream_t stream){
int total_elements = batch_size * num_beams * vocab_size;
constexpr int blockSize = 256;
const int gridSize = (total_elements + blockSize - 1) / blockSize;
VocabMaskKernel<float><<<gridSize, blockSize, 0, stream>>>(log_probs, vocab_mask, vocab_size, total_elements);
}

// Instantiation
template void LaunchVocabMaskKernel(
  float* log_probs,
  const int* vocab_mask,
  int batch_size,
  int num_beams,
  int vocab_size,
  cudaStream_t stream);

template<typename T>
__global__
void AddProbsKernel(T* log_probs, 
                  T* cum_log_probs, 
                  const int vocab_size, 
                  const int total_elements)
{
int index = blockIdx.x * blockDim.x + threadIdx.x;
int batch_beam_index = index / vocab_size;

if (index < total_elements)
  log_probs[index] += cum_log_probs[batch_beam_index];
}

void LaunchAddProbsKernel(float* log_probs, 
                        float* cum_log_probs, 
                        const int batch_size, 
                        const int num_beams, 
                        const int vocab_size, 
                        cudaStream_t stream)
{
int total_elements = batch_size * num_beams * vocab_size;
constexpr int blockSize = 256;
const int gridSize = (total_elements + blockSize - 1) / blockSize;
AddProbsKernel<float><<<gridSize, blockSize, 0, stream>>>(log_probs, cum_log_probs, vocab_size, total_elements);
}




template <typename T>
__global__
void UpdateInputsKernel(const T* old_mask_data,
                        T* mask_data,
                        int64_t* next_positions,
                        int batch_beam_size,
                        int current_length)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < batch_beam_size * current_length)
  {
    // Update attention mask like the following:
    //   for (int i = 0; i < batch_beam_size; i++) {
    //     for (int j = 0; j < current_length - 1; j++) {
    //       mask_data[i * current_length + j] = old_mask_data[i * (current_length - 1) + j];
    //     }
    //     mask_data[i * current_length + current_length - 1] = 1.0f;
    //   }
    int i = index / current_length;
    int j = index % current_length;
    mask_data[index] = (j < current_length - 1) ? old_mask_data[i * (current_length - 1) + j] : static_cast<T>(1.0f);

    // Update sequence length (or next positions) like the following:
    //   for (int i = 0; i < batch_beam_size; i++) {
    //     next_positions[i]++;
    //   }
    if (index < batch_beam_size) {
      next_positions[i]++;
    }
  }
}

template <typename T>
void LaunchUpdateKernel(const T* old_mask_data,
                        T* mask_data,
                        int64_t* next_positions,
                        int batch_beam_size, 
                        int current_length,
                        cudaStream_t stream) {
  assert(current_length > 0);
  int total_elements = batch_beam_size * current_length;
  constexpr int blockSize = 256;
  const int gridSize = (total_elements + blockSize - 1) / blockSize;
  UpdateInputsKernel<T><<<gridSize, blockSize, 0, stream>>>(old_mask_data, mask_data, next_positions, batch_beam_size, current_length);
}

// Instantiation
template void LaunchUpdateKernel(const float* old_mask_data,
                                 float* mask_data,
                                 int64_t* next_positions,
                                 int batch_beam_size, 
                                 int current_length,
                                 cudaStream_t stream);

/*
template <typename T>
__global__
void ProcessLogitsKernel(
int step,
int vocab_size, 
int beam_width,
T* log_probs, 
int* current_ids,
int* previous_ids,
int* parent_ids,
int  end_id,
float inv_temp,
float len_penalty,
float repeat_penalty,
int* vocab_mask) {

int tid = threadIdx.x;
int bid = blockIdx.x;
int bbid = blockIdx.y;
int bbsize = gridDim.y;
int batchid = bbid / beam_width;
// int beamid = bbid % beam_width;

for (int i = tid + bid*blockDim.x; i < vocab_size; i +=  blockDim.x*gridDim.x) {
  log_probs[i + bbid * vocab_size] *= inv_temp;
}
if (tid == 0 && bid == 0) {
  // apply repetition penalty (this can apply the penalty multiple times to a repeated word).
  int prev_id = current_ids[bbid];
  if (log_probs[prev_id+bbid*vocab_size] > T(0)) {
    log_probs[prev_id+bbid*vocab_size] = float(log_probs[prev_id+bbid*vocab_size]) / repeat_penalty;
  } else {
    log_probs[prev_id+bbid*vocab_size] = float(log_probs[prev_id+bbid*vocab_size]) * repeat_penalty;
  }
  if (step > 1) {
    int parent_beamid = parent_ids[bbsize*(step-2) + bbid];
    for (int i = step-2; i > 0; --i) {
      prev_id = previous_ids[bbsize*i+batchid*beam_width+parent_beamid];
      if (log_probs[prev_id+bbid*vocab_size] > T(0)) {
        log_probs[prev_id+bbid*vocab_size] = float(log_probs[prev_id+bbid*vocab_size]) / repeat_penalty;
      } else {
        log_probs[prev_id+bbid*vocab_size] = float(log_probs[prev_id+bbid*vocab_size]) * repeat_penalty;
      }
      //if (i > 0) parent_beamid = parent_ids[bbsize*(i-1)+parent_beamid];
      parent_beamid = parent_ids[bbsize*(i-1)+parent_beamid];
    }
  }
  prev_id = previous_ids[batchid*beam_width];
  if (log_probs[prev_id+bbid*vocab_size] > T(0)) {
    log_probs[prev_id+bbid*vocab_size] = float(log_probs[prev_id+bbid*vocab_size]) / repeat_penalty;
  } else {
    log_probs[prev_id+bbid*vocab_size] = float(log_probs[prev_id+bbid*vocab_size]) * repeat_penalty;
  }
  // apply length penalty
  if (log_probs[end_id+bbid*vocab_size] > T(0))  {
    log_probs[end_id+bbid*vocab_size] = float(log_probs[end_id+bbid*vocab_size]) / len_penalty;
  } else {
    log_probs[end_id+bbid*vocab_size] = float(log_probs[end_id+bbid*vocab_size]) * len_penalty;
  }
}
}

template <typename T>
void LaunchLogitsProcessors(int step, 
                            T* log_probs, 
                            int* current_ids,
                            int* previous_ids, 
                            int* parent_ids,

                            int eos_token_id,
                            float temperature,
                            float length_penalty,
                            float repetition_penalty,
                            int* vocab_mask,
                            int vocab_size,
                            int batch_size,
                            int num_beams,

                            cudaStream_t stream) {
  int beam_width = 1; //?

  dim3 block(256);
  dim3 grid((vocab_size + block.x - 1) / block.x, beam_width * batch_size);

  ProcessLogitsKernel<T><<<grid, block, 0, stream>>>(
    step, 
    vocab_size, 
    beam_width, 
    log_probs, 
    current_ids,
    previous_ids, 
    parent_ids,
    eos_token_id, 
    1.f / temperature, 
    length_penalty,
    repetition_penalty, 
    vocab_mask);
}
*/
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime