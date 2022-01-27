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
/**
 * Memory Allocator
 **/

#pragma once

#include <cuda_runtime.h>
#include <vector>

#include "cuda_utils.h"

namespace fastertransformer {

enum class AllocatorType
{
    CUDA,
    TF,
    TH
};

class IAllocator {
public:
    virtual void* malloc(size_t size, const bool is_set_zero = true) const = 0;
    virtual void free(void* ptr) const = 0;
};

template<AllocatorType AllocType_>
class Allocator;

template<>
class Allocator<AllocatorType::CUDA>: public IAllocator {
    const int device_id_;

public:
    Allocator(int device_id): device_id_(device_id) {}
    virtual ~Allocator() {}

    void* malloc(size_t size, const bool is_set_zero = true) const
    {
        void* ptr = nullptr;
        int o_device = 0;
        check_cuda_error(getSetDevice(device_id_, &o_device));
        check_cuda_error(cudaMalloc(&ptr, (size_t)(ceil(size / 32.)) * 32));
        check_cuda_error(getSetDevice(o_device));
        return ptr;
    }

    void free(void* ptr) const
    {
        int o_device = 0;
        check_cuda_error(getSetDevice(device_id_, &o_device));
        check_cuda_error(cudaFree(ptr));
        check_cuda_error(getSetDevice(o_device));
        return;
    }
};


}  // namespace fastertransformer
