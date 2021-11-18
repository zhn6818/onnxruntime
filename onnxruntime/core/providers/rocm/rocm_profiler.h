// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/common/profiler_common.h"

#ifdef USE_ROCM

#include "core/platform/ort_mutex.h"
#include <atomic>
#include <mutex>
#include <vector>

#include <roctracer/roctracer_hip.h>
#include <roctracer/roctracer_hcc.h>
#include <roctracer/roctracer_ext.h>
#include <roctracer/roctracer_roctx.h>
#include <roctracer/roctx.h>

namespace onnxruntime {
namespace profiling {

class RocmProfiler final : public EpProfiler {
 public:
  RocmProfiler() = default;
  RocmProfiler(const RocmProfiler&) = delete;
  RocmProfiler& operator=(const RocmProfiler&) = delete;
  RocmProfiler(RocmProfiler&& rocm_profiler) noexcept {
    initialized_ = rocm_profiler.initialized_;
    rocm_profiler.initialized_ = false;
  }
  RocmProfiler& operator=(RocmProfiler&& rocm_profiler) noexcept {
    initialized_ = rocm_profiler.initialized_;
    rocm_profiler.initialized_ = false;
    return *this;
  }
  ~RocmProfiler();
  bool StartProfiling() override;
  void EndProfiling(TimePoint start_time, std::vector<EventRecord>& events) override;
  void Start(uint64_t) override;
  void Stop(uint64_t) override;

 private:
  static void ApiCallback(uint32_t domain, uint32_t cid, const void* callback_data, void* arg);
  static void OpsCallback(const char* begin, const char* end, void* arg);
 
  struct KernelExecutionDesc {
    std::string name_ = {};
    uint32_t stream_ = 0;
    int32_t grid_x_ = 0;
    int32_t grid_y_ = 0;
    int32_t grid_z_ = 0;
    int32_t block_x_ = 0;
    int32_t block_y_ = 0;
    int32_t block_z_ = 0;
    int64_t start_ = 0;
    int64_t stop_ = 0;
    uint32_t correlation_id = 0;
  };
  static std::atomic_flag enabled;

  roctracer_pool_t *hccPool;

  void DisableEvents();
  void Clear();
  bool initialized_ = false;
};

}  // namespace profiling
}  // namespace onnxruntime

#else 

namespace onnxruntime {
namespace profiling {

class RocmProfiler final : public EpProfiler {
 public:
  bool StartProfiling() override { return true; }
  void EndProfiling(TimePoint, std::vector<EventRecord>&) override{};
  void Start(uint64_t) override{};
  void Stop(uint64_t) override{};
};

}  // namespace profiling
}  // namespace onnxruntime

#endif

