// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "rocm_profiler.h"
#include <map>
#include <string>
#include <iostream>

namespace onnxruntime {
namespace profiling {

#define ROCT_OK(command) \
  do { \
    ORT_ENFORCE((command) == ROCTRACER_STATUS_SUCCESS, "Failed roctracer operation on ", roctracer_error_string()); \
  } while (0)

std::atomic_flag RocmProfiler::enabled{0};

void RocmProfiler::ApiCallback(uint32_t domain, uint32_t cid, const void* callback_data, void* arg) {

  // capture correlation_id and kernel name and launch parameters
  const hip_api_data_t* data = (const hip_api_data_t*)(callback_data);

  if (data->phase == ACTIVITY_API_PHASE_ENTER) {
    switch (cid) {
        case HIP_API_ID_hipLaunchKernel:
        case HIP_API_ID_hipExtLaunchKernel:
        {
          std::string name(hipKernelNameRefByPtr(
            data->args.hipLaunchKernel.function_address, 
            data->args.hipLaunchKernel.stream));
	  printf("tracing %s\n", name.c_str());
	  fflush(stdout);

          /* data->correlation_id;
          data->args.hipLaunchKernel.blockDimX;
          data->args.hipLaunchKernel.blockDimY;
          data->args.hipLaunchKernel.blockDimZ;
          data->args.hipLaunchKernel.gridDimX;
          data->args.hipLaunchKernel.gridDimY;
          data->args.hipLaunchKernel.gridDimZ; */

          // push to queue to push to db (api table with cid primary key)
        } 
        break;

        case HIP_API_ID_hipHccModuleLaunchKernel:
        case HIP_API_ID_hipModuleLaunchKernel:
        case HIP_API_ID_hipExtModuleLaunchKernel:
        {
          const hipFunction_t f = data->args.hipModuleLaunchKernel.f;
          if (f != nullptr) {
              std::string name(hipKernelNameRef(f));
              // data->correlation_id;

   	      printf("tracing %s\n", name.c_str());
	      fflush(stdout);
          } 

          // push to queue to push to db (api table with cid primary key)
        } 
        break;
    }
  }
}

void RocmProfiler::OpsCallback(const char* begin, const char* end, void* arg) {

  // capture correlation_id and kernel begin/end and external id
  // push to queue to push to db (ops table with cid primary key)
}

bool RocmProfiler::StartProfiling() {
  if (!enabled.test_and_set()) {
    try {
      printf("Initialize roctracer profiling\n");
      fflush(stdout);

      ROCT_OK(roctracer_set_properties(ACTIVITY_DOMAIN_HIP_API, NULL));
      
      ROCT_OK(roctracer_enable_op_callback(ACTIVITY_DOMAIN_HIP_API, HIP_API_ID_hipLaunchKernel, ApiCallback, NULL));
      ROCT_OK(roctracer_enable_op_callback(ACTIVITY_DOMAIN_HIP_API, HIP_API_ID_hipExtLaunchKernel, ApiCallback, NULL));
      ROCT_OK(roctracer_enable_op_callback(ACTIVITY_DOMAIN_HIP_API, HIP_API_ID_hipHccModuleLaunchKernel, ApiCallback, NULL));
      ROCT_OK(roctracer_enable_op_callback(ACTIVITY_DOMAIN_HIP_API, HIP_API_ID_hipModuleLaunchKernel, ApiCallback, NULL));
      ROCT_OK(roctracer_enable_op_callback(ACTIVITY_DOMAIN_HIP_API, HIP_API_ID_hipExtModuleLaunchKernel, ApiCallback, NULL));
      ROCT_OK(roctracer_enable_op_callback(ACTIVITY_DOMAIN_HIP_API, HIP_API_ID_hipMemcpy, ApiCallback, NULL));
      ROCT_OK(roctracer_enable_op_callback(ACTIVITY_DOMAIN_HIP_API, HIP_API_ID_hipMemcpyAsync, ApiCallback, NULL));
      ROCT_OK(roctracer_enable_op_callback(ACTIVITY_DOMAIN_HIP_API, HIP_API_ID_hipMemcpyWithStream, ApiCallback, NULL));

      roctracer_properties_t properties;
      memset(&properties, 0, sizeof(roctracer_properties_t));
      properties.buffer_size = 0x1000;
      ROCT_OK(roctracer_open_pool(&properties));
  
      roctracer_properties_t hcc_cb_properties;
      memset(&hcc_cb_properties, 0, sizeof(roctracer_properties_t));
      hcc_cb_properties.buffer_size = 0x40000;
      hcc_cb_properties.buffer_callback_fun = OpsCallback;
      ROCT_OK(roctracer_open_pool_expl(&hcc_cb_properties, &hccPool));
  
      roctracer_start();
      initialized_ = true;
      return true;
    }
    catch(const OnnxRuntimeException& ex) {
      // LOGS_DEFAULT(WARNING) << ex.what();
      DisableEvents();
      enabled.clear();
      return false;
    };
  }

  return false;
}

void RocmProfiler::EndProfiling(TimePoint start_time, std::vector<EventRecord>& events) {
  std::map<uint64_t, std::vector<EventRecord>> event_map;
  if (initialized_) {

     // read from db and join api and ops table using cid
     // modify events by looking up eid for every cid
  } 
}

RocmProfiler::~RocmProfiler() {
  if (initialized_) {
    DisableEvents();
    Clear();
  }
}

void RocmProfiler::Start(uint64_t id) {
  if (initialized_) {
    roctracer_activity_push_external_correlation_id(id);
  }
}

void RocmProfiler::Stop(uint64_t) {
  if (initialized_) {
    uint64_t last_id{0};
    roctracer_activity_pop_external_correlation_id(&last_id);
  }
}

void RocmProfiler::DisableEvents() {
    roctracer_stop();
    roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API);
    roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HSA_OPS);
    roctracer_disable_domain_callback(ACTIVITY_DOMAIN_ROCTX);

    roctracer_flush_activity();
    roctracer_flush_activity_expl(hccPool);
}

void RocmProfiler::Clear() {
  if (initialized_) {
    initialized_ = false;
    enabled.clear();
  }
}

}  // namespace profiling
}  // namespace onnxruntime
