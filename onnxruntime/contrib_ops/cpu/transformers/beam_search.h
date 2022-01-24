// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/controlflow/utils.h"
#include "beam_search_parameters.h"
#include "gpt_subgraph.h"
#include "beam_search_device_helper.h"

namespace onnxruntime {
class FeedsFetchesManager;

namespace contrib {
namespace transformers {

using namespace onnxruntime::controlflow; // namespace of IControlFlowKernel

class BeamSearch : public IControlFlowKernel {
 public:
  BeamSearch(const OpKernelInfo& info) : IControlFlowKernel(info) {
    Init(info);
  }

  void Init(const OpKernelInfo& info);

  Status Compute(OpKernelContext* ctx) const override;

  Status SetupSubgraphExecutionInfo(const SessionState& session_state,
                                    const std::string& attribute_name,
                                    const SessionState& subgraph_session_state) override;

 protected:
  void SetComputeStream(void* stream) { stream_ = stream; }

  void SetDeviceHelpers(const BeamSearchDeviceHelper::CreateInputsFunc& create_inputs_func,
                        const BeamSearchDeviceHelper::TopkFunc& topk_func) {
    create_inputs_func_ = create_inputs_func;
    topk_func_ = topk_func;
  }

 private:
  // Device specific functions
  BeamSearchDeviceHelper::CreateInputsFunc create_inputs_func_;
  BeamSearchDeviceHelper::TopkFunc topk_func_;

  // Subgraph and FeedsFetchesManager re-used for each subgraph execution.
  std::unique_ptr<GptSubgraph> gpt_subgraph_;
  FeedsFetchesManager* feeds_fetches_manager_;

  void* stream_;

  BeamSearchParameters parameters_;
};

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
