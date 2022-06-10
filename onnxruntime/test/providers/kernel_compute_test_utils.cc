// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRAINING

#include "test/providers/kernel_compute_test_utils.h"

#include "core/optimizer/optimizer_execution_frame.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

void KernelComputeTester::Run(std::unordered_set<int> strided_outputs) {
  auto cpu_ep = DefaultCpuExecutionProvider();
  auto cpu_ep_type = cpu_ep->Type();
  DataTransferManager dtm;
  auto cpu_transfer = cpu_ep->GetDataTransfer();
  dtm.RegisterDataTransfer(std::move(cpu_transfer));
  ExecutionProviders execution_providers;
  execution_providers.Add(cpu_ep_type, std::move(cpu_ep));
  auto ep_type = cpu_ep_type;
#ifdef USE_CUDA
  if (provider_ == kCudaExecutionProvider) {
    auto cuda_ep = DefaultCudaExecutionProvider();
    ep_type = cuda_ep->Type();
    auto cuda_transfer = cuda_ep->GetDataTransfer();
    dtm.RegisterDataTransfer(std::move(cuda_transfer));
    execution_providers.Add(ep_type, std::move(cuda_ep));
  }
#endif
#ifdef USE_ROCM
  if (provider_ == kRocmExecutionProvider) {
    auto rocm_ep = DefaultRocmExecutionProvider();
    ep_type = rocm_ep->Type();
    auto rocm_transfer = rocm_ep->GetDataTransfer();
    dtm.RegisterDataTransfer(std::move(rocm_transfer));
    execution_providers.Add(ep_type, std::move(rocm_ep));
  }
#endif

  Model model("test", false, ModelMetaData(), ORT_TSTR(""), IOnnxRuntimeOpSchemaRegistryList(),
              {{domain_, opset_version_}}, {}, DefaultLoggingManager().DefaultLogger());

  std::vector<NodeArg*> input_args;
  std::unordered_map<std::string, OrtValue> initializer_map;
  for (auto& data : input_data_) {
    input_args.emplace_back(&data.def_);
    const auto& name = data.def_.Name();
    if (provider_ == kCpuExecutionProvider || data.is_cpu_data_) {
      initializer_map[name] = data.value_;
    }
#if defined(USE_CUDA) || defined(USE_ROCM)
    if ((provider_ == kCudaExecutionProvider || provider_ == kRocmExecutionProvider) && !data.is_cpu_data_) {
      OrtValue gpu_value;
      const Tensor& tensor = data.value_.Get<Tensor>();
      Tensor::InitOrtValue(tensor.DataType(), tensor.Shape(),
                           execution_providers.Get(ep_type)->GetAllocator(0, OrtMemTypeDefault), gpu_value,
                           tensor.Strides());
      dtm.CopyTensor(tensor, *gpu_value.GetMutable<Tensor>());
      initializer_map[name] = gpu_value;
    }
#endif
  }

  std::vector<NodeArg*> output_args;
  for (auto& data : output_data_) {
    output_args.emplace_back(&data.def_);
  }

  Graph& graph = model.MainGraph();
  auto& node = graph.AddNode("node", op_, op_, input_args, output_args, nullptr, domain_);
  for (auto& add_attribute_fn : add_attribute_funcs_) {
    add_attribute_fn(node);
  }
  graph.Resolve();

  node.SetExecutionProviderType(ep_type);
  OptimizerExecutionFrame::Info info({&node}, initializer_map, graph.ModelPath(), *execution_providers.Get(ep_type),
                                     [](std::string const&) { return false; });
  const KernelCreateInfo* kernel_create_info = nullptr;
  info.TryFindKernel(&node, &kernel_create_info);
  if (!kernel_create_info) {
    ORT_THROW("Could not find kernel");
  }

  const auto& may_strided_outputs_map = kernel_create_info->kernel_def->MayStridedOutput();
  std::vector<OrtValue> outputs;
  for (size_t i = 0; i < output_data_.size(); ++i) {
    OrtValue output;
    const Tensor& tensor = output_data_[i].value_.Get<Tensor>();
    if (strided_outputs.find(static_cast<int>(i)) != strided_outputs.end()) {
      for (auto& pair : may_strided_outputs_map) {
        if (pair.second == static_cast<int>(i)) {
          Tensor::InitOrtValue(tensor.DataType(), tensor.Shape(),
                               initializer_map[input_data_[static_cast<size_t>(pair.first)].def_.Name()]
                                   .GetMutable<Tensor>()
                                   ->MutableDataRaw(),
                               execution_providers.Get(ep_type)->GetAllocator(0, OrtMemTypeDefault)->Info(), output);
          break;
        }
      }
    } else {
      Tensor::InitOrtValue(tensor.DataType(), tensor.Shape(),
                           execution_providers.Get(ep_type)->GetAllocator(0, OrtMemTypeDefault), output);
    }
    outputs.emplace_back(output);
  }

  auto kernel = info.CreateKernel(&node);
  if (!kernel) {
    ORT_THROW("Could not create kernel");
  }

  std::vector<int> fetch_mlvalue_idxs;
  for (const auto* node_out : node.OutputDefs()) {
    fetch_mlvalue_idxs.push_back(info.GetMLValueIndex(node_out->Name()));
  }

  OptimizerExecutionFrame frame(info, fetch_mlvalue_idxs, outputs);
  OpKernelContext op_kernel_context(&frame, kernel.get(), nullptr, DefaultLoggingManager().DefaultLogger());
  kernel->Compute(&op_kernel_context);
  frame.GetOutputs(outputs);
  for (size_t i = 0; i < output_data_.size(); ++i) {
    if (strided_outputs.find(static_cast<int>(i)) != strided_outputs.end()) {
      for (auto& pair : may_strided_outputs_map) {
        if (pair.second == static_cast<int>(i)) {
          EXPECT_EQ(outputs[i].Get<Tensor>().DataRaw(),
                    initializer_map[input_data_[static_cast<size_t>(pair.first)].def_.Name()].Get<Tensor>().DataRaw());
          EXPECT_EQ(outputs[i].Get<Tensor>().Strides(), output_data_[i].value_.Get<Tensor>().Strides());
          break;
        }
      }
    } else {
      OrtValue cpu_value;
      if (provider_ == kCpuExecutionProvider) {
        cpu_value = outputs[i];
      } else {
        const Tensor& tensor = outputs[i].Get<Tensor>();
        Tensor::InitOrtValue(tensor.DataType(), tensor.Shape(),
                             execution_providers.Get(cpu_ep_type)->GetAllocator(0, OrtMemTypeDefault), cpu_value,
                             tensor.Strides());
        dtm.CopyTensor(tensor, *cpu_value.GetMutable<Tensor>());
      }
      optional<float> rel;
      optional<float> abs;
      OpTester::Data expected(std::move(output_data_[i].def_), std::move(output_data_[i].value_), std::move(rel),
                              std::move(abs));
      Check(expected, cpu_value.Get<Tensor>(), provider_);
    }
  }
}

}  // namespace test
}  // namespace onnxruntime

#endif
