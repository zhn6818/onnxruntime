// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// OpTester creates a single node graph for testing, but graph inputs and outputs doesn't support strided tensor for
// now. To test strided tensor as Op input/output, here we create a graph with more than 2 nodes, in which some nodes at
// the graph beginning will generate strided tensor as outputs, and the nodes at the end will consume them. But we
// cannot validate the outputs of the nodes that generate strided tensors, as they are intermediate outputs and cannot
// be fetched during execution, we rely on ExecutionFrame testing to guarantee that they are strided tensor if possible.
// The test will valicate the final outputs, which are produced by those nodes that consume strided tensors.

#if defined(USE_CUDA) || defined(ENABLE_TRAINING)

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/providers/compare_provider_test_utils.h"
#include "test/util/include/asserts.h"
#include "core/session/IOBinding.h"
#include "test/framework/test_utils.h"

namespace onnxruntime {
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Cuda(
    const OrtCUDAProviderOptions* provider_options);
}

namespace onnxruntime {
namespace test {

std::unique_ptr<IExecutionProvider> CreateCudaExecutionProvider() {
  OrtCUDAProviderOptions options{};
  options.device_id = 0;
  auto factory = CreateExecutionProviderFactory_Cuda(&options);
  return factory->CreateProvider();
}

typedef std::vector<onnxruntime::NodeArg*> ArgVec;

void TestTranspose(const std::vector<int64_t>& perm1, const std::vector<int64_t>& perm2,
                   const std::vector<int64_t>& input_dims, const std::vector<float>& input_data,
                   const std::vector<int64_t>& output_dims, const std::vector<float>& output_data) {
  size_t rank = input_dims.size();
  auto cuda_xp = CreateCudaExecutionProvider();
  auto cuda_xp_type = cuda_xp->Type();
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[onnxruntime::kOnnxDomain] = 13;
  onnxruntime::Model model("strided_transpose_test", true, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           DefaultLoggingManager().DefaultLogger());
  onnxruntime::Graph& graph = model.MainGraph();
  ONNX_NAMESPACE::TypeProto tensor_float;
  tensor_float.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  onnxruntime::NodeArg input_def("X", &tensor_float), trans1_out_def("T", &tensor_float),
      trans2_out_def("Y", &tensor_float);
  ONNX_NAMESPACE::AttributeProto attr_perm1;
  const std::string attribute_name = "perm";
  attr_perm1.set_name(attribute_name);
  attr_perm1.set_type(ONNX_NAMESPACE::AttributeProto::INTS);
  for (size_t i = 0; i < rank; i++) attr_perm1.add_ints(perm1[i]);
  NodeAttributes attributes1({{attribute_name, attr_perm1}});
  ONNX_NAMESPACE::AttributeProto attr_perm2;
  attr_perm2.set_name(attribute_name);
  attr_perm2.set_type(ONNX_NAMESPACE::AttributeProto::INTS);
  for (size_t i = 0; i < rank; i++) attr_perm2.add_ints(perm2[i]);
  NodeAttributes attributes2({{attribute_name, attr_perm2}});
  graph.AddNode("trans1", "Transpose", "trans1", ArgVec{&input_def}, ArgVec{&trans1_out_def}, &attributes1)
      .SetExecutionProviderType(cuda_xp_type);
  graph.AddNode("trans2", "Transpose", "trans2", ArgVec{&trans1_out_def}, ArgVec{&trans2_out_def}, &attributes2)
      .SetExecutionProviderType(cuda_xp_type);
  ASSERT_STATUS_OK(graph.Resolve());

  OrtValue x_value;
  CPUExecutionProviderInfo epi;
  auto cpu_xp = std::make_unique<::onnxruntime::CPUExecutionProvider>(epi);
  CreateMLValue<float>(cpu_xp->GetAllocator(0, OrtMemTypeDefault), input_dims, input_data, &x_value);
  SessionOptions so;
  so.session_logid = "StridedTransposeTest";
  // Disable optimizations.
  so.graph_optimization_level = TransformerLevel::Default;
  InferenceSession session_obj{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_obj.RegisterExecutionProvider(std::move(cuda_xp)));
  std::stringstream buffer;
  model.ToProto().SerializeToOstream(&buffer);
  ASSERT_STATUS_OK(session_obj.Load(buffer));
  ASSERT_STATUS_OK(session_obj.Initialize());

  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", x_value));
  std::vector<std::string> output_names;
  output_names.push_back("Y");
  std::vector<OrtValue> fetches;
  RunOptions run_options;
  ASSERT_STATUS_OK(session_obj.Run(run_options, feeds, output_names, &fetches));
  EXPECT_EQ(fetches[0].Get<Tensor>().Shape(), TensorShape(output_dims));
  EXPECT_THAT(fetches[0].Get<Tensor>().DataAsSpan<float>(), ::testing::ContainerEq(gsl::make_span(output_data)));
}

void Permute(const std::vector<int64_t>& perm, const std::vector<int64_t>& input_dims,
             const std::vector<float>& input_data, std::vector<int64_t>& output_dims, std::vector<float>& output_data) {
  size_t rank = input_dims.size();
  output_dims.resize(rank);
  for (size_t i = 0; i < rank; ++i) {
    output_dims[i] = input_dims[static_cast<size_t>(perm[i])];
  }
  int64_t input_running_size = 1;
  int64_t output_running_size = 1;
  std::vector<int64_t> input_strides(rank);
  std::vector<int64_t> output_strides(rank);
  for (size_t i = rank - 1;; --i) {
    input_strides[i] = input_running_size;
    output_strides[i] = output_running_size;
    input_running_size *= input_dims[i];
    output_running_size *= output_dims[i];
    if (i == 0) break;
  }
  std::vector<int64_t> new_input_strides(rank);
  for (size_t i = 0; i < rank; i++) {
    new_input_strides[i] = input_strides[static_cast<size_t>(perm[i])];
  }
  output_data.resize(static_cast<size_t>(input_running_size));
  for (size_t i = 0; i < static_cast<size_t>(input_running_size); ++i) {
    size_t idx = 0;
    size_t remaining = i;
    for (size_t dim = 0; dim < rank; ++dim) {
      idx += (remaining / static_cast<size_t>(output_strides[dim])) * static_cast<size_t>(new_input_strides[dim]);
      remaining %= static_cast<size_t>(output_strides[dim]);
    }
    output_data[i] = input_data[idx];
  }
}

TEST(StridedTensorTest, Transpose) {
  // Transpose3DImpl from strided tensor to contiguous tensor.
  {
    const std::vector<int64_t> perm1{1, 2, 0};
    const std::vector<int64_t> perm2{0, 2, 1};
    const std::vector<int64_t> input_dims{32, 8, 16};
    std::vector<float> input_data(8 * 16 * 32);
    for (size_t i = 0; i < 8 * 16 * 32; ++i) input_data[i] = 0.01f * static_cast<float>(i);
    std::vector<int64_t> intermediate_dims;
    std::vector<float> intermediate_data;
    Permute(perm1, input_dims, input_data, intermediate_dims, intermediate_data);
    std::vector<int64_t> output_dims;
    std::vector<float> output_data;
    Permute(perm2, intermediate_dims, intermediate_data, output_dims, output_data);
    TestTranspose(perm1, perm2, input_dims, input_data, output_dims, output_data);
  }

  // Generic solution from strided tensor to contiguous tensor.
  {
    const std::vector<int64_t> perm1{1, 3, 2, 0};
    const std::vector<int64_t> perm2{0, 2, 1, 3};
    const std::vector<int64_t> input_dims{8, 16, 8, 4};
    std::vector<float> input_data(4 * 8 * 8 * 16);
    for (size_t i = 0; i < 4 * 8 * 8 * 16; ++i) input_data[i] = 0.01f * static_cast<float>(i);
    std::vector<int64_t> intermediate_dims;
    std::vector<float> intermediate_data;
    Permute(perm1, input_dims, input_data, intermediate_dims, intermediate_data);
    std::vector<int64_t> output_dims;
    std::vector<float> output_data;
    Permute(perm2, intermediate_dims, intermediate_data, output_dims, output_data);
    TestTranspose(perm1, perm2, input_dims, input_data, output_dims, output_data);
  }
}

void TestBinaryElementwise(const std::vector<int64_t>& lhs_perm, const std::vector<int64_t>& lhs_input_dims,
                           const std::vector<float>& lhs_input_data, const std::vector<int64_t>& rhs_perm,
                           const std::vector<int64_t>& rhs_input_dims, const std::vector<float>& rhs_input_data,
                           const std::vector<int64_t>& output_dims, const std::vector<float>& output_data) {
  auto cuda_xp = CreateCudaExecutionProvider();
  auto cuda_xp_type = cuda_xp->Type();
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[onnxruntime::kOnnxDomain] = 13;
  onnxruntime::Model model("strided_transpose_test", true, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           DefaultLoggingManager().DefaultLogger());
  onnxruntime::Graph& graph = model.MainGraph();
  ONNX_NAMESPACE::TypeProto tensor_float;
  tensor_float.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  onnxruntime::NodeArg lhs_input_def("X1", &tensor_float), rhs_input_def("X2", &tensor_float),
      lhs_trans_out_def("T1", &tensor_float), rhs_trans_out_def("T2", &tensor_float), output_def("Y", &tensor_float);
  size_t lhs_rank = lhs_perm.size();
  size_t rhs_rank = rhs_perm.size();
  const std::string attribute_name = "perm";
  if (lhs_rank > 0) {
    ONNX_NAMESPACE::AttributeProto lhs_attr_perm;
    lhs_attr_perm.set_name(attribute_name);
    lhs_attr_perm.set_type(ONNX_NAMESPACE::AttributeProto::INTS);
    for (size_t i = 0; i < lhs_rank; i++) lhs_attr_perm.add_ints(lhs_perm[i]);
    NodeAttributes lhs_attributes({{attribute_name, lhs_attr_perm}});
    graph
        .AddNode("lhs_trans", "Transpose", "lhs_trans", ArgVec{&lhs_input_def}, ArgVec{&lhs_trans_out_def},
                 &lhs_attributes)
        .SetExecutionProviderType(cuda_xp_type);
  }
  if (rhs_rank > 0) {
    ONNX_NAMESPACE::AttributeProto rhs_attr_perm;
    rhs_attr_perm.set_name(attribute_name);
    rhs_attr_perm.set_type(ONNX_NAMESPACE::AttributeProto::INTS);
    for (size_t i = 0; i < rhs_rank; i++) rhs_attr_perm.add_ints(rhs_perm[i]);
    NodeAttributes rhs_attributes({{attribute_name, rhs_attr_perm}});
    graph
        .AddNode("rhs_trans", "Transpose", "rhs_trans", ArgVec{&rhs_input_def}, ArgVec{&rhs_trans_out_def},
                 &rhs_attributes)
        .SetExecutionProviderType(cuda_xp_type);
  }
  ArgVec add_inputs;
  add_inputs.emplace_back(lhs_rank > 0 ? &lhs_trans_out_def : &lhs_input_def);
  add_inputs.emplace_back(rhs_rank > 0 ? &rhs_trans_out_def : &rhs_input_def);
  graph.AddNode("add", "Add", "add", add_inputs, ArgVec{&output_def});
  ASSERT_STATUS_OK(graph.Resolve());

  OrtValue x1_value, x2_value;
  CPUExecutionProviderInfo epi;
  auto cpu_xp = std::make_unique<::onnxruntime::CPUExecutionProvider>(epi);
  CreateMLValue<float>(cpu_xp->GetAllocator(0, OrtMemTypeDefault), lhs_input_dims, lhs_input_data, &x1_value);
  CreateMLValue<float>(cpu_xp->GetAllocator(0, OrtMemTypeDefault), rhs_input_dims, rhs_input_data, &x2_value);
  SessionOptions so;
  so.session_logid = "StridedTransposeTest";
  // Disable optimizations.
  so.graph_optimization_level = TransformerLevel::Default;
  InferenceSession session_obj{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_obj.RegisterExecutionProvider(std::move(cuda_xp)));
  std::stringstream buffer;
  model.ToProto().SerializeToOstream(&buffer);
  ASSERT_STATUS_OK(session_obj.Load(buffer));
  ASSERT_STATUS_OK(session_obj.Initialize());

  NameMLValMap feeds;
  feeds.insert(std::make_pair("X1", x1_value));
  feeds.insert(std::make_pair("X2", x2_value));
  std::vector<std::string> output_names;
  output_names.push_back("Y");
  std::vector<OrtValue> fetches;
  RunOptions run_options;
  ASSERT_STATUS_OK(session_obj.Run(run_options, feeds, output_names, &fetches));
  EXPECT_EQ(fetches[0].Get<Tensor>().Shape(), TensorShape(output_dims));
  EXPECT_THAT(fetches[0].Get<Tensor>().DataAsSpan<float>(), ::testing::ContainerEq(gsl::make_span(output_data)));
}

TEST(StridedTensorTest, BinaryElementwise) {
  // Non-broadcast, left strided.
  {
    const std::vector<int64_t> lhs_perm{1, 2, 0};
    const std::vector<int64_t> lhs_input_dims{2, 2, 3};
    const std::vector<int64_t> rhs_input_dims{2, 3, 2};
    const std::vector<float> input_data{.1f, .2f, .3f, .4f, .5f, .6f, .7f, .8f, .9f, 1.f, 1.1f, 1.2f};
    const std::vector<int64_t> output_dims{2, 3, 2};
    const std::vector<float> output_data{.2f, .9f, .5f, 1.2f, .8f, 1.5f, 1.1f, 1.8f, 1.4f, 2.1f, 1.7f, 2.4f};
    TestBinaryElementwise(lhs_perm, lhs_input_dims, input_data, {}, rhs_input_dims, input_data, output_dims,
                          output_data);
  }

  // Broadcast, right strided.
  {
    const std::vector<int64_t> lhs_input_dims{3, 1, 1};
    const std::vector<float> lhs_input_data{1.f, 2.f, 3.f};
    const std::vector<int64_t> rhs_perm{1, 0, 2};
    const std::vector<int64_t> rhs_input_dims{2, 3, 2};
    const std::vector<float> rhs_input_data{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f};
    const std::vector<int64_t> output_dims{3, 2, 2};
    const std::vector<float> output_data{2.f, 3.f, 8.f, 9.f, 5.f, 6.f, 11.f, 12.f, 8.f, 9.f, 14.f, 15.f};
    TestBinaryElementwise({}, lhs_input_dims, lhs_input_data, rhs_perm, rhs_input_dims, rhs_input_data, output_dims,
                          output_data);
  }

  // Batched broadcast, left transpose, but still contiguous.
  {
    const std::vector<int64_t> lhs_perm{1, 0};
    const std::vector<int64_t> lhs_input_dims{1, 2};
    const std::vector<float> lhs_input_data{1.f, 2.f};
    const std::vector<int64_t> rhs_input_dims{2, 2, 3};
    const std::vector<float> rhs_input_data{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f};
    const std::vector<int64_t> output_dims{2, 2, 3};
    const std::vector<float> output_data{2.f, 3.f, 4.f, 6.f, 7.f, 8.f, 8.f, 9.f, 10.f, 12.f, 13.f, 14.f};
    TestBinaryElementwise(lhs_perm, lhs_input_dims, lhs_input_data, {}, rhs_input_dims, rhs_input_data, output_dims,
                          output_data);
  }

  // Broadcast, both strided.
  {
    const std::vector<int64_t> lhs_perm{2, 1, 0};
    const std::vector<int64_t> lhs_input_dims{2, 1, 3};
    const std::vector<float> lhs_input_data{1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    const std::vector<int64_t> rhs_perm{1, 0, 2};
    const std::vector<int64_t> rhs_input_dims{2, 3, 2};
    const std::vector<float> rhs_input_data{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f};
    const std::vector<int64_t> output_dims{3, 2, 2};
    const std::vector<float> output_data{2.f, 6.f, 8.f, 12.f, 5.f, 9.f, 11.f, 15.f, 8.f, 12.f, 14.f, 18.f};
    TestBinaryElementwise(lhs_perm, lhs_input_dims, lhs_input_data, rhs_perm, rhs_input_dims, rhs_input_data,
                          output_dims, output_data);
  }
}

}  // namespace test
}  // namespace onnxruntime

#endif
