/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "import/OMKernelConfigureBuilder.h"
#include "core/OMUtils.h"
#include "OMStatus.h"
#include "execute/OMRuntimeKernel.h"
#include "core/OMShape.h"

using namespace onert_micro;
using namespace onert_micro::core;

namespace
{

constexpr uint32_t input1TensorIdx = 0;
constexpr uint32_t outputTensorIdx = 0;

} // namespace


OMStatus onert_micro::import::configure_kernel_CircleWhile(const OMConfigureArgs &config_args)
{
  OMRuntimeContext &runtime_context = config_args.runtime_context;
  uint16_t op_index = config_args.kernel_index;
  core::OMRuntimeModule &runtime_module = config_args.runtime_module;

  onert_micro::execute::OMRuntimeKernel runtime_kernel;

  OMStatus status = runtime_kernel.readKernel(op_index, runtime_context);
  if (status != Ok)
    return status;

  const auto *options = runtime_kernel.first_operator->builtin_options_as_WhileOptions();
  const auto body_subgraph_index = options->body_subgraph_index();
  const auto cond_subgraph_index = options->cond_subgraph_index();

  OMRuntimeGraph *body_runtime_graph = nullptr;
  OMRuntimeGraph *cond_runtime_graph = nullptr;

  status = runtime_module.getRuntimeGraphAt(0, &body_runtime_graph);
  if (status != Ok)
    return status;

  status = runtime_module.getRuntimeGraphAt(0, &cond_runtime_graph);
  if (status != Ok)
    return status;

  //body_runtime_graph->selectOwnSubgraph();

  OMRuntimeContext &body_runtime_context = body_runtime_graph->getRuntimeContext();
  OMRuntimeContext &cond_runtime_context = cond_runtime_graph->getRuntimeContext();

  const auto body_input_size = body_runtime_graph->getNumberOfInputs();
  const auto body_output_size = body_runtime_graph->getNumberOfOutputs();

  status = utils::checkCondition(body_input_size == runtime_kernel.inputs_num);
  if (status != Ok)
    return status;

  status = utils::checkCondition(body_output_size == runtime_kernel.outputs_num);
  if (status != Ok)
    return status;

  status = utils::checkCondition(body_output_size == runtime_kernel.inputs_num);
  if (status != Ok)
    return status;

  const auto cond_input_size = body_runtime_graph->getNumberOfInputs();
  const auto cond_output_size = body_runtime_graph->getNumberOfOutputs();

  status = utils::checkCondition(cond_input_size == runtime_kernel.inputs_num);
  if (status != Ok)
    return status;

  status = utils::checkCondition(cond_output_size == 1);
  if (status != Ok)
    return status;

  const auto cond_output_tensor_index = cond_runtime_context.getGraphOutputTensorIndex(0);
  const circle::Tensor *cond_output_tensor = cond_runtime_context.getTensorByIndex(cond_output_tensor_index);
  status = utils::checkCondition(cond_output_tensor->type() == circle::TensorType_BOOL);
  if (status != Ok)
    return status;

  return status;
}
