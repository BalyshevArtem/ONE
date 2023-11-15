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

#include "optimize/OMOptimizePassesBuilder.h"
#include "OMStatus.h"
#include "OMConfig.h"
#include "core/OMRuntimeStorage.h"
#include "core/OMRuntimeStorage.h"
#include "core/OMKernelType.h"
#include "core/reader/OMCircleReader.h"

using namespace onert_micro;

namespace
{
OMStatus isInplaceOperation(core::OMBuilderID op, bool &is_inplace)
{
  OMStatus status = Ok;
  circle::BuiltinOperator builtin_operator;
  status = core::getBuiltinOperatorByBuilderId(op, builtin_operator);

  if (status != Ok)
    return status;

  switch (builtin_operator)
  {
    case circle::BuiltinOperator_ABS:
    case circle::BuiltinOperator_CEIL:
    case circle::BuiltinOperator_LOGISTIC:
    case circle::BuiltinOperator_RESHAPE:
    case circle::BuiltinOperator_ELU:
    case circle::BuiltinOperator_EXPAND_DIMS:
    case circle::BuiltinOperator_EXP:
    case circle::BuiltinOperator_TANH:
    case circle::BuiltinOperator_LEAKY_RELU:
    case circle::BuiltinOperator_LOG:
    case circle::BuiltinOperator_RELU:
    case circle::BuiltinOperator_RELU6:
    case circle::BuiltinOperator_ROUND:
    case circle::BuiltinOperator_ADD:
    case circle::BuiltinOperator_MUL:
    case circle::BuiltinOperator_SUB:
    case circle::BuiltinOperator_WHILE:
    case circle::BuiltinOperator_ZEROS_LIKE: {
      is_inplace = true;
      break;
    }
    case circle::BuiltinOperator_CUSTOM:
    {
      core::OMBuilderCustomID custom_id;
      status = core::getCustomOperatorByBuilderId(op, custom_id);

      if (status != Ok)
        return status;
      switch (custom_id)
      {
        case onert_micro::core::custom_gru:
          is_inplace = true;
          break;
        default:
          is_inplace = false;
      }
    }
    default:
      is_inplace = false;
  }
  return status;
}

bool isSingleUsageOfTensor(core::reader::OMCircleReader &reader, const int32_t tensor_index)
{
  uint32_t usage_count = 0;

  const auto operators = reader.operators();
  for (uint32_t i = 0; i < operators->size(); ++i)
  {
    const auto *op = operators->operator[](i);
    assert(op != nullptr);

    const auto *op_inputs = op->inputs();
    for (int32_t j = 0; j < op_inputs->size(); ++j)
    {
      const auto input_index = op_inputs->operator[](j);
      if (input_index == tensor_index)
      {
        if (++usage_count > 1)
          return false;
      }
    }
  }

  // Let's check that it is not graph output
  if (usage_count == 1)
  {
    const auto &outputs_indexes = reader.outputs();
    bool is_graph_output = (std::find(outputs_indexes->begin(), outputs_indexes->end(),
                                      tensor_index) != outputs_indexes->end());
    if (is_graph_output)
      return false;
  }

  return true;
}

OMStatus checkInplaceOp(core::reader::OMCircleReader &reader,  uint16_t input_operator_index,
                        uint16_t output_operator_index, bool &is_inplace)
{
  const auto operators = reader.operators();
  const auto graph_outputs = reader.outputs();
  const auto *input_op = operators->operator[](input_operator_index);
  const auto *output_op = operators->operator[](output_operator_index);
  const auto *op_inputs = input_op->inputs();
  const auto *op_outputs = output_op->outputs();

  auto non_const_input_it = op_inputs->begin();
  while (non_const_input_it != op_inputs->end())
  {
    auto dist = std::distance(op_inputs->begin(), non_const_input_it);

    const auto non_const_input_idx = *non_const_input_it;

    if (reader.isConstTensor(non_const_input_idx))
    {
      non_const_input_it++;
      continue;
    }

    // Check single usage of input tensor
    if (not isSingleUsageOfTensor(reader, non_const_input_idx))
    {
      is_inplace = false;
      break;
    }

    // Let's check single usage of output tensor
    if (dist >= op_outputs->size() and op_outputs->size() == 1)
      dist = 0;

    assert(dist < op_outputs->size());
    if (dist >= op_outputs->size())
      return UnknownError;

    const auto output_index = op_outputs->operator[](dist);
    if (not isSingleUsageOfTensor(reader, output_index))
    {
      is_inplace = false;
      break;
    }

    // Check that num elements are equal
    {
      const auto *input_non_const_tensor = reader.getTensorByIndex(non_const_input_idx);
      const auto *output_tensor = reader.getTensorByIndex(output_index);
      if (input_non_const_tensor->shape()->size() != output_tensor->shape()->size())
      {
        is_inplace = false;
        break;
      }
    }

    // Let's check that output is not a graph output tensor
    // TODO: check this statement
    {
      if (std::find(graph_outputs->begin(), graph_outputs->end(), output_index) !=
          graph_outputs->end())
      {
        is_inplace = false;
        break;
      }
    }

    non_const_input_it++;
  }

  return Ok;
}

OMStatus findInplaceOp(std::vector<core::OMKernel> &kernels, core::OMRuntimeContext &context, const OMConfig &configs, bool &is_changed)
{
  OMStatus status;
  core::reader::OMCircleReader &reader = context.getCircleReader();

  for (uint32_t i = 0; i < kernels.size(); ++i)
  {
    core::OMKernel &cur_kernel = kernels.at(i);
    bool is_inplace = false;
    status = isInplaceOperation(cur_kernel.getBuilderID(), is_inplace);
    if (status != Ok)
      return status;

    if (is_inplace == false)
      continue;

    uint16_t input_operator_index = cur_kernel.getKernelOperators().front();
    uint16_t output_operator_index = cur_kernel.getKernelOperators().back();

    is_inplace = true;

    status = checkInplaceOp(reader, input_operator_index, output_operator_index, is_inplace);

    if (status != Ok)
      return status;

    if (not is_inplace)
      continue;

    is_changed = true;
    cur_kernel.setKernelType(onert_micro::core::Inplace);
  }

  return status;
}

} // namespace

optimize::OMGraphStatus optimize::onert_micro_FindInplaceOpPass(core::OMRuntimeStorage &storage, core::OMRuntimeContext &context, const OMConfig &configs)
{
  bool changed = false;

  std::vector<core::OMKernel> &kernels = storage.getKernels();

  OMGraphStatus graph_status = {Unchanged, Ok};
  do
  {
    changed = false;
    graph_status.main_status = findInplaceOp(kernels, context, configs, changed);

    if (graph_status.main_status != Ok)
      return graph_status;

    if (changed)
      graph_status.graph_status = Changed;

  } while (changed);

  return graph_status;
}

