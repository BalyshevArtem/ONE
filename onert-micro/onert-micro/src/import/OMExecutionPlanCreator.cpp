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

#include "import/OMExecutionPlanCreator.h"

#include <map>

using namespace onert_micro::core;
using namespace onert_micro::import;
using namespace onert_micro;

OMStatus OMExecutionPlanCreator::createExecutionPlan(core::OMRuntimeStorage &runtime_storage,
                                                     core::OMRuntimeContext &runtime_context,
                                                     core::memory::OMRuntimeAllocator &allocator,
                                                     const OMConfig &configs)
{
  bool keep_input = configs.keep_input;
  reader::OMCircleReader &reader = runtime_context.getCircleReader();

  std::vector<std::vector<uint16_t>> &alloc_plan = allocator.getAllocPlan();
  std::vector<std::vector<uint16_t>> &dealloc_plan = allocator.getDeallocPlan();

  using Lifetime = std::pair<int32_t, int32_t>;

  std::map<uint16_t , Lifetime> lifetimes;

  std::vector<OMKernel> &kernels = runtime_storage.getKernels();

  const size_t num_kernels = kernels.size();

  if (not keep_input)
  {
    for (const auto input_ind : *reader.inputs())
    {
      assert(lifetimes.count(input_ind) == 0);
      lifetimes[input_ind] = Lifetime(-1, 0);
    }
  }

  for (int32_t index = 0; index < num_kernels; ++index)
  {
    auto &kernel = kernels.at(index);

    uint16_t input_operator_index = kernel.getKernelOperators().front();
    uint16_t output_operator_index = kernel.getKernelOperators().back();

    const auto operators = reader.operators();
    const auto *input_op = operators->operator[](input_operator_index);
    const auto *output_op = operators->operator[](output_operator_index);
    const auto *op_inputs = input_op->inputs();
    const auto *op_outputs = output_op->outputs();

    for (int32_t j = 0; j < op_inputs->size(); ++j)
    {
      const auto input_index = op_inputs->operator[](j);

      if (input_index == -1)
        continue;

      // Pass constant tensors
      if (reader.isConstTensor(input_index))
        continue;

      if (lifetimes.count(input_index) > 0)
      {
        if (kernel.getKernelType() == Inplace)
          lifetimes.at(input_index).second = -1;
        else
          lifetimes.at(input_index).second = index;
      }
    }

    for (int32_t j = 0; j < op_outputs->size(); ++j)
    {
      const auto output_index = op_outputs->operator[](j);

      if (kernel.getKernelType() == Inplace)
        lifetimes[output_index] = Lifetime(-1, index);
      else
        lifetimes[output_index] = Lifetime(index, index);
    }
  }

  for (const auto output_ind : *reader.outputs())
  {
    if (lifetimes.count(output_ind) > 0)
      lifetimes.at(output_ind).second = static_cast<int32_t>(num_kernels);
  }

  alloc_plan.assign(num_kernels, std::vector<uint16_t>());
  dealloc_plan.assign(num_kernels + 1, std::vector<uint16_t>());

  for (const auto &item : lifetimes)
  {
    if (item.second.first != -1)
      alloc_plan[item.second.first].push_back(item.first);
    if (item.second.second != -1)
      dealloc_plan[item.second.second].push_back(item.first);
  }

  return Ok;
}
