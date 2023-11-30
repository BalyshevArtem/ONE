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
#include "core/OMDataType.h"

using namespace onert_micro;
using namespace onert_micro::core;

namespace
{

constexpr uint32_t numOutput = 1;

} // namespace


OMStatus onert_micro::import::configure_kernel_CircleConcatenation(core::OMRuntimeStorage &runtime_storage, core::OMRuntimeContext &runtime_context,
                                                                  core::OMKernel &kernel, const OMConfig&)
{
  const auto cur_op_index = kernel.getKernelOperators().front();
  const auto cur_op = runtime_context.getCircleOperatorAt(cur_op_index);

  if (cur_op == nullptr)
    return UnknownError;

  const int num_inputs = cur_op->inputs()->size();
  assert(num_inputs > 0);

  auto input_index = cur_op->inputs()->operator[](0);
  auto output_index = cur_op->outputs()->operator[](0);

  assert(input_index != -1);
  assert(output_index != -1);

  const auto *t0 = runtime_context.getTensorByIndex(input_index);
  const auto *output = runtime_context.getTensorByIndex(output_index);

  const auto *params = cur_op->builtin_options_as_ConcatenationOptions();

  // TODO: Support concat with fused activation function
  if (params->fused_activation_function() != circle::ActivationFunctionType_NONE)
    return UnknownError;

  OMShape input_shape(t0);
  int axis = params->axis();
  if (axis < 0)
    axis += input_shape.rank();

  if (axis < 0 or axis > input_shape.rank())
    return FailedCheckCondition;

  for (int i = 1; i < num_inputs; ++i)
  {
    input_index = cur_op->inputs()->operator[](i);
    const auto *tensor = runtime_context.getTensorByIndex(input_index);
    if (tensor->type() != t0->type())
      return FailedCheckCondition;

    OMShape cur_shape(tensor);

    if (cur_shape.num_elements() != cur_shape.num_elements())
      return FailedCheckCondition;
  }

  if (t0->type() != circle::TensorType_INT8 and t0->type() != circle::TensorType_INT16)
    return Ok;

  #ifndef DIS_QUANT
  // If input tensors are INT8 type then quantization parameters of all input tensors and the output
  // should be the same
  for (int i = 0; i < num_inputs; ++i)
  {
    input_index = cur_op->inputs()->operator[](i);
    const auto *tensor = runtime_context.getTensorByIndex(input_index);

    if (tensor->quantization() == nullptr)
      return FailedCheckCondition;

    if (tensor->quantization()->scale()->size() != 1)
      return FailedCheckCondition;

    if (tensor->quantization()->zero_point()->size() != 1)
      return FailedCheckCondition;

    if (*tensor->quantization()->scale()->begin() != *output->quantization()->scale()->begin())
      return FailedCheckCondition;

    if (*tensor->quantization()->zero_point()->begin() != *output->quantization()->zero_point()->begin())
      return FailedCheckCondition;
  }
#endif // DIS_QUANT

  return Ok;
}
