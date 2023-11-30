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

#include "execute/OMUtils.h"
#include "execute/OMKernelExecutionBuilder.h"
#include "OMStatus.h"
#include "execute/OMRuntimeKernel.h"
#include "core/OMUtils.h"
#include "core/OMShape.h"
#include "core/OMRuntimeShape.h"
#include "PALAdd.h"

using namespace onert_micro;
using namespace onert_micro::execute;

namespace
{

constexpr uint32_t numOutput = 1;

template <typename T>
OMStatus evalGeneric(const circle::Operator *cur_op, core::OMRuntimeStorage &runtime_storage, core::OMRuntimeContext &runtime_context,)
{
  const auto output_index = cur_op->outputs()->operator[](0);

  assert(output_index != -1);

  auto output = runtime_context.getTensorByIndex(output_index);

  const auto *options = cur_op->builtin_options_as_ConcatenationOptions();

  core::OMRuntimeShape output_shape(output);

  int axis = options->axis();
  if (axis < 0)
    axis += output_shape.dimensionsCount();

  const auto input_sizes = cur_op->inputs()->size();

  std::vector<const T *> all_input_data;
  std::vector<core::OMRuntimeShape> all_shape;
  std::vector<core::OMRuntimeShape *> all_shape_ptr;

  OMStatus status = Ok;
  for (int32_t i = 0; i < input_sizes; ++i)
  {
    auto input_index = cur_op->inputs()->operator[](i);
    const auto *tensor = runtime_context.getTensorByIndex(input_index);

    const uint8_t *tensor_data = nullptr;
    status = runtime_storage.getDataByTensorIndex(&tensor_data, input_index);
    if (tensor_data == nullptr)
      status = runtime_context.getConstDataByTensorIndex(&tensor_data, tensor);

    if (status != Ok)
      return onert_micro::UnknownError;

    auto *data = reinterpret_cast<const T *>(tensor_data);

    all_input_data.push_back(data);
    all_shape.emplace_back(output);
  }

  for (core::OMRuntimeShape &shape : all_shape)
  {
    all_shape_ptr.push_back(&shape);
  }

  uint8_t *output_data = nullptr;
  status = runtime_storage.getDataByTensorIndex(&output_data, output);

  luci_interpreter_pal::ConcatenationParams params{};
  params.axis = axis;
  params.inputs_count = all_shape.size();
  luci_interpreter_pal::Concatenation(params, all_shape_ptr.data(), all_input_data.data(),
                                      kernels::getTensorShape(output), output_data);
}

} // namespace

} // namespace


OMStatus onert_micro::execute::configure_kernel_CircleConcatenation(core::OMRuntimeStorage &runtime_storage, core::OMRuntimeContext &runtime_context,
                                                                   core::OMKernel &kernel)
{
  const auto cur_op_index = kernel.getKernelOperators().front();
  const auto cur_op = runtime_context.getCircleOperatorAt(cur_op_index);

  if (cur_op == nullptr)
    return UnknownError;

  const int num_inputs = cur_op->inputs()->size();
  assert(num_inputs > 0);

  auto input_index = cur_op->inputs()->operator[](0);

  assert(input_index != -1);

  const auto *t0 = runtime_context.getTensorByIndex(input_index);
  OMStatus status = Ok;

  switch (t0->type())
  {
#ifndef DIS_FLOAT
    case circle::TensorType_FLOAT32:
      status = evalGeneric<float>(cur_op, runtime_context, runtime_storage);
      break;
#endif // DIS_FLOAT
#ifndef DIS_QUANT
    case circle::TensorType_INT8:
      status = evalGeneric<int8_t>(cur_op, runtime_context, runtime_storage);
      break;
#endif // DIS_QUANT
    case circle::TensorType_INT32:
      status = evalGeneric<int32_t>(cur_op, runtime_context, runtime_storage);
      break;
    case circle::TensorType_INT64:
      status = evalGeneric<int64_t>(cur_op, runtime_context, runtime_storage);
      break;
    default:
      assert(false && "Unsupported type.");
  }

  return status;
}
