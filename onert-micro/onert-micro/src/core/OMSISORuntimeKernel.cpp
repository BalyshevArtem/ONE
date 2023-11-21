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

#include "core/OMSISORuntimeKernel.h"

using namespace onert_micro::core;
using namespace onert_micro;

onert_micro::core::OMSISORuntimeKernel::OMSISORuntimeKernel(core::OMKernel &kernel, core::OMRuntimeContext &runtime_context)
{
  const std::vector<uint16_t> kernel_operators = kernel.getKernelOperators();

  const uint16_t first_operator_index = kernel_operators.front();
  const uint16_t last_operator_index = kernel_operators.back();

  const circle::Operator *first_operator = runtime_context.getCircleOperatorAt(first_operator_index);
  const circle::Operator *last_operator = runtime_context.getCircleOperatorAt(last_operator_index);

  input_index = first_operator->inputs()->operator[](0);
  output_index = last_operator->outputs()->operator[](0);

  assert(input_index != -1);
  assert(output_index != -1);

  input = runtime_context.getTensorByIndex(input_index);
  output = runtime_context.getTensorByIndex(output_index);
}

OMStatus onert_micro::core::OMSISORuntimeKernel::getDataFromStorage(core::OMKernel &kernel,
                                                                    core::OMRuntimeStorage &storage,
                                                                    uint8_t **input_data,
                                                                    uint8_t **output_data)
{
  OMStatus status = Ok;

  status = storage.getDataByTensorIndex(input_data, input_index);
  if (status != Ok)
    return status;

  if (kernel.getKernelType() != Inplace)
  {
    status = storage.getDataByTensorIndex(output_data, output_index);
  } else
  {
    *output_data = *input_data;
    storage.removeTensorFromTensorIndexToData(input_index);
    storage.saveDataToTensorIndex(*output_data, output_index);
  }

  return status;
}
