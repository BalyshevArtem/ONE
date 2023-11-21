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

#include "execute/OMKernelExecutionBuilder.h"
#include "OMStatus.h"
#include "core/OMSISORuntimeKernel.h"
#include "core/OMUtils.h"
#include "PALAbs.h"

using namespace onert_micro;
using namespace onert_micro::execute;

OMStatus onert_micro::execute::execute_kernel_CircleAbs(core::OMRuntimeStorage &runtime_storage, core::OMRuntimeContext &runtime_context,
                                  core::OMKernel &kernel)
{
  core::OMSISORuntimeKernel runtime_kernel(kernel, runtime_context);

  const circle::Tensor *input = runtime_kernel.input;
  const circle::Tensor *output = runtime_kernel.output;

  assert(input != nullptr);
  assert(output != nullptr);

  OMStatus status = Ok;

  uint8_t *input_data = nullptr;
  uint8_t *output_data = nullptr;
  status = runtime_kernel.getDataFromStorage(kernel, runtime_storage, &input_data, &output_data);
  if (status != Ok)
    return status;

  assert(input_data != nullptr);
  assert(output_data != nullptr);

  const uint32_t flat_size = core::utils::numElements(input);

  switch (input->type())
  {
#ifndef DIS_FLOAT
    case circle::TensorType_FLOAT32:
      status = pal::Abs(flat_size, core::utils::castInputData<float>(input_data),
               core::utils::castOutputData<float>(output_data));
      break;
#endif // DIS_FLOAT
      default: {
      status = UnsupportedType;
      assert(false && "Unsupported type.");
    }
  }

  return status;
}