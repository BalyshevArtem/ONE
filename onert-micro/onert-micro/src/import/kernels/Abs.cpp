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
#include "core/OMSISORuntimeKernel.h"

using namespace onert_micro;
using namespace onert_micro::core;

OMStatus onert_micro::import::configure_kernel_CircleAbs(core::OMRuntimeStorage &runtime_storage, core::OMRuntimeContext &runtime_context,
                                                         core::OMKernel &kernel)
{
  OMSISORuntimeKernel runtime_kernel(kernel, runtime_context);

  const circle::Tensor *input = runtime_kernel.input;
  const circle::Tensor *output = runtime_kernel.output;

  assert(input != nullptr);
  assert(output != nullptr);

  OMStatus status = Ok;
  status = utils::checkCondition(input->type() == output->type());
  if (status != Ok)
    return status;

  const auto *input_shape = input->shape();
  const auto *output_shape = output->shape();

  status = utils::checkCondition(input_shape->size() == output_shape->size());
  if (status != Ok)
    return status;

  status = utils::checkCondition(utils::numElements(input) == utils::numElements(output));

  return status;
}
