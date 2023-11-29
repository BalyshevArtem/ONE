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

constexpr uint32_t numInput = 1;
constexpr uint32_t numOutput = 1;

constexpr uint32_t inputTensorIdx = 0;
constexpr uint32_t outputTensorIdx = 0;

} // namespace


OMStatus onert_micro::import::configure_kernel_CircleAbs(core::OMRuntimeStorage &runtime_storage, core::OMRuntimeContext &runtime_context,
                                                         core::OMKernel &kernel, const OMConfig&)
{
  onert_micro::execute::OMRuntimeKernel runtime_kernel(numInput, numOutput);

  OMStatus status = runtime_kernel.readKernel(kernel, runtime_context);
  if (status != Ok)
    return status;

  const circle::Tensor *input = runtime_kernel.inputs[inputTensorIdx];
  const circle::Tensor *output = runtime_kernel.outputs[outputTensorIdx];

  assert(input != nullptr);
  assert(output != nullptr);

  status = utils::checkCondition(input->type() == output->type());
  if (status != Ok)
    return status;

  OMShape input_shape(input);
  OMShape output_shape(output);

  status = utils::checkCondition(input_shape.rank() == output_shape.rank());
  if (status != Ok)
    return status;

  status = utils::checkCondition(input_shape.num_elements() == output_shape.num_elements());

  return status;
}
