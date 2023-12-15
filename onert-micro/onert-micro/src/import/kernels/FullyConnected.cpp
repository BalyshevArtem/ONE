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
#include "execute/OMUtils.h"
#include "core/OMShape.h"
#include "core/OMKernelData.h"
#include "OMStatus.h"
#include "execute/OMRuntimeKernel.h"

using namespace onert_micro;
using namespace onert_micro::core;

namespace
{

constexpr uint32_t numInput = 3;
constexpr uint32_t numOutput = 1;

constexpr uint32_t inputTensorIdx = 0;
constexpr uint32_t weightTensorIdx = 1;
constexpr uint32_t biasTensorIdx = 2;

constexpr uint32_t outputTensorIdx = 0;

} // namespace

OMStatus onert_micro::import::configure_kernel_CircleFullyConnected(core::OMRuntimeStorage &runtime_storage, core::OMRuntimeContext &runtime_context,
                                                                    uint16_t op_index, const OMConfig &configs)
{
  execute::OMRuntimeKernel runtime_kernel;
  runtime_kernel.readKernel(op_index, runtime_context);

  const circle::Tensor *input = runtime_kernel.inputs[inputTensorIdx];
  const circle::Tensor *weight = runtime_kernel.inputs[weightTensorIdx];
  const circle::Tensor *bias = runtime_kernel.inputs[biasTensorIdx];

  const circle::Tensor *output = runtime_kernel.outputs[outputTensorIdx];

  assert(input != nullptr);
  assert(weight != nullptr);
  // Bias can be nullptr
  assert(output != nullptr);

  OMStatus status = Ok;

  if ((input->type() == circle::TensorType_FLOAT32 && weight->type() != circle::TensorType_FLOAT32) or
    (input->type() == circle::TensorType_INT8 && weight->type() != circle::TensorType_INT8) or
    (input->type() == circle::TensorType_INT16 && weight->type() != circle::TensorType_INT16))
  {
    return UnsupportedType;
  }

  core::OMShape weight_shape(weight);
  core::OMShape bias_shape(bias);

  status = utils::checkCondition(weight_shape.rank() == 2);
  if (status != Ok)
    return status;

  status = utils::checkCondition(bias == nullptr or weight_shape.dim(0) == bias_shape.num_elements());

  if (input->type() == circle::TensorType_FLOAT32)
    return status;

#ifndef DIS_QUANT

  // Check quantized version
  if (input->quantization() == nullptr or output->quantization() == nullptr or weight->quantization() == nullptr)
    return NoQuantization;

  if (output->quantization()->scale() == nullptr or output->quantization()->scale()->size() != 1)
    return NoQuantization;

  if (weight->quantization()->scale() == nullptr or weight->quantization()->scale()->size() != 1)
    return NoQuantization;

#endif // DIS_QUANT
  return status;
}
