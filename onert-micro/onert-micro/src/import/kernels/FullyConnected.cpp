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

#ifndef DIS_QUANT
void calculateOpDataFullyConnected(core::OMKernel &kernel, const circle::Tensor *input,
                                   const circle::Tensor *weights, const circle::Tensor *output,
                                   circle::ActivationFunctionType activation)
{
  double real_multiplier = 0.0;
  int output_shift;
  int32_t output_activation_min;
  int32_t output_activation_max;
  int32_t output_multiplier;

  assert(input->quantization() != nullptr);                 // Fix caller
  assert(input->quantization()->scale()->size() == 1);      // Fix caller
  assert(input->quantization()->zero_point()->size() == 1); // Fix caller

  assert(weights->quantization() != nullptr);                 // Fix caller
  assert(weights->quantization()->scale()->size() == 1);      // Fix caller
  assert(weights->quantization()->zero_point()->size() == 1); // Fix caller

  assert(output->quantization() != nullptr);                 // Fix caller
  assert(output->quantization()->scale()->size() == 1);      // Fix caller
  assert(output->quantization()->zero_point()->size() == 1); // Fix caller

  const float input_scale = input->quantization()->scale()->operator[](0);
  const float weight_scale = weights->quantization()->scale()->operator[](0);
  const float output_scale = output->quantization()->scale()->operator[](0);

  const float input_zero_point = input->quantization()->zero_point()->operator[](0);
  const float weights_zero_point = weights->quantization()->zero_point()->operator[](0);
  const float output_zero_point = output->quantization()->zero_point()->operator[](0);

  real_multiplier =
    execute::getQuantizedConvolutionMultipler(input_scale, weight_scale, output_scale);
  execute::quantizeMultiplier(real_multiplier, &output_multiplier, &output_shift);
  execute::calculateActivationRangeQuantized(activation, output_zero_point, output_scale,
                                             output->type(), &output_activation_min,
                                             &output_activation_max);

  int32_t input_offset = -input_zero_point;
  int32_t filter_offset = 0;
  filter_offset = -weights_zero_point;
  int32_t output_offset = output_zero_point;

  DataFullyConnected *op_params = new DataFullyConnected;
  op_params->input_offset = input_offset;
  op_params->weights_offset = filter_offset;
  op_params->output_offset = output_offset;
  op_params->output_multiplier = output_multiplier;
  op_params->output_shift = output_shift;
  op_params->quantized_activation_min = output_activation_min;
  op_params->quantized_activation_max = output_activation_max;
  op_params->lhs_cacheable = false;
  op_params->rhs_cacheable = false;

  kernel.setKernelData(reinterpret_cast<uint8_t *>(op_params));
}
#endif

} // namespace

OMStatus onert_micro::import::configure_kernel_CircleFullyConnected(core::OMRuntimeStorage &runtime_storage, core::OMRuntimeContext &runtime_context,
                                                                    core::OMKernel &kernel)
{
  execute::OMRuntimeKernel runtime_kernel(numInput, numOutput);
  runtime_kernel.readKernel(kernel, runtime_context);

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

  // TODO: add precalculations for quantized values;
  // TODO check scales

  return status;
}
