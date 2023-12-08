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
#include "execute/OMUtils.h"
#include "core/OMUtils.h"
#include "core/OMKernelData.h"
#include "core/OMShape.h"
#include "OMStatus.h"
#include "execute/OMRuntimeKernel.h"

#include "PALFullyConnected.h"

using namespace onert_micro;
using namespace onert_micro::core;
using namespace onert_micro::execute;

namespace
{

constexpr uint32_t numInput = 3;
constexpr uint32_t numOutput = 1;

constexpr uint32_t inputTensorIdx = 0;
constexpr uint32_t weightTensorIdx = 1;
constexpr uint32_t biasTensorIdx = 2;

constexpr uint32_t outputTensorIdx = 0;

} // namespace


// NOTE: doesnt currently support dynamic shapes
OMStatus onert_micro::execute::execute_kernel_CircleFullyConnected(core::OMRuntimeStorage &runtime_storage, core::OMRuntimeContext &runtime_context,
                                                                   core::OMKernel &kernel)
{
  const circle::Tensor *input;
  const circle::Tensor *weight;
  const circle::Tensor *output;

  uint8_t *input_data;
  uint8_t *weight_data;
  uint8_t *bias_data;
  uint8_t *output_data;

  const circle::FullyConnectedOptions *options;
  // Read kernel
  {
    execute::OMRuntimeKernel runtime_kernel;
    runtime_kernel.readKernel(kernel, runtime_context);

    input = runtime_kernel.inputs[inputTensorIdx];
    weight = runtime_kernel.inputs[weightTensorIdx];
    output = runtime_kernel.outputs[outputTensorIdx];
    assert(input != nullptr);
    assert(weight != nullptr);
    // Bias can be nullptr
    assert(output != nullptr);

    runtime_kernel.getDataFromStorage(kernel, runtime_storage, runtime_context);

    input_data = runtime_kernel.inputs_data[inputTensorIdx];
    weight_data = runtime_kernel.inputs_data[weightTensorIdx];
    bias_data = runtime_kernel.inputs_data[biasTensorIdx];
    output_data = runtime_kernel.outputs_data[outputTensorIdx];
    assert(input_data != nullptr);
    assert(weight_data != nullptr);
    // Bias can be nullptr
    assert(output_data != nullptr);

    options = runtime_kernel.first_operator->builtin_options_as_FullyConnectedOptions();
  }

  OMStatus status;

  switch (input->type())
  {
#ifndef DIS_FLOAT
    case circle::TensorType_FLOAT32: {
      FullyConnectedParams params{};
      status = calculateActivationRange(options->fused_activation_function(),
                                        &params.float_activation_min, &params.float_activation_max);
      if (status != Ok)
        return status;

      status = pal::FullyConnected(
        &params, core::utils::castInputData<float>(input_data), OMRuntimeShape(weight),
        core::utils::castInputData<float>(weight_data),
        core::utils::castInputData<float>(bias_data), OMRuntimeShape(output),
        core::utils::castOutputData<float>(output_data));
    }
    break;
#endif // DIS_FLOAT
#ifndef DIS_QUANT
    case circle::TensorType_INT8: {
      FullyConnectedParams op_params{};
      op_params.input_offset = *input->quantization()->zero_point()->begin();
      op_params.weights_offset = *weight->quantization()->zero_point()->begin();
      op_params.output_offset = *output->quantization()->zero_point()->begin();

      op_params.input_scale = *input->quantization()->scale()->begin();
      op_params.weight_scale = *weight->quantization()->scale()->begin();
      op_params.output_scale = *output->quantization()->scale()->begin();

      execute::calculateActivationRangeQuantized(options->fused_activation_function(), op_params.output_offset, op_params.output_scale,
                                                 output->type(), &op_params.quantized_activation_min,
                                                 &op_params.quantized_activation_max);

      DataFullyConnected *fc_data = reinterpret_cast<DataFullyConnected *>(kernel.getKernelData());
      if (fc_data != nullptr)
      {
        op_params.output_multiplier = fc_data->output_multiplier;
        op_params.output_shift = fc_data->output_shift;
      }

     status = pal::FullyConnected(&op_params, core::utils::castInputData<int8_t>(input_data),
                                  OMRuntimeShape(weight), core::utils::castInputData<int8_t>(weight_data),
                                  core::utils::castInputData<int32_t>(bias_data), OMRuntimeShape(output),
                                  core::utils::castOutputData<int8_t>(output_data));
    }
    break;
#endif // DIS_QUANT
    default: {
      status = UnsupportedActivation;
      assert(false && "Unsupported type.");
    }
  }

  return status;
}
