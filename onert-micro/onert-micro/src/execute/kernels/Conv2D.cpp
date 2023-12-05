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
#include "OMStatus.h"
#include "execute/OMRuntimeKernel.h"

#include "PALConv2d.h"

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


// NOTE: doesn't currently support dynamic shapes
OMStatus onert_micro::execute::execute_kernel_CircleConv2D(core::OMRuntimeStorage &runtime_storage,
                                                           core::OMRuntimeContext &runtime_context,
                                                           core::OMKernel &kernel)
{
  const circle::Tensor *input;
  const circle::Tensor *weight;
  const circle::Tensor *output;

  uint8_t *input_data;
  uint8_t *weight_data;
  uint8_t *bias_data;
  uint8_t *output_data;

  const circle::Conv2DOptions *options;
  // Read kernel
  {
    execute::OMRuntimeKernel runtime_kernel;
    OMStatus status = runtime_kernel.readKernel(kernel, runtime_context);
    if (status != Ok)
      return status;

    input = runtime_kernel.inputs[inputTensorIdx];
    weight = runtime_kernel.inputs[weightTensorIdx];
    output = runtime_kernel.outputs[outputTensorIdx];
    assert(input != nullptr);
    assert(weight != nullptr);
    // Bias can be nullptr
    assert(output != nullptr);

    status = runtime_kernel.getDataFromStorage(kernel, runtime_storage, runtime_context);
    if (status != Ok)
      return status;

    input_data = runtime_kernel.inputs_data[inputTensorIdx];
    weight_data = runtime_kernel.inputs_data[weightTensorIdx];
    bias_data = runtime_kernel.inputs_data[biasTensorIdx];
    output_data = runtime_kernel.outputs_data[outputTensorIdx];
    assert(input_data != nullptr);
    assert(weight_data != nullptr);
    // Bias can be nullptr
    assert(output_data != nullptr);

    options = runtime_kernel.first_operator->builtin_options_as_Conv2DOptions();
  }

  OMStatus status;

  switch (input->type())
  {
#ifndef DIS_FLOAT
    case circle::TensorType_FLOAT32: {

      DataConv2D *data = reinterpret_cast<DataConv2D *>(kernel.getKernelData());
      assert(data != nullptr);
      if (data == nullptr)
        return UnknownError;

      FloatConv2D params{};
      status = calculateActivationRange(options->fused_activation_function(),
                                        &params.activation_min, &params.activation_max);
      params.stride_w = options->stride_w();
      params.stride_h = options->stride_h();
      params.dilation_width_factor = options->dilation_w_factor();
      params.dilation_height_factor = options->dilation_h_factor();
      params.pad_h = data->padding_h;
      params.pad_w = data->padding_w;

      if (status != Ok)
        return status;

      status = pal::ConvFloat(&params, OMRuntimeShape(input),
                              core::utils::castInputData<float>(input_data), OMRuntimeShape(weight),
                              core::utils::castInputData<float>(weight_data),
                              core::utils::castInputData<float>(bias_data), OMRuntimeShape(output),
                              core::utils::castOutputData<float>(output_data));
      assert(status == Ok);
    }
    break;
#endif // DIS_FLOAT
    default: {
      status = UnsupportedActivation;
      assert(false && "Unsupported type.");
    }
  }

  return status;
}
