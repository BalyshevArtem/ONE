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
#include "execute/OMUtils.h"
#include "core/OMKernelData.h"

using namespace onert_micro;
using namespace onert_micro::core;

namespace
{

constexpr uint32_t inputTensorIdx = 0;
constexpr uint32_t outputTensorIdx = 0;

} // namespace


OMStatus onert_micro::import::configure_kernel_CircleMaxPool2D(core::OMRuntimeStorage &runtime_storage, core::OMRuntimeContext &runtime_context,
                                                         core::OMKernel &kernel, const OMConfig&)
{
  onert_micro::execute::OMRuntimeKernel runtime_kernel;

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

  status = utils::checkCondition(input_shape.rank() == 4);

  auto option = runtime_kernel.first_operator->builtin_options_as_Pool2DOptions();

  if (option == nullptr)
    return UnknownError;

  assert(option != nullptr);

  int32_t padding_h = 0;
  int32_t padding_w = 0;

  const int input_width = input_shape.dim(2);
  const int input_height = input_shape.dim(1);
  execute::computePaddingHeightWidth(option->stride_h(), option->stride_w(), 1 /* dilation_rate_height */,
                                     1 /* dilation_rate_width */, input_height, input_width, option->filter_height(), option->filter_width(),
                                     option->padding(), &padding_h, &padding_w);

  DataMaxPool2D *data = new DataMaxPool2D;

  data->padding_w = padding_w;
  data->padding_h = padding_h;

  kernel.setKernelData(reinterpret_cast<uint8_t *>(data));

  if (input->type() != circle::TensorType_INT8 and input->type() != circle::TensorType_INT16)
    return status;

  // Check quantization params
  if (input->quantization() == nullptr)
  {
    return NoQuantization;
  }

  if (input->quantization()->scale()->size() != 1)
  {
    return UnsupportedType;
  }

  return status;
}
