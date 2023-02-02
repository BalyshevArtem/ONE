/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Builders.h"

#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/reference/integer_ops/pooling.h>
#include <tensorflow/lite/kernels/internal/reference/pooling.h>

namespace luci_interpreter
{


namespace
{
using namespace kernels;
void evalFloat(const Tensor *input, Tensor *output,
               const circle::Pool2DOptions *options)
{
  const int32_t batches = input->dim(0);
  const int32_t input_height = input->dim(1);
  const int32_t input_width = input->dim(2);
  const int32_t depth = input->dim(3);

  const int32_t output_height =
    kernels::computeOutputSize(static_cast<Padding>(options->padding()), input_height, options->filter_height(), options->stride_h());
  const int32_t output_width =
    kernels::computeOutputSize(static_cast<Padding>(options->padding()), input_width, options->filter_width(), options->stride_w());

  const auto padding_height =
    kernels::computePadding(options->stride_h(), 1, input_height, options->filter_height(), output_height);
  const auto padding_width =
    kernels::computePadding(options->stride_w(), 1, input_width, options->filter_width(), output_width);



  float activation_min{};
  float activation_max{};
  kernels::calculateActivationRange(luci_actfunc(options->fused_activation_function()), &activation_min, &activation_max);
  tflite::PoolParams params{};
  params.padding_values.height = padding_height;
  params.padding_values.width = padding_width;
  params.stride_height = options->stride_h();
  params.stride_width = options->stride_w();
  params.filter_height = options->filter_height();
  params.filter_width = options->filter_width();
  params.float_activation_min = activation_min;
  params.float_activation_max = activation_max;

  tflite::reference_ops::MaxPool(params, kernels::getTensorShape(input), kernels::getTensorData<float>(input),
                                 kernels::getTensorShape(output), kernels::getTensorData<float>(output));
}

} // namespace

void configure_kernel_CircleMaxPool2D(const circle::Operator *cur_op,
                                    IBaseRuntimeGraph *runtime_graph)
{
  const auto input_index = cur_op->inputs()->operator[](0);
  const auto output_index = cur_op->outputs()->operator[](0);

  assert(input_index != -1);
  assert(output_index != -1);

  const auto input = runtime_graph->getTensorByIndex(input_index);
  auto output = runtime_graph->getTensorByIndex(output_index);

  //const auto *options = cur_op->builtin_options_as_Pool2DOptions();

  LUCI_INTERPRETER_CHECK(input->element_type() == output->element_type());
//  assert(input->shape().num_dims() == 4);
//  const int32_t batches = input->dim(0);
//  const int32_t input_height = input->dim(1);
//  const int32_t input_width = input->dim(2);
//  const int32_t depth = input->dim(3);

//  const int32_t output_height =
//    computeOutputSize(options->padding(), input_height, options->filter_height(), options->stride_h());
//  const int32_t output_width =
//    computeOutputSize(options->padding(), input_width, options->filter_width(), options.stride_w());
//
//  _padding_height =
//    computePadding(_params.stride_height, 1, input_height, _params.filter_height, output_height);
//  _padding_width =
//    computePadding(_params.stride_width, 1, input_width, _params.filter_width, output_width);

//  // TODO: enable it only if kernel with dynamic shapes
//  //output()->resize({batches, output_height, output_width, depth});
//  if (input()->element_type() == DataType::U8)
//  {
//    LUCI_INTERPRETER_CHECK(std::abs(output()->scale() - input()->scale()) <= 1.0e-6);
//    LUCI_INTERPRETER_CHECK(output()->zero_point() == input()->zero_point());
//  }
//  else if (input()->element_type() == DataType::S16)
//  {
//    LUCI_INTERPRETER_CHECK(std::abs(output()->scale() - input()->scale()) <= 1.0e-6);
//    LUCI_INTERPRETER_CHECK(input()->zero_point() == 0 && output()->zero_point() == 0);
//  }
}

void execute_kernel_CircleMaxPool2D(const circle::Operator *cur_op,
                                  IBaseRuntimeGraph *runtime_graph,
                                  bool is_inplace)
{
  const auto input_index = cur_op->inputs()->operator[](0);
  const auto output_index = cur_op->outputs()->operator[](0);

  assert(input_index != -1);
  assert(output_index != -1);

  const auto input = runtime_graph->getTensorByIndex(input_index);
  auto output = runtime_graph->getTensorByIndex(output_index);

  const auto *options = cur_op->builtin_options_as_Pool2DOptions();

  switch (input->element_type())
  {
    case DataType::FLOAT32:
      evalFloat(input, output, options);
      break;
//    case DataType::U8:
//      evalQuantized();
//      break;
//    case DataType::S16:
//      evalSInt16();
//      break;
    default:
      assert(false && "Unsupported type.");
  }
}

//MaxPool2D::MaxPool2D(const Tensor *input, Tensor *output, const Pool2DParams &params)
//  : KernelWithParams<Pool2DParams>({input}, {output}, params)
//{
//}

//void MaxPool2D::configure()
//{
//  LUCI_INTERPRETER_CHECK(input()->element_type() == output()->element_type());
//  assert(input()->shape().num_dims() == 4);
//  const Shape &input_shape = input()->shape();
//  const int32_t batches = input_shape.dim(0);
//  const int32_t input_height = input_shape.dim(1);
//  const int32_t input_width = input_shape.dim(2);
//  const int32_t depth = input_shape.dim(3);
//
//  const int32_t output_height =
//    computeOutputSize(_params.padding, input_height, _params.filter_height, _params.stride_height);
//  const int32_t output_width =
//    computeOutputSize(_params.padding, input_width, _params.filter_width, _params.stride_width);
//
//  _padding_height =
//    computePadding(_params.stride_height, 1, input_height, _params.filter_height, output_height);
//  _padding_width =
//    computePadding(_params.stride_width, 1, input_width, _params.filter_width, output_width);
//
//  // TODO: enable it only if kernel with dynamic shapes
//  output()->resize({batches, output_height, output_width, depth});
//  if (input()->element_type() == DataType::U8)
//  {
//    LUCI_INTERPRETER_CHECK(std::abs(output()->scale() - input()->scale()) <= 1.0e-6);
//    LUCI_INTERPRETER_CHECK(output()->zero_point() == input()->zero_point());
//  }
//  else if (input()->element_type() == DataType::S16)
//  {
//    LUCI_INTERPRETER_CHECK(std::abs(output()->scale() - input()->scale()) <= 1.0e-6);
//    LUCI_INTERPRETER_CHECK(input()->zero_point() == 0 && output()->zero_point() == 0);
//  }
//}
//
//void MaxPool2D::execute() const
//{
//  switch (input()->element_type())
//  {
//    case DataType::FLOAT32:
//      evalFloat();
//      break;
//    case DataType::U8:
//      evalQuantized();
//      break;
//    case DataType::S16:
//      evalSInt16();
//      break;
//    default:
//      assert(false && "Unsupported type.");
//  }
//}
//
//void MaxPool2D::evalFloat() const
//{
//  float activation_min{};
//  float activation_max{};
//  calculateActivationRange(_params.activation, &activation_min, &activation_max);
//
//  tflite::PoolParams params{};
//  params.padding_values.height = _padding_height;
//  params.padding_values.width = _padding_width;
//  params.stride_height = _params.stride_height;
//  params.stride_width = _params.stride_width;
//  params.filter_height = _params.filter_height;
//  params.filter_width = _params.filter_width;
//  params.float_activation_min = activation_min;
//  params.float_activation_max = activation_max;
//
//  tflite::reference_ops::MaxPool(params, getTensorShape(input()), getTensorData<float>(input()),
//                                 getTensorShape(output()), getTensorData<float>(output()));
//}
//
//void MaxPool2D::evalQuantized() const
//{
//  int32_t activation_min{};
//  int32_t activation_max{};
//  calculateActivationRangeQuantized(_params.activation, output(), &activation_min, &activation_max);
//
//  tflite::PoolParams params{};
//  params.padding_values.height = _padding_height;
//  params.padding_values.width = _padding_width;
//  params.stride_height = _params.stride_height;
//  params.stride_width = _params.stride_width;
//  params.filter_height = _params.filter_height;
//  params.filter_width = _params.filter_width;
//  params.quantized_activation_min = activation_min;
//  params.quantized_activation_max = activation_max;
//
//  tflite::reference_ops::MaxPool(params, getTensorShape(input()), getTensorData<uint8_t>(input()),
//                                 getTensorShape(output()), getTensorData<uint8_t>(output()));
//}
//
//void MaxPool2D::evalSInt16() const
//{
//  int32_t activation_min{};
//  int32_t activation_max{};
//  calculateActivationRangeQuantized(_params.activation, output(), &activation_min, &activation_max);
//
//  tflite::PoolParams params{};
//  params.padding_values.height = _padding_height;
//  params.padding_values.width = _padding_width;
//  params.stride_height = _params.stride_height;
//  params.stride_width = _params.stride_width;
//  params.filter_height = _params.filter_height;
//  params.filter_width = _params.filter_width;
//  params.quantized_activation_min = activation_min;
//  params.quantized_activation_max = activation_max;
//
//  tflite::reference_integer_ops::MaxPool(
//    params, getTensorShape(input()), getTensorData<int16_t>(input()), //
//    getTensorShape(output()), getTensorData<int16_t>(output()));
//}

} // namespace luci_interpreter
