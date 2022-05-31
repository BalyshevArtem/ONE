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

#include "kernels/MaxPool2D.h"

#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/reference/integer_ops/pooling.h>
#include <tensorflow/lite/kernels/internal/reference/pooling.h>

#include <stdexcept>

namespace luci_interpreter
{

namespace kernels
{

MaxPool2D::MaxPool2D(std::vector<std::pair<const Tensor *, int32_t>> &&inputs, std::vector<std::pair<Tensor *, int32_t>> &&outputs)
  : Kernel(std::move(inputs), std::move(outputs))
{
}

void MaxPool2D::configure(luci::CircleReader *circle_reader, int32_t index)
{
  //LUCI_INTERPRETER_CHECK(input()->element_type() == output()->element_type());
  //assert(input()->shape().num_dims() == 4);
  const auto input_tensor_shape = luci::wrap(circle_reader->tensors()[input_ind()]->shape());

  Shape input_shape(static_cast<int>(input_tensor_shape.size()));
  for (uint32_t r = 0; r < input_tensor_shape.size(); ++r)
  {
    input_shape.dim(r) = input_tensor_shape[r];
  }

  //const Shape &input_shape = input()->shape();
  const int32_t batches = input_shape.dim(0);
  const int32_t input_height = input_shape.dim(1);
  const int32_t input_width = input_shape.dim(2);
  const int32_t depth = input_shape.dim(3);

  circle::OperatorT oper_t;
  circle_reader->operators()[index]->UnPackTo(&oper_t);
  const auto *options = oper_t.builtin_options.AsPool2DOptions();

  auto padding = luci::luci_padding(options->padding);
  auto stride_h = options->stride_h;
  auto stride_w = options->stride_w;
  auto filter_h = options->filter_height;
  auto filter_w = options->filter_width;

  const int32_t output_height =
    computeOutputSize(padding, input_height, filter_h, stride_h);
  const int32_t output_width =
    computeOutputSize(padding, input_width, filter_w, stride_w);

  _padding_height =
    computePadding(stride_h, 1, input_height, filter_h, output_height);
  _padding_width =
    computePadding(stride_w, 1, input_width, filter_w, output_width);

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
}

void MaxPool2D::execute(luci::CircleReader *circle_reader, int32_t index) const
{
  auto input_dtype = luci::luci_datatype(circle_reader->tensors()[input_ind()]->type());
  switch (input_dtype)
  {
    case DataType::FLOAT32:
      evalFloat(circle_reader, index);
      break;
//    case DataType::U8:
//      evalQuantized();
//      break;
//    case DataType::S16:
//      evalSInt16();
//      break;
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

void MaxPool2D::evalFloat(luci::CircleReader *circle_reader, int32_t index) const
{
  circle::OperatorT oper_t;
  circle_reader->operators()[index]->UnPackTo(&oper_t);
  const auto *options = oper_t.builtin_options.AsPool2DOptions();

  float activation_min{};
  float activation_max{};
  calculateActivationRange(luci::luci_actfunc(options->fused_activation_function), &activation_min, &activation_max);


//  float activation_min{};
//  float activation_max{};
//  calculateActivationRange(_params.activation, &activation_min, &activation_max);

  tflite::PoolParams params{};
  params.padding_values.height = _padding_height;
  params.padding_values.width = _padding_width;
  params.stride_height = options->stride_h;
  params.stride_width = options->stride_w;
  params.filter_height = options->filter_height;
  params.filter_width = options->filter_width;
  params.float_activation_min = activation_min;
  params.float_activation_max = activation_max;

  const auto input_tensor_shape = luci::wrap(circle_reader->tensors()[input_ind()]->shape());
  tflite::RuntimeShape input_runtime_shape(input_tensor_shape.size());
  for (int i = 0; i < input_tensor_shape.size(); ++i)
  {
    input_runtime_shape.SetDim(i, input_tensor_shape[i]);
  }

  const auto output_tensor_shape = luci::wrap(circle_reader->tensors()[output_ind()]->shape());
  tflite::RuntimeShape output_runtime_shape(output_tensor_shape.size());
  for (int i = 0; i < output_tensor_shape.size(); ++i)
  {
    output_runtime_shape.SetDim(i, output_tensor_shape[i]);
  }

  tflite::reference_ops::MaxPool(params, input_runtime_shape, getTensorData<float>(input()),
                                 output_runtime_shape, getTensorData<float>(output()));
}
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

} // namespace kernels
} // namespace luci_interpreter
