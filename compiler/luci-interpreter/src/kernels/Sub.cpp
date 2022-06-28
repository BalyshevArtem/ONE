/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

#include "kernels/Sub.h"
#include "kernels/Utils.h"

#include "PALSub.h"

#include <tensorflow/lite/kernels/internal/reference/process_broadcast_shapes.h>

#include <stdexcept>

namespace luci_interpreter
{
namespace kernels
{

Sub::Sub(std::vector<std::pair<const Tensor *, int32_t>> &&inputs, std::vector<std::pair<Tensor *, int32_t>> &&outputs)
  : Kernel(std::move(inputs), std::move(outputs))
{
}

void Sub::configure(luci::CircleReader *circle_reader, int32_t index)
{
//  LUCI_INTERPRETER_CHECK(!(input1()->element_type() != input2()->element_type()))
//  LUCI_INTERPRETER_CHECK(!(input1()->element_type() != output()->element_type()))
//  output()->resize(calculateShapeForBroadcast(input1()->shape(), input2()->shape()));
}

void Sub::execute(luci::CircleReader *circle_reader, int32_t index) const
{
  auto input_dtype = luci::luci_datatype(circle_reader->tensors()[input1_ind()]->type());

  switch (input_dtype)
  {
    case DataType::FLOAT32:
      evalFloat(circle_reader, index);
      break;
//    case DataType::S64:
//      evalInteger<int64_t>();
//      break;
//    case DataType::S32:
//      evalInteger<int32_t>();
//      break;
//    case DataType::U8:
//      evalQuantized();
//      break;
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

void Sub::evalFloat(luci::CircleReader *circle_reader, int32_t index) const
{
  circle::OperatorT oper_t;
  circle_reader->operators()[index]->UnPackTo(&oper_t);
  const auto *options = oper_t.builtin_options.AsSubOptions();

//  float activation_min{};
//  float activation_max{};
//  calculateActivationRange(luci::luci_actfunc(options->fused_activation_function), &activation_min, &activation_max);

  tflite::ArithmeticParams params{};
  fillArithmeticActivationRange<float>(params, luci::luci_actfunc(options->fused_activation_function));

  const auto input_tensor_shape_1 = luci::wrap(circle_reader->tensors()[input1_ind()]->shape());
  tflite::RuntimeShape input_runtime_shape_1(input_tensor_shape_1.size());
  for (int i = 0; i < input_tensor_shape_1.size(); ++i)
  {
    input_runtime_shape_1.SetDim(i, input_tensor_shape_1[i]);
  }

  const auto input_tensor_shape_2 = luci::wrap(circle_reader->tensors()[input2_ind()]->shape());
  tflite::RuntimeShape input_runtime_shape_2(input_tensor_shape_2.size());
  for (int i = 0; i < input_tensor_shape_2.size(); ++i)
  {
    input_runtime_shape_2.SetDim(i, input_tensor_shape_2[i]);
  }

  const auto output_tensor_shape = luci::wrap(circle_reader->tensors()[output_ind()]->shape());
  tflite::RuntimeShape output_runtime_shape(output_tensor_shape.size());
  for (int i = 0; i < output_tensor_shape.size(); ++i)
  {
    output_runtime_shape.SetDim(i, output_tensor_shape[i]);
  }

  const bool need_broadcast = tflite::reference_ops::ProcessBroadcastShapes(
    input_runtime_shape_1, input_runtime_shape_2, &params);

  if (need_broadcast)
  {
    tflite::reference_ops::BroadcastSubSlow(
      params, input_runtime_shape_1, getTensorData<float>(input1()), input_runtime_shape_2,
      getTensorData<float>(input2()), output_runtime_shape, getTensorData<float>(output()));
  }
  else
  {
    luci_interpreter_pal::Sub(params, input_runtime_shape_1, getTensorData<float>(input1()),
                              input_runtime_shape_1, getTensorData<float>(input2()),
                              output_runtime_shape, getTensorData<float>(output()));
  }
}
//
//template <typename T> void Sub::evalInteger() const
//{
//  tflite::ArithmeticParams params{};
//  fillArithmeticActivationRange<T>(params, _params.activation);
//
//  const bool need_broadcast = tflite::reference_ops::ProcessBroadcastShapes(
//    getTensorShape(input1()), getTensorShape(input2()), &params);
//
//  if (need_broadcast)
//  {
//    tflite::reference_ops::BroadcastSubSlow(
//      params, getTensorShape(input1()), getTensorData<T>(input1()), getTensorShape(input2()),
//      getTensorData<T>(input2()), getTensorShape(output()), getTensorData<T>(output()));
//  }
//  else
//  {
//    tflite::reference_ops::Sub(params, getTensorShape(input1()), getTensorData<T>(input1()),
//                               getTensorShape(input2()), getTensorData<T>(input2()),
//                               getTensorShape(output()), getTensorData<T>(output()));
//  }
//}
//
//void Sub::evalQuantized() const
//{
//  const auto input1_scale = static_cast<double>(input1()->scale());
//  const auto input2_scale = static_cast<double>(input2()->scale());
//  const auto output_scale = static_cast<double>(output()->scale());
//
//  const int left_shift = 20;
//  const double twice_max_input_scale = 2 * std::max(input1_scale, input2_scale);
//  const double real_input1_multiplier = input1_scale / twice_max_input_scale;
//  const double real_input2_multiplier = input2_scale / twice_max_input_scale;
//  const double real_output_multiplier = twice_max_input_scale / ((1 << left_shift) * output_scale);
//
//  int32_t input1_multiplier{}, input2_multiplier{}, output_multiplier{};
//  int input1_shift{}, input2_shift{}, output_shift{};
//  quantizeMultiplierSmallerThanOneExp(real_input1_multiplier, &input1_multiplier, &input1_shift);
//  quantizeMultiplierSmallerThanOneExp(real_input2_multiplier, &input2_multiplier, &input2_shift);
//  quantizeMultiplierSmallerThanOneExp(real_output_multiplier, &output_multiplier, &output_shift);
//
//  int32_t activation_min{};
//  int32_t activation_max{};
//  calculateActivationRangeQuantized(_params.activation, output(), &activation_min, &activation_max);
//
//  tflite::ArithmeticParams params{};
//  params.left_shift = left_shift;
//  // The kernel expects inputs' zero points to be negated.
//  params.input1_offset = -input1()->zero_point(); // Note the '-'.
//  params.input1_multiplier = input1_multiplier;
//  params.input1_shift = input1_shift;
//  params.input2_offset = -input2()->zero_point(); // Note the '-'.
//  params.input2_multiplier = input2_multiplier;
//  params.input2_shift = input2_shift;
//  params.output_offset = output()->zero_point();
//  params.output_multiplier = output_multiplier;
//  params.output_shift = output_shift;
//  params.quantized_activation_min = activation_min;
//  params.quantized_activation_max = activation_max;
//
//  const bool need_broadcast = tflite::reference_ops::ProcessBroadcastShapes(
//    getTensorShape(input1()), getTensorShape(input2()), &params);
//
//  if (need_broadcast)
//  {
//    tflite::reference_ops::BroadcastSubSlow(
//      params, getTensorShape(input1()), getTensorData<uint8_t>(input1()), getTensorShape(input2()),
//      getTensorData<uint8_t>(input2()), getTensorShape(output()), getTensorData<uint8_t>(output()));
//  }
//  else
//  {
//    tflite::reference_ops::Sub(params, getTensorShape(input1()), getTensorData<uint8_t>(input1()),
//                               getTensorShape(input2()), getTensorData<uint8_t>(input2()),
//                               getTensorShape(output()), getTensorData<uint8_t>(output()));
//  }
//}

} // namespace kernels
} // namespace luci_interpreter
