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

#include "kernels/FullyConnected.h"

#include "kernels/Utils.h"

#include "PALFullyConnected.h"

#include <stdexcept>

namespace luci_interpreter
{

namespace kernels
{

FullyConnected::FullyConnected(std::vector<std::pair<const Tensor *, int32_t>> &&inputs, std::vector<std::pair<Tensor *, int32_t>> &&outputs)
  : Kernel(std::move(inputs), std::move(outputs))
{
}

void FullyConnected::configure(luci::CircleReader *circle_reader, int32_t index)
{
//  if (weights()->element_type() == DataType::U8)
//  {
//    LUCI_INTERPRETER_CHECK(input()->element_type() == DataType::U8);
//    LUCI_INTERPRETER_CHECK(output()->element_type() == DataType::U8);
//    LUCI_INTERPRETER_CHECK(!bias() || bias()->element_type() == DataType::S32)
//  }
//  else if (weights()->element_type() == DataType::FLOAT32)
//  {
//    LUCI_INTERPRETER_CHECK(input()->element_type() == DataType::FLOAT32);
//    LUCI_INTERPRETER_CHECK(output()->element_type() == DataType::FLOAT32);
//    LUCI_INTERPRETER_CHECK(!bias() || bias()->element_type() == DataType::FLOAT32)
//  }
//  else if (weights()->element_type() == DataType::S8)
//  {
//    LUCI_INTERPRETER_CHECK(input()->element_type() == DataType::S8);
//    LUCI_INTERPRETER_CHECK(output()->element_type() == DataType::S8);
//    LUCI_INTERPRETER_CHECK(!bias() || bias()->element_type() == DataType::S32)
//  }
//  else
//  {
//    throw std::runtime_error("Unsupported type.");
//  }
//
//  const Shape &input_shape = input()->shape();
//  const Shape &weights_shape = weights()->shape();
//
//  LUCI_INTERPRETER_CHECK(weights_shape.num_dims() == 2);
//  LUCI_INTERPRETER_CHECK(bias() == nullptr ||
//                         bias()->shape().num_elements() == weights_shape.dim(0));
//
//  LUCI_INTERPRETER_CHECK(input_shape.num_elements() % weights_shape.dim(1) == 0);
//  const auto input_tensor_shape = luci::wrap(circle_reader->tensors()[input()->_index]->shape());
//
//  Shape input_shape(static_cast<int>(input_tensor_shape.size()));
//  for (uint32_t r = 0; r < input_tensor_shape.size(); ++r)
//  {
//    input_shape.dim(r) = input_tensor_shape[r];
//  }
//
//
//  const auto weight_tensor_shape = luci::wrap(circle_reader->tensors()[weights()->_index]->shape());
//
//  Shape weight_shape(static_cast<int>(weight_tensor_shape.size()));
//  for (uint32_t r = 0; r < weight_tensor_shape.size(); ++r)
//  {
//    weight_shape.dim(r) = weight_tensor_shape[r];
//  }
//
//  const int32_t batch_size = input_shape.num_elements() / weight_shape.dim(1);
//  const int32_t num_units = weight_shape.dim(0);
////
////  if (bias())
////    LUCI_INTERPRETER_CHECK(bias()->shape().num_elements() == weights()->shape().dim(0));
//  circle::OperatorT oper_t;
//  circle_reader->operators()[index]->UnPackTo(&oper_t);
//  const auto *options = oper_t.builtin_options.AsFullyConnectedOptions();
//  const auto keep_num_dims = options->keep_num_dims;
//  if (not keep_num_dims)
//  {
//    //output()->resize({batch_size, num_units});
//  }
//  else
//  {
////    luci_interpreter::Shape output_shape(input_shape.num_dims());
////    for (int i = 0; i < input_shape.num_dims(); ++i)
////      output_shape.dim(i) = input_shape.dim(i);
////    output_shape.dim(input_shape.num_dims() - 1) = num_units;
////    output()->resize(output_shape);
//  }
}

void FullyConnected::execute(luci::CircleReader *circle_reader, int32_t index) const
{
  auto input_dtype = luci::luci_datatype(circle_reader->tensors()[input_ind()]->type());

  switch (input_dtype)
  {
//    case DataType::U8:
//      evalQuantized(circle_reader, index);
//      break;
//    case DataType::S8:
//      evalQuantizedS8(op);
//      break;
    case DataType::FLOAT32:
      evalFloat(circle_reader, index);
      break;
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

void FullyConnected::evalFloat(luci::CircleReader *circle_reader, int32_t index) const
{
  float activation_min{};
  float activation_max{};

  //const auto *options = op.builtin_options.AsFullyConnectedOptions();
  circle::OperatorT oper_t;
  circle_reader->operators()[index]->UnPackTo(&oper_t);
  const auto *options = oper_t.builtin_options.AsFullyConnectedOptions();
  const auto activation = luci::luci_actfunc(options->fused_activation_function);
  calculateActivationRange(activation, &activation_min, &activation_max);

  tflite::FullyConnectedParams params{};
  params.float_activation_min = activation_min;
  params.float_activation_max = activation_max;
  params.weights_format = tflite::FullyConnectedWeightsFormat::kDefault;

  const auto input_tensor_shape = luci::wrap(circle_reader->tensors()[input_ind()]->shape());
  tflite::RuntimeShape input_runtime_shape(input_tensor_shape.size());
  for (int i = 0; i < input_tensor_shape.size(); ++i)
  {
    input_runtime_shape.SetDim(i, input_tensor_shape[i]);
  }

  const auto output_tensor_shape = luci::wrap(circle_reader->tensors()[outputs_ind()]->shape());
  tflite::RuntimeShape output_runtime_shape(output_tensor_shape.size());
  for (int i = 0; i < output_tensor_shape.size(); ++i)
  {
    output_runtime_shape.SetDim(i, output_tensor_shape[i]);
  }

  const auto weight_tensor_shape = luci::wrap(circle_reader->tensors()[weights_ind()]->shape());
  tflite::RuntimeShape weight_runtime_shape(weight_tensor_shape.size());
  for (int i = 0; i < weight_tensor_shape.size(); ++i)
  {
    weight_runtime_shape.SetDim(i, weight_tensor_shape[i]);
  }

  tflite::reference_ops::FullyConnected(
    params, input_runtime_shape, getTensorData<float>(input()), weight_runtime_shape,
    getTensorData<float>(weights()), tflite::RuntimeShape(), getTensorData<float>(bias()),
    output_runtime_shape, getTensorData<float>(output()));
}

void FullyConnected::evalQuantized(const circle::OperatorT &op) const
{
//  double real_multiplier = 0.0;
//  int output_shift;
//  int32_t output_activation_min;
//  int32_t output_activation_max;
//  int32_t output_multiplier;
//  real_multiplier =
//    getQuantizedConvolutionMultipler(input()->scale(), weights()->scale(), output()->scale());
//  quantizeMultiplier(real_multiplier, &output_multiplier, &output_shift);
//  const auto *options = op.builtin_options.AsFullyConnectedOptions();
//  const auto activation = luci::luci_actfunc(options->fused_activation_function);
//  calculateActivationRangeQuantized(activation, output(), &output_activation_min,
//                                    &output_activation_max);
//
//  int32_t input_offset = -input()->zero_point();
//  int32_t filter_offset = -weights()->zero_point();
//  int32_t output_offset = output()->zero_point();
//
//  tflite::FullyConnectedParams op_params{};
//  op_params.input_offset = input_offset;
//  op_params.weights_offset = filter_offset;
//  op_params.output_offset = output_offset;
//  op_params.output_multiplier = output_multiplier;
//  op_params.output_shift = output_shift;
//  op_params.quantized_activation_min = output_activation_min;
//  op_params.quantized_activation_max = output_activation_max;
//  op_params.lhs_cacheable = false;
//  op_params.rhs_cacheable = false;
//  tflite::reference_ops::FullyConnected(
//    op_params, getTensorShape(input()), getTensorData<uint8_t>(input()), getTensorShape(weights()),
//    getTensorData<uint8_t>(weights()), getTensorShape(bias()), getTensorData<int32_t>(bias()),
//    getTensorShape(output()), getTensorData<uint8_t>(output()));
}

void FullyConnected::evalQuantizedS8(const circle::OperatorT &op) const
{
//  double real_multiplier = 0.0;
//  int output_shift;
//  int32_t output_activation_min;
//  int32_t output_activation_max;
//  int32_t output_multiplier;
//  real_multiplier =
//    getQuantizedConvolutionMultipler(input()->scale(), weights()->scale(), output()->scale());
//  quantizeMultiplier(real_multiplier, &output_multiplier, &output_shift);
//  const auto *options = op.builtin_options.AsFullyConnectedOptions();
//  const auto activation = luci::luci_actfunc(options->fused_activation_function);
//  calculateActivationRangeQuantized(activation, output(), &output_activation_min,
//                                    &output_activation_max);
//
//  int32_t input_offset = -input()->zero_point();
//  int32_t filter_offset = -weights()->zero_point();
//  int32_t output_offset = output()->zero_point();
//
//  tflite::FullyConnectedParams op_params{};
//  op_params.input_offset = input_offset;
//  op_params.weights_offset = filter_offset;
//  op_params.output_offset = output_offset;
//  op_params.output_multiplier = output_multiplier;
//  op_params.output_shift = output_shift;
//  op_params.quantized_activation_min = output_activation_min;
//  op_params.quantized_activation_max = output_activation_max;
//  op_params.lhs_cacheable = false;
//  op_params.rhs_cacheable = false;
//  luci_interpreter_pal::FullyConnected<int8_t>(
//    op_params, getTensorShape(input()), getTensorData<int8_t>(input()), getTensorShape(weights()),
//    getTensorData<int8_t>(weights()), getTensorShape(bias()), getTensorData<int32_t>(bias()),
//    getTensorShape(output()), getTensorData<int8_t>(output()));
}

} // namespace kernels
} // namespace luci_interpreter
