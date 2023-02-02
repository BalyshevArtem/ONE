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

#include "PALFullyConnected.h"

namespace luci_interpreter
{


namespace
{
void evalFloat(const Tensor *input, const Tensor *weights, const Tensor *bias, Tensor *output,
               const circle::FullyConnectedOptions *options)
{
  float activation_min{};
  float activation_max{};
  kernels::calculateActivationRange(luci_actfunc(options->fused_activation_function()),
                                    &activation_min, &activation_max);

  tflite::FullyConnectedParams params{};
  params.float_activation_min = activation_min;
  params.float_activation_max = activation_max;
  params.weights_format = tflite::FullyConnectedWeightsFormat::kDefault;

  tflite::reference_ops::FullyConnected(
    params, kernels::getTensorShape(input), kernels::getTensorData<float>(input),
    kernels::getTensorShape(weights), kernels::getTensorData<float>(weights),
    kernels::getTensorShape(bias), kernels::getTensorData<float>(bias),
    kernels::getTensorShape(output), kernels::getTensorData<float>(output));
}

#ifndef DIS_QUANT
void evalQuantized() const
{
  double real_multiplier = 0.0;
  int output_shift;
  int32_t output_activation_min;
  int32_t output_activation_max;
  int32_t output_multiplier;
  real_multiplier =
    getQuantizedConvolutionMultipler(input()->scale(), weights()->scale(), output()->scale());
  quantizeMultiplier(real_multiplier, &output_multiplier, &output_shift);
  calculateActivationRangeQuantized(params().activation, output(), &output_activation_min,
                                    &output_activation_max);

  int32_t input_offset = -input()->zero_point();
  int32_t filter_offset = -weights()->zero_point();
  int32_t output_offset = output()->zero_point();

  tflite::FullyConnectedParams op_params{};
  op_params.input_offset = input_offset;
  op_params.weights_offset = filter_offset;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = output_multiplier;
  op_params.output_shift = output_shift;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;
  op_params.lhs_cacheable = false;
  op_params.rhs_cacheable = false;
  tflite::reference_ops::FullyConnected(
    op_params, getTensorShape(input()), getTensorData<uint8_t>(input()), getTensorShape(weights()),
    getTensorData<uint8_t>(weights()), getTensorShape(bias()), getTensorData<int32_t>(bias()),
    getTensorShape(output()), getTensorData<uint8_t>(output()));
}

void evalQuantizedS8() const
{
  double real_multiplier = 0.0;
  int output_shift;
  int32_t output_activation_min;
  int32_t output_activation_max;
  int32_t output_multiplier;
  real_multiplier =
    getQuantizedConvolutionMultipler(input()->scale(), weights()->scale(), output()->scale());
  quantizeMultiplier(real_multiplier, &output_multiplier, &output_shift);
  calculateActivationRangeQuantized(params().activation, output(), &output_activation_min,
                                    &output_activation_max);

  int32_t input_offset = -input()->zero_point();
  int32_t filter_offset = -weights()->zero_point();
  int32_t output_offset = output()->zero_point();

  tflite::FullyConnectedParams op_params{};
  op_params.input_offset = input_offset;
  op_params.weights_offset = filter_offset;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = output_multiplier;
  op_params.output_shift = output_shift;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;
  op_params.lhs_cacheable = false;
  op_params.rhs_cacheable = false;
  luci_interpreter_pal::FullyConnected<int8_t>(
    op_params, getTensorShape(input()), getTensorData<int8_t>(input()), getTensorShape(weights()),
    getTensorData<int8_t>(weights()), getTensorShape(bias()), getTensorData<int32_t>(bias()),
    getTensorShape(output()), getTensorData<int8_t>(output()));
}
#endif

} // namespace

// TODO think how remove unused param
void configure_kernel_CircleFullyConnected(const circle::Operator *cur_op,
                                           IBaseRuntimeGraph *runtime_graph)
{
  const auto input_index = cur_op->inputs()->operator[](0);
  const auto weight_index = cur_op->inputs()->operator[](1);
  const auto bias_index = cur_op->inputs()->operator[](2);

  const auto output_index = cur_op->outputs()->operator[](0);

  assert(input_index != -1);
  assert(weight_index != -1);
  assert(output_index != -1);

  const auto input = runtime_graph->getTensorByIndex(input_index);

  const auto raw_weights = runtime_graph->getCircleTensorByIndex(weight_index);
  const auto raw_bias = runtime_graph->getCircleTensorByIndex(bias_index);

  assert(raw_weights != nullptr);

  auto weights_data = runtime_graph->getDataBufferFromCircleTensor(raw_weights);
  auto bias_data = runtime_graph->getDataBufferFromCircleTensor(raw_bias);

  Tensor weights(raw_weights);
  weights.writeDataWithoutCopy(static_cast<void *>(weights_data));

  Tensor bias(raw_bias);
  if (bias_data != nullptr)
    weights.writeDataWithoutCopy(static_cast<void *>(bias_data));

  //const auto weights = runtime_graph->getTensorByIndex(weight_index);
  //const auto bias = runtime_graph->getTensorByIndex(bias_index);

  auto output = runtime_graph->getTensorByIndex(output_index);

  assert(input != nullptr);
  assert(output != nullptr);

  if (weights.element_type() == DataType::U8)
  {
    LUCI_INTERPRETER_CHECK(input->element_type() == DataType::U8);
    LUCI_INTERPRETER_CHECK(output->element_type() == DataType::U8);
    LUCI_INTERPRETER_CHECK(!raw_bias || bias.element_type() == DataType::S32)
  }
  else if (weights.element_type() == DataType::FLOAT32)
  {
    LUCI_INTERPRETER_CHECK(input->element_type() == DataType::FLOAT32);
    LUCI_INTERPRETER_CHECK(output->element_type() == DataType::FLOAT32);
    LUCI_INTERPRETER_CHECK(!raw_bias || bias.element_type() == DataType::FLOAT32)
  }
  else if (weights.element_type() == DataType::S8)
  {
    LUCI_INTERPRETER_CHECK(input->element_type() == DataType::S8);
    LUCI_INTERPRETER_CHECK(output->element_type() == DataType::S8);
    LUCI_INTERPRETER_CHECK(!raw_bias || bias.element_type() == DataType::S32)
  }
  else
  {
    assert(false && "Unsupported type.");
  }

  // const Shape &input_shape = input()->shape();
  // const Shape &weights_shape = weights()->shape();

  LUCI_INTERPRETER_CHECK(weights.num_dims() == 2); // weights_shape.num_dims() == 2);
  LUCI_INTERPRETER_CHECK(raw_bias == nullptr || bias.num_elements() == weights.dim(0));
  // bias()->shape().num_elements() == weights_shape.dim(0));

  LUCI_INTERPRETER_CHECK(input->num_elements() % weights.dim(1) == 0);
  const int32_t batch_size = input->num_elements() / weights.dim(1);
  const int32_t num_units = weights.dim(0);

  if (raw_bias)
    LUCI_INTERPRETER_CHECK(bias.num_elements() == weights.dim(0));

  const auto *options = cur_op->builtin_options_as_FullyConnectedOptions();

  // TODO: handle with it
  // TODO: enable it only if kernel with dynamic shapes
  if (!options->keep_num_dims())
  {
    // output()->resize({batch_size, num_units});
  }
  else
  {
    //    luci_interpreter::Shape output_shape(input_shape.num_dims());
    //    for (int i = 0; i < input_shape.num_dims(); ++i)
    //      output_shape.dim(i) = input_shape.dim(i);
    //    output_shape.dim(input_shape.num_dims() - 1) = num_units;
    //    output()->resize(output_shape);
  }
}

// TODO think how remove unused param
void execute_kernel_CircleFullyConnected(const circle::Operator *cur_op,
                                         IBaseRuntimeGraph *runtime_graph,
                                         bool)
{
  const auto input_index = cur_op->inputs()->operator[](0);
  const auto weight_index = cur_op->inputs()->operator[](1);
  const auto bias_index = cur_op->inputs()->operator[](2);

  const auto output_index = cur_op->outputs()->operator[](0);

  assert(input_index != -1);
  assert(weight_index != -1);
  assert(output_index != -1);

  const auto input = runtime_graph->getTensorByIndex(input_index);

  const auto raw_weights = runtime_graph->getCircleTensorByIndex(weight_index);
  const auto raw_bias = runtime_graph->getCircleTensorByIndex(bias_index);

  assert(raw_weights != nullptr);

  auto weights_data = runtime_graph->getDataBufferFromCircleTensor(raw_weights);
  auto bias_data = runtime_graph->getDataBufferFromCircleTensor(raw_bias);

  Tensor weights(raw_weights);
  weights.writeDataWithoutCopy(static_cast<void *>(weights_data));

  Tensor bias(raw_bias);
  if (bias_data != nullptr)
    bias.writeDataWithoutCopy(static_cast<void *>(bias_data));

  //const auto weights = runtime_graph->getTensorByIndex(weight_index);
  //const auto bias = runtime_graph->getTensorByIndex(bias_index);

  auto output = runtime_graph->getTensorByIndex(output_index);

  const auto *options = cur_op->builtin_options_as_FullyConnectedOptions();

  switch (input->element_type())
  {
#ifndef DIS_QUANT
    case DataType::U8:
      evalQuantized();
      break;
    case DataType::S8:
      evalQuantizedS8();
      break;
#endif
    case DataType::FLOAT32:
      evalFloat(input, &weights, raw_bias == nullptr ? nullptr : &bias, output, options);
      break;
    default:
      assert(false && "Unsupported type.");
  }
}

} // namespace luci_interpreter
