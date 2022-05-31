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

#include "kernels/Logistic.h"

#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/reference/logistic.h>
#include "tensorflow/lite/kernels/internal/reference/integer_ops/logistic.h"
#include <tensorflow/lite/kernels/internal/quantization_util.h>

namespace luci_interpreter
{
namespace kernels
{

Logistic::Logistic(std::vector<std::pair<const Tensor *, int32_t>> &&inputs, std::vector<std::pair<Tensor *, int32_t>> &&outputs)
  : Kernel(std::move(inputs), std::move(outputs))
{

}

void Logistic::calculateOpDataLogistic()
{
//  const int kInputIntegerBits = 4;
//
//  const double input_real_multiplier = static_cast<double>(input()->scale()) * static_cast<double>(1 << (31 - kInputIntegerBits));
//
//  int input_left_shift;
//  const double q = std::frexp(input_real_multiplier, &input_left_shift);
//
//  int32_t input_multiplier = static_cast<int32_t>(tflite::TfLiteRound(q * (1ll << 31)));
//
//  int32_t input_range_radius = tflite::CalculateInputRadius(kInputIntegerBits, input_left_shift, 31);
//
//  _op_data_logistic = OpDataLogistic{input()->zero_point(), input_range_radius, input_multiplier, input_left_shift};
}

void Logistic::configure(luci::CircleReader *circle_reader, int32_t index)
{
//  LUCI_INTERPRETER_CHECK(input()->element_type() == output()->element_type());
//  if (input()->element_type() == DataType::U8)
//  {
//    calculateOpDataLogistic();
//  }
  //output()->resize(input()->shape());
}

void Logistic::execute(luci::CircleReader *circle_reader, int32_t index) const
{
  auto input_dtype = luci::luci_datatype(circle_reader->tensors()[input_ind()]->type());
  switch (input_dtype)
  {
    case DataType::FLOAT32:
      evalFloat(circle_reader);
      break;
    case DataType::U8:
      evalQuantized();
      break;
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

void Logistic::evalFloat(luci::CircleReader *circle_reader) const
{
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


  tflite::reference_ops::Logistic(input_runtime_shape, getTensorData<float>(input()),
                                  output_runtime_shape, getTensorData<float>(output()));
}

void Logistic::evalQuantized() const
{
//  tflite::reference_integer_ops::Logistic(_op_data_logistic.input_zero_point, _op_data_logistic.input_range_radius,
//                                  _op_data_logistic.input_multiplier, _op_data_logistic.input_left_shift,
//                                  input()->shape().num_elements(),
//                                  getTensorData<int8_t>(input()), getTensorData<int8_t>(output()));
}

} // namespace kernels
} // namespace luci_interpreter
