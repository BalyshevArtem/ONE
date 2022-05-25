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

Logistic::Logistic(const Tensor *input, Tensor *output) : Kernel({input}, {output}) {}

void Logistic::calculateOpDataLogistic()
{
  const int kInputIntegerBits = 4;

  const double input_real_multiplier = static_cast<double>(input()->scale()) * static_cast<double>(1 << (31 - kInputIntegerBits));

  int input_left_shift;
  const double q = std::frexp(input_real_multiplier, &input_left_shift);

  int32_t input_multiplier = static_cast<int32_t>(tflite::TfLiteRound(q * (1ll << 31)));

  int32_t input_range_radius = tflite::CalculateInputRadius(kInputIntegerBits, input_left_shift, 31);

  _op_data_logistic = OpDataLogistic{input()->zero_point(), input_range_radius, input_multiplier, input_left_shift};
}

void Logistic::configure()
{
  LUCI_INTERPRETER_CHECK(input()->element_type() == output()->element_type());
  if (input()->element_type() == DataType::U8)
  {
    calculateOpDataLogistic();
  }
  //output()->resize(input()->shape());
}

void Logistic::execute() const
{
  switch (input()->element_type())
  {
    case DataType::FLOAT32:
      evalFloat();
      break;
    case DataType::U8:
      evalQuantized();
      break;
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

void Logistic::evalFloat() const
{
  tflite::reference_ops::Logistic(getTensorShape(input()), getTensorData<float>(input()),
                                  getTensorShape(output()), getTensorData<float>(output()));
}

void Logistic::evalQuantized() const
{
  tflite::reference_integer_ops::Logistic(_op_data_logistic.input_zero_point, _op_data_logistic.input_range_radius,
                                  _op_data_logistic.input_multiplier, _op_data_logistic.input_left_shift,
                                  input()->shape().num_elements(),
                                  getTensorData<int8_t>(input()), getTensorData<int8_t>(output()));
}

} // namespace kernels
} // namespace luci_interpreter
