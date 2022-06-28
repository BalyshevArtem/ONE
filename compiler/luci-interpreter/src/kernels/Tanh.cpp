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

#include "kernels/Tanh.h"

#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/reference/tanh.h>

namespace luci_interpreter
{
namespace kernels
{

Tanh::Tanh(std::vector<std::pair<const Tensor *, int32_t>> &&inputs, std::vector<std::pair<Tensor *, int32_t>> &&outputs)
  : Kernel(std::move(inputs), std::move(outputs))
{
}

void Tanh::configure(luci::CircleReader *circle_reader, int32_t index)
{
//  LUCI_INTERPRETER_CHECK(input()->element_type() == output()->element_type());
//  if (input()->element_type() == DataType::U8)
//  {
//    populateLookupTable();
//  }
//  output()->resize(input()->shape());
}

void Tanh::execute(luci::CircleReader *circle_reader, int32_t index) const
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
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

void Tanh::evalFloat(luci::CircleReader *circle_reader, int32_t index) const
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

  tflite::reference_ops::Tanh(input_runtime_shape, getTensorData<float>(input()),
                              output_runtime_shape, getTensorData<float>(output()));
}
//
//void Tanh::evalQuantized() const
//{
//  const int size = tflite::MatchingFlatSize(getTensorShape(input()), getTensorShape(output()));
//  uint8_t *output_data = getTensorData<uint8_t>(output());
//  const uint8_t *input_data = getTensorData<uint8_t>(input());
//  for (int i = 0; i < size; ++i)
//  {
//    output_data[i] = getTableValue(input_data[i]);
//  }
//}
//
//void Tanh::populateLookupTable()
//{
//  const auto input_scale = static_cast<double>(input()->scale());
//  const auto input_zero_point = static_cast<int32_t>(input()->zero_point());
//  const auto output_scale = static_cast<double>(output()->scale());
//  const auto output_zero_point = static_cast<int32_t>(output()->zero_point());
//  const float inverse_scale = 1 / output_scale;
//  int32_t maxval = std::numeric_limits<uint8_t>::max();
//  int32_t minval = std::numeric_limits<uint8_t>::min();
//  for (int32_t val = minval; val <= maxval; ++val)
//  {
//    const float dequantized = input_scale * (val - input_zero_point);
//    const float transformed = std::tanh(dequantized);
//    const float rescaled = std::round(transformed * inverse_scale);
//    const int32_t quantized = static_cast<int32_t>(rescaled + output_zero_point);
//    setTableValue(static_cast<uint8_t>(std::max(std::min(maxval, quantized), minval)),
//                  static_cast<uint8_t>(val));
//  }
//}

} // namespace kernels
} // namespace luci_interpreter
