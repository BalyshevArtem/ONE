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

#include "kernels/Pad.h"

#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/reference/pad.h>

namespace luci_interpreter
{
namespace kernels
{

Pad::Pad(std::vector<std::pair<const Tensor *, int32_t>> &&inputs, std::vector<std::pair<Tensor *, int32_t>> &&outputs)
  : Kernel(std::move(inputs), std::move(outputs))
{
}

void Pad::configure(luci::CircleReader *circle_reader, int32_t index)
{
//  const Shape &input_shape = input()->shape();
//  const int num_dims = input_shape.num_dims();
//
//  if (num_dims > 4)
//    throw std::runtime_error("Unsupported number of dimensions.");
//
//  assert(output()->element_type() == input()->element_type());
//  assert(paddings()->element_type() == DataType::S32);
//  // Paddings shape should be [N, 2].
//  assert(paddings()->shape().num_dims() == 2);
//  assert(paddings()->shape().dim(0) == num_dims);
//  assert(paddings()->shape().dim(1) == 2);
//
//  Shape output_shape(num_dims);
//  const auto *paddings_data = getTensorData<int32_t>(paddings());
//  for (int i = 0; i < num_dims; ++i)
//  {
//    const int32_t padding_before = paddings_data[i * 2];
//    const int32_t padding_after = paddings_data[i * 2 + 1];
//    assert(padding_before >= 0 && padding_after >= 0);
//    output_shape.dim(i) = input_shape.dim(i) + padding_before + padding_after;
//  }
//
//  output()->resize(output_shape);
}

void Pad::execute(luci::CircleReader *circle_reader, int32_t index) const
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

  const int num_dims = input_tensor_shape.size();

  tflite::PadParams params{};
  params.left_padding_count = num_dims;
  params.right_padding_count = num_dims;

  const auto *paddings_data = getTensorData<int32_t>(paddings());
  for (int i = num_dims - 1; i >= 0; --i)
  {
    params.left_padding[i] = paddings_data[i * 2];
    params.right_padding[i] = paddings_data[i * 2 + 1];
  }

  auto input_dtype = luci::luci_datatype(circle_reader->tensors()[input_ind()]->type());
  switch (input_dtype)
  {
    case DataType::FLOAT32:
    {
      const float pad_value = 0.0f;
      tflite::reference_ops::Pad(params, input_runtime_shape, getTensorData<float>(input()),
                                 &pad_value, output_runtime_shape,
                                 getTensorData<float>(output()));
      break;
    }
//    case DataType::U8:
//    {
//      assert(output()->zero_point() >= std::numeric_limits<uint8_t>::min());
//      assert(output()->zero_point() <= std::numeric_limits<uint8_t>::max());
//      const auto pad_value = static_cast<uint8_t>(output()->zero_point());
//      tflite::reference_ops::Pad(params, getTensorShape(input()), getTensorData<uint8_t>(input()),
//                                 &pad_value, getTensorShape(output()),
//                                 getTensorData<uint8_t>(output()));
//      break;
//    }
//    case DataType::S8:
//    {
//      assert(output()->zero_point() >= std::numeric_limits<int8_t>::min());
//      assert(output()->zero_point() <= std::numeric_limits<int8_t>::max());
//      const auto pad_value = static_cast<int8_t>(output()->zero_point());
//      tflite::reference_ops::Pad(params, getTensorShape(input()), getTensorData<int8_t>(input()),
//                                 &pad_value, getTensorShape(output()),
//                                 getTensorData<int8_t>(output()));
//      break;
//    }
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

} // namespace kernels
} // namespace luci_interpreter
