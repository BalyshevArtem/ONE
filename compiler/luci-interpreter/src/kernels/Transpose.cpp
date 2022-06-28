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

#include "kernels/Transpose.h"

#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/reference/transpose.h>

#include <stdexcept>

namespace luci_interpreter
{

namespace kernels
{

Transpose::Transpose(std::vector<std::pair<const Tensor *, int32_t>> &&inputs, std::vector<std::pair<Tensor *, int32_t>> &&outputs)
  : Kernel(std::move(inputs), std::move(outputs))
{
}

void Transpose::configure(luci::CircleReader *circle_reader, int32_t index)
{
  // Transpose op only supports 1D-4D input arrays.
//  int dims = input()->shape().num_dims();
//  const int32_t *perm_data = getTensorData<int32_t>(perm());
//
//  assert(input()->shape().num_dims() <= 4);
//  assert(input()->element_type() == output()->element_type());
//
//  assert(perm()->shape().num_dims() == 1);
//  assert(perm()->shape().dim(0) == dims);
//
//  Shape output_shape(dims);
//  for (int i = 0; i < dims; i++)
//  {
//    assert(perm_data[i] < dims && perm_data[i] >= 0);
//    output_shape.dim(i) = input()->shape().dim(perm_data[i]);
//  }
//
//  output()->resize(output_shape);
}

void Transpose::execute(luci::CircleReader *circle_reader, int32_t index) const
{
  tflite::TransposeParams params{};
  const int32_t *perm_data = getTensorData<int32_t>(perm());

  const auto input_tensor_shape = luci::wrap(circle_reader->tensors()[input_ind()]->shape());
  tflite::RuntimeShape input_runtime_shape(input_tensor_shape.size());
  for (int i = 0; i < input_tensor_shape.size(); ++i)
  {
    input_runtime_shape.SetDim(i, input_tensor_shape[i]);
  }

  const auto perm_tensor_shape = luci::wrap(circle_reader->tensors()[perm_ind()]->shape());
  tflite::RuntimeShape perm_runtime_shape(perm_tensor_shape.size());
  for (int i = 0; i < perm_tensor_shape.size(); ++i)
  {
    perm_runtime_shape.SetDim(i, perm_tensor_shape[i]);
  }

  const auto output_tensor_shape = luci::wrap(circle_reader->tensors()[output_ind()]->shape());
  tflite::RuntimeShape output_runtime_shape(output_tensor_shape.size());
  for (int i = 0; i < output_tensor_shape.size(); ++i)
  {
    output_runtime_shape.SetDim(i, output_tensor_shape[i]);
  }

  const int32_t size = perm_tensor_shape[0]; //perm()->shape().dim(0);
  params.perm_count = size;
  for (int i = 0; i < size; i++)
    params.perm[i] = perm_data[i];

  auto input_dtype = luci::luci_datatype(circle_reader->tensors()[input_ind()]->type());
  switch (input_dtype)
  {
    case DataType::FLOAT32:
      tflite::reference_ops::Transpose(params, input_runtime_shape,
                                       getTensorData<float>(input()), output_runtime_shape,
                                       getTensorData<float>(output()));
      break;
//    case DataType::U8:
//      tflite::reference_ops::Transpose(params, getTensorShape(input()),
//                                       getTensorData<uint8_t>(input()), getTensorShape(output()),
//                                       getTensorData<uint8_t>(output()));
//      break;
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

} // namespace kernels
} // namespace luci_interpreter
