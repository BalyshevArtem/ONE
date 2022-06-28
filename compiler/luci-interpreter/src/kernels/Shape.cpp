/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/Shape.h"
#include "kernels/Utils.h"

namespace luci_interpreter
{
namespace kernels
{

ShapeKernel::ShapeKernel(std::vector<std::pair<const Tensor *, int32_t>> &&inputs, std::vector<std::pair<Tensor *, int32_t>> &&outputs)
  : Kernel(std::move(inputs), std::move(outputs))
{
}

void ShapeKernel::configure(luci::CircleReader *circle_reader, int32_t index)
{
//  LUCI_INTERPRETER_CHECK(output()->element_type() == DataType::S32 or
//                         output()->element_type() == DataType::S64);
//  const auto input_shape = input()->shape();
//
//  Shape output_shape(1);
//  output_shape.dim(0) = input_shape.num_dims();
//
//  output()->resize(output_shape);
}

void ShapeKernel::execute(luci::CircleReader *circle_reader, int32_t index) const
{
  auto output_dtype = luci::luci_datatype(circle_reader->tensors()[output_ind()]->type());

  switch (output_dtype)
  {
    case DataType::S32:
      evalInt<int32_t>(circle_reader, index);
      break;
    case DataType::S64:
      evalInt<int64_t>(circle_reader, index);
      break;
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

template <typename T> void ShapeKernel::evalInt(luci::CircleReader *circle_reader, int32_t index) const
{
  const auto input_tensor_shape = luci::wrap(circle_reader->tensors()[input_ind()]->shape());
//  tflite::RuntimeShape input_runtime_shape(input_tensor_shape.size());
//  for (int i = 0; i < input_tensor_shape.size(); ++i)
//  {
//    input_runtime_shape.SetDim(i, input_tensor_shape[i]);
//  }

  auto output_data = getTensorData<T>(output());

  for (int i = 0; i < input_tensor_shape.size(); ++i)
  {
    output_data[i] = input_tensor_shape[i];
  }
}

} // namespace kernels
} // namespace luci_interpreter
