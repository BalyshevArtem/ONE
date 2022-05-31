/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/BatchToSpaceND.h"
#include "kernels/Utils.h"

#include "PALBatchToSpaceND.h"

#include <stdexcept>

namespace luci_interpreter
{

namespace kernels
{

namespace
{
const int kInputMinDimensionNum = 3;
const int kInputMaxDimensionNum = 4;
} // namespace

BatchToSpaceND::BatchToSpaceND(std::vector<std::pair<const Tensor *, int32_t>> &&inputs, std::vector<std::pair<Tensor *, int32_t>> &&outputs)
  : Kernel(std::move(inputs), std::move(outputs))
{
}

void BatchToSpaceND::configure(luci::CircleReader *circle_reader, int32_t index)
{
//  if (input()->shape().num_dims() == 3)
//  {
//    output()->resize(_own_shape);
//    return;
//  }
//    return;
//  const auto *block_shape_data = block_shape()->data<int32_t>();
//  const auto *crops_data = crops()->data<int32_t>();
//  LUCI_INTERPRETER_CHECK(input()->shape().num_dims() >= kInputMinDimensionNum);
//  LUCI_INTERPRETER_CHECK(input()->shape().num_dims() <= kInputMaxDimensionNum);
//  LUCI_INTERPRETER_CHECK(input()->element_type() == output()->element_type());
//
//  int spatial_dims_num = input()->shape().num_dims() - 2;
//
//  LUCI_INTERPRETER_CHECK(block_shape()->shape().num_dims() == 1);
//  LUCI_INTERPRETER_CHECK(block_shape()->shape().dim(0) == spatial_dims_num);
//
//  LUCI_INTERPRETER_CHECK(crops()->shape().num_dims() == 2);
//  LUCI_INTERPRETER_CHECK(crops()->shape().dim(0) == spatial_dims_num);
//  LUCI_INTERPRETER_CHECK(crops()->shape().dim(1) == 2);
//  for (int i = 0; i < spatial_dims_num * 2; ++i)
//  {
//    LUCI_INTERPRETER_CHECK(crops_data[i] >= 0);
//  }
//
//  Shape output_shape = Shape(input()->shape().num_dims());
//  int output_batch_size = input()->shape().dim(0);
//  for (int i = 0; i < spatial_dims_num; ++i)
//  {
//    LUCI_INTERPRETER_CHECK(output_batch_size % block_shape_data[i] == 0);
//    output_batch_size = output_batch_size / block_shape_data[i];
//    output_shape.dim(i + 1) =
//      input()->shape().dim(i + 1) * block_shape_data[i] - crops_data[i * 2] - crops_data[i * 2 + 1];
//  }
//
//  output_shape.dim(0) = output_batch_size;
//  output_shape.dim(input()->shape().num_dims() - 1) =
//    input()->shape().dim(input()->shape().num_dims() - 1);
//  output()->resize(output_shape);
}

void BatchToSpaceND::execute(luci::CircleReader *circle_reader, int32_t index) const
{
  auto input_dtype = luci::luci_datatype(circle_reader->tensors()[input_ind()]->type());
  switch (input_dtype)
  {
    case DataType::FLOAT32:
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

      const auto block_tensor_shape = luci::wrap(circle_reader->tensors()[block_ind()]->shape());
      tflite::RuntimeShape block_runtime_shape(block_tensor_shape.size());
      for (int i = 0; i < block_tensor_shape.size(); ++i)
      {
        block_runtime_shape.SetDim(i, block_tensor_shape[i]);
      }

      const auto pad_tensor_shape = luci::wrap(circle_reader->tensors()[crops_ind()]->shape());
      tflite::RuntimeShape pad_runtime_shape(pad_tensor_shape.size());
      for (int i = 0; i < pad_tensor_shape.size(); ++i)
      {
        pad_runtime_shape.SetDim(i, pad_tensor_shape[i]);
      }

      luci_interpreter_pal::BatchToSpaceND(
        input_runtime_shape, getTensorData<float>(input()), block_runtime_shape,
        getTensorData<int32_t>(block_shape()), pad_runtime_shape,
        getTensorData<int32_t>(crops()), output_runtime_shape, getTensorData<float>(output()));
      break;
    }
//    case DataType::U8:
//      luci_interpreter_pal::BatchToSpaceND(
//        getTensorShape(input()), getTensorData<uint8_t>(input()), getTensorShape(block_shape()),
//        getTensorData<int32_t>(block_shape()), getTensorShape(crops()),
//        getTensorData<int32_t>(crops()), getTensorShape(output()),
//        getTensorData<uint8_t>(output()));
//      break;
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

} // namespace kernels
} // namespace luci_interpreter
