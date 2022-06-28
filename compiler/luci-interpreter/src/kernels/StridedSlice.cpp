/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

#include "kernels/StridedSlice.h"

#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/reference/strided_slice.h>

#include <stdexcept>

namespace luci_interpreter
{

namespace kernels
{

StridedSlice::StridedSlice(std::vector<std::pair<const Tensor *, int32_t>> &&inputs, std::vector<std::pair<Tensor *, int32_t>> &&outputs)
  : Kernel(std::move(inputs), std::move(outputs))
{
}

void StridedSlice::configure(luci::CircleReader *circle_reader, int32_t index)
{
//  assert(begin()->shape().num_dims() == 1);
//  assert(end()->shape().num_dims() == 1);
//  assert(strides()->shape().num_dims() == 1);
//  assert(input()->element_type() == output()->element_type());
//  assert(begin()->element_type() == DataType::S32);
//  assert(end()->element_type() == DataType::S32);
//  assert(strides()->element_type() == DataType::S32);
//  assert(input()->shape().num_dims() <= 4);
//  if (params().ellipsis_mask != 0)
//  {
//    throw std::runtime_error("ellipsis_mask is not implemented yet.");
//  }
//  if (params().new_axis_mask != 0)
//  {
//    throw std::runtime_error("new_axis_mask is not implemented yet.");
//  }
//  if (input()->element_type() == DataType::U8)
//  {
//    assert(input()->scale() == output()->scale());
//    assert(input()->zero_point() == output()->zero_point());
//  }

  circle::OperatorT oper_t;
  circle_reader->operators()[index]->UnPackTo(&oper_t);
  const auto *options = oper_t.builtin_options.AsStridedSliceOptions();

  const auto input_tensor_shape = luci::wrap(circle_reader->tensors()[input_ind()]->shape());
  tflite::RuntimeShape input_runtime_shape(input_tensor_shape.size());
  for (int i = 0; i < input_tensor_shape.size(); ++i)
  {
    input_runtime_shape.SetDim(i, input_tensor_shape[i]);
  }

  tflite::StridedSliceParams op_params{};
  op_params.start_indices_count = input_tensor_shape.size(); //input()->shape().num_dims();
  op_params.stop_indices_count = input_tensor_shape.size();//input()->shape().num_dims();
  op_params.strides_count = input_tensor_shape.size();//input()->shape().num_dims();

  for (int i = 0; i < input_tensor_shape.size(); i++)
  {
    op_params.start_indices[i] = getTensorData<int32_t>(begin())[i];
    op_params.stop_indices[i] = getTensorData<int32_t>(end())[i];
    op_params.strides[i] = getTensorData<int32_t>(strides())[i];
  }
  op_params.begin_mask = options->begin_mask;
  op_params.ellipsis_mask = 0;
  op_params.end_mask = options->end_mask;
  op_params.new_axis_mask = 0;
  op_params.shrink_axis_mask = options->shrink_axis_mask;
  std::vector<int32_t> output_shape_vector;
  for (int i = 0; i < input_tensor_shape.size(); i++)
  {
    int idx = input_tensor_shape.size() - i - 1;
    int32_t stride = getTensorData<int32_t>(strides())[idx];
    assert(stride != 0);
    int32_t begin = ::tflite::strided_slice::StartForAxis(op_params, input_runtime_shape, idx);
    int32_t end =
      ::tflite::strided_slice::StopForAxis(op_params, input_runtime_shape, idx, begin);

    const bool shrink_axis = options->shrink_axis_mask & (1 << idx);
    if (shrink_axis)
    {
      end = begin + 1;
    }

    int32_t dim_shape = std::ceil((end - begin) / static_cast<float>(stride));
    dim_shape = dim_shape < 0 ? 0 : dim_shape;
    if (!shrink_axis)
    {
      output_shape_vector.push_back(dim_shape);
    }
  }

  auto size = 1;
  for (uint32_t r = 0; r < output_shape_vector.size(); ++r)
  {
    //shape.dim(r) = dims[r];
    size *= output_shape_vector[output_shape_vector.size() - r - 1];
  }

  auto output_dtype = luci::luci_datatype(circle_reader->tensors()[output_ind()]->type());

  size *= getDataTypeSize(output_dtype);

  //Shape output_shape = Shape(output_shape_vector.size());
  for (size_t i = 0; i < output_shape_vector.size(); i++)
  {
    //output_shape.dim(i) = output_shape_vector[output_shape_vector.size() - i - 1];
  }
  output()->set_alloc_size(size);
}

void StridedSlice::execute(luci::CircleReader *circle_reader, int32_t index) const
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

  circle::OperatorT oper_t;
  circle_reader->operators()[index]->UnPackTo(&oper_t);
  const auto *options = oper_t.builtin_options.AsStridedSliceOptions();

  tflite::StridedSliceParams op_params{};
  op_params.start_indices_count = input_tensor_shape.size(); //input()->shape().num_dims();
  op_params.stop_indices_count = input_tensor_shape.size();//input()->shape().num_dims();
  op_params.strides_count = input_tensor_shape.size();//input()->shape().num_dims();

  for (int i = 0; i < input_tensor_shape.size(); i++)
  {
    op_params.start_indices[i] = getTensorData<int32_t>(begin())[i];
    op_params.stop_indices[i] = getTensorData<int32_t>(end())[i];
    op_params.strides[i] = getTensorData<int32_t>(strides())[i];
  }
  op_params.begin_mask = options->begin_mask;
  op_params.ellipsis_mask = 0;
  op_params.end_mask = options->end_mask;
  op_params.new_axis_mask = 0;
  op_params.shrink_axis_mask = options->shrink_axis_mask;

  auto input_dtype = luci::luci_datatype(circle_reader->tensors()[input_ind()]->type());

  switch (input_dtype)
  {
    case DataType::FLOAT32:
      tflite::reference_ops::StridedSlice(op_params, input_runtime_shape,
                                          getTensorData<float>(input()), output_runtime_shape,
                                          getTensorData<float>(output()));
      break;
    case DataType::U8:
      tflite::reference_ops::StridedSlice(op_params, input_runtime_shape,
                                          getTensorData<uint8_t>(input()), output_runtime_shape,
                                          getTensorData<uint8_t>(output()));
      break;
    case DataType::S32:
      tflite::reference_ops::StridedSlice(op_params, input_runtime_shape,
                                          getTensorData<int32_t>(input()), output_runtime_shape,
                                          getTensorData<int32_t>(output()));
      break;
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

} // namespace kernels
} // namespace luci_interpreter
