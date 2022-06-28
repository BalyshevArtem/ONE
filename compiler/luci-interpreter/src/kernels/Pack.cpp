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

#include "kernels/Pack.h"
#include "kernels/Utils.h"

//#include <tensorflow/lite/kernels/internal/reference/reference_ops.h>

#include <stdexcept>

namespace luci_interpreter
{
namespace kernels
{
namespace
{
template <typename Scalar>
void Pack_impl(const tflite::PackParams& params, std::vector<tflite::RuntimeShape> &input_shapes,
          std::vector<const Scalar *> &input_data, const tflite::RuntimeShape& output_shape,
          Scalar* output_data) {
  //ruy::profiler::ScopeLabel label("Pack");
  const int dimensions = output_shape.DimensionsCount();
  int axis = params.axis;
  int inputs_count = params.inputs_count;

  int outer_size = 1;
  for (int i = 0; i < axis; i++) {
    outer_size *= output_shape.Dims(i);
  }
  int copy_size = 1;
  for (int i = params.axis + 1; i < dimensions; i++) {
    copy_size *= output_shape.Dims(i);
  }
  // TFLITE_DCHECK_EQ((**input_shapes).FlatSize(), copy_size * outer_size);

  for (int i = 0; i < inputs_count; ++i) {
    for (int k = 0; k < outer_size; k++) {
      const Scalar* input_ptr = input_data[i] + copy_size * k;
      int loc = k * inputs_count * copy_size + i * copy_size;
      memcpy(output_data + loc, input_ptr, copy_size * sizeof(Scalar));
    }
  }
}
}

Pack::Pack(std::vector<std::pair<const Tensor *, int32_t>> &&inputs, std::vector<std::pair<Tensor *, int32_t>> &&outputs)
  : Kernel(std::move(inputs), std::move(outputs))
{
}

void Pack::configure(luci::CircleReader *circle_reader, int32_t index)
{
//  LUCI_INTERPRETER_CHECK(_inputs.size() == static_cast<uint32_t>(params().values_count));
//  const Tensor *t0 = _inputs[0];
//  const int dimension_size = t0->shape().num_dims() + 1;
//  int axis = params().axis;
//  if (axis < 0)
//  {
//    axis += dimension_size;
//  }
//  LUCI_INTERPRETER_CHECK(axis >= 0 && axis <= t0->shape().num_dims());
//
//  if (t0->element_type() != DataType::S32 && t0->element_type() != DataType::FLOAT32 &&
//      t0->element_type() != DataType::U8 && t0->element_type() != DataType::S8 &&
//      t0->element_type() != DataType::S16 && t0->element_type() != DataType::S64)
//  {
//    throw std::runtime_error("Unsupported type.");
//  }
//
//  for (uint32_t i = 1; i < _inputs.size(); ++i)
//  {
//    const Tensor *tensor = _inputs[i];
//    LUCI_INTERPRETER_CHECK(tensor->element_type() == t0->element_type());
//    LUCI_INTERPRETER_CHECK(tensor->shape().num_dims() == t0->shape().num_dims());
//    for (int d = 0; d < t0->shape().num_dims(); ++d)
//    {
//      LUCI_INTERPRETER_CHECK(tensor->shape().dim(d) == t0->shape().dim(d));
//    }
//  }
//
//  Shape output_shape(dimension_size);
//  int i = 0;
//  for (int index = 0; index < dimension_size; ++index)
//  {
//    if (index == axis)
//    {
//      output_shape.dim(index) = params().values_count;
//    }
//    else
//    {
//      output_shape.dim(index) = t0->shape().dim(i++);
//    }
//  }
//
//  if (t0->element_type() == DataType::S32 || t0->element_type() == DataType::U8 ||
//      t0->element_type() == DataType::S8 || t0->element_type() == DataType::S16 ||
//      t0->element_type() == DataType::S64)
//  {
//    LUCI_INTERPRETER_CHECK(output()->zero_point() == t0->zero_point());
//    LUCI_INTERPRETER_CHECK(output()->scale() == t0->scale());
//    // Guarantee input/output quantization params match as we do not support
//    // packing quantized tensors.
//    for (int i = 0; i < params().values_count; i++)
//    {
//      LUCI_INTERPRETER_CHECK(_inputs[i]->zero_point() == t0->zero_point());
//      LUCI_INTERPRETER_CHECK(_inputs[i]->scale() == t0->scale());
//    }
//  }
//
//  output()->resize(output_shape);
}

void Pack::execute(luci::CircleReader *circle_reader, int32_t index) const
{
  auto input_dtype = luci::luci_datatype(circle_reader->tensors()[input_ind(0)]->type());

  switch (input_dtype)
  {
    case DataType::FLOAT32:
      evalGeneric<float>(circle_reader, index);
      break;
    case DataType::U8:
      evalGeneric<uint8_t>(circle_reader, index);
      break;
    case DataType::S8:
      evalGeneric<int8_t>(circle_reader, index);
      break;
    case DataType::S16:
      evalGeneric<int16_t>(circle_reader, index);
      break;
    case DataType::S32:
      evalGeneric<int32_t>(circle_reader, index);
      break;
    case DataType::S64:
      evalGeneric<int64_t>(circle_reader, index);
      break;
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

template <typename T> void Pack::evalGeneric(luci::CircleReader *circle_reader, int32_t index) const
{
//  const Tensor *t0 = _inputs[0];
//  const int dimension_size = t0->shape().num_dims() + 1;
//  int axis = params().axis;
//  if (axis < 0)
//  {
//    axis += dimension_size;
//  }

  circle::OperatorT oper_t;
  circle_reader->operators()[index]->UnPackTo(&oper_t);
  const auto *options = oper_t.builtin_options.AsPackOptions();
  int axis = options->axis;
  const auto t0_shape = luci::wrap(circle_reader->tensors()[_inputs[0].second]->shape());
  const int dimension_size = t0_shape.size() + 1;
  if (axis < 0)
  {
    axis += dimension_size;
  }

  std::vector<tflite::RuntimeShape> inputs_shapes;
  std::vector<const T *> all_data;

  for (uint32_t i = 0; i < _inputs.size(); ++i)
  {
    const auto input_tensor_shape = luci::wrap(circle_reader->tensors()[_inputs[i].second]->shape());
    tflite::RuntimeShape input_runtime_shape(input_tensor_shape.size());
    for (int j = 0; j < input_tensor_shape.size(); ++j)
    {
      input_runtime_shape.SetDim(j, input_tensor_shape[j]);
    }
    inputs_shapes.push_back(input_runtime_shape);

    all_data.push_back(getTensorData<T>(_inputs.at(i).first));
  }

  const auto output_tensor_shape = luci::wrap(circle_reader->tensors()[output_ind()]->shape());
  tflite::RuntimeShape output_runtime_shape(output_tensor_shape.size());
  for (int i = 0; i < output_tensor_shape.size(); ++i)
  {
    output_runtime_shape.SetDim(i, output_tensor_shape[i]);
  }

  //VectorOfTensors<T, true> inputs(_inputs);
  tflite::PackParams params{};
  params.axis = axis;
  params.inputs_count = _inputs.size();
  Pack_impl<T>(params, inputs_shapes, all_data, output_runtime_shape,
                                 getTensorData<T>(output()));
}

} // namespace kernels
} // namespace luci_interpreter
