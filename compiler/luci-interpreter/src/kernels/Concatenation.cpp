/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/Concatenation.h"
#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/reference/concatenation.h>

#include <stdexcept>

namespace luci_interpreter
{
namespace kernels
{

Concatenation::Concatenation(std::vector<std::pair<const Tensor *, int32_t>> &&inputs, std::vector<std::pair<Tensor *, int32_t>> &&outputs)
  : Kernel(std::move(inputs), std::move(outputs))
{
}

void Concatenation::configure(luci::CircleReader *circle_reader, int32_t index)
{
//  const int num_inputs = _inputs.size();
//  LUCI_INTERPRETER_CHECK(num_inputs > 0);
//  const Tensor *t0 = _inputs[0].first;

  // TODO: Support concat with fused activation function
  //LUCI_INTERPRETER_CHECK(params().activation == luci::FusedActFunc::NONE);

//  circle::OperatorT oper_t;
//  circle_reader->operators()[index]->UnPackTo(&oper_t);
//  const auto *options = oper_t.builtin_options.AsConcatenationOptions();
//
//  int axis = options->axis;
//  if (axis < 0)
//    axis += t0->shape().num_dims();
//  LUCI_INTERPRETER_CHECK(axis >= 0 && axis < t0->shape().num_dims());
//
//  int32_t sum_axis = t0->shape().dim(axis);
//  for (int i = 1; i < num_inputs; ++i)
//  {
//    const Tensor *tensor = _inputs[i];
//    LUCI_INTERPRETER_CHECK(tensor->element_type() == t0->element_type());
//    LUCI_INTERPRETER_CHECK(tensor->shape().num_dims() == t0->shape().num_dims());
//    for (int d = 0; d < t0->shape().num_dims(); ++d)
//    {
//      if (d == axis)
//      {
//        sum_axis += tensor->shape().dim(axis);
//      }
//      else
//      {
//        LUCI_INTERPRETER_CHECK(tensor->shape().dim(d) == t0->shape().dim(d));
//      }
//    }
//  }
//
//  Shape output_shape = t0->shape();
//  output_shape.dim(axis) = sum_axis;
//
//  // If input tensors are INT8 type then quantization parameters of all input tensors and the output
//  // should be the same
//  for (auto current_tensor : _inputs)
//  {
//    if (current_tensor->element_type() == DataType::S8)
//    {
//      LUCI_INTERPRETER_CHECK(current_tensor->quantized_dimension() ==
//                             output()->quantized_dimension());
//
//      LUCI_INTERPRETER_CHECK(current_tensor->zero_points().size() ==
//                             current_tensor->scales().size());
//      LUCI_INTERPRETER_CHECK(current_tensor->zero_points() == output()->zero_points());
//      LUCI_INTERPRETER_CHECK(current_tensor->scales() == output()->scales());
//    }
//  }
//  output()->resize(output_shape);
}

void Concatenation::execute(luci::CircleReader *circle_reader, int32_t index) const
{
  auto input_dtype = luci::luci_datatype(circle_reader->tensors()[input_ind()]->type());
  switch (input_dtype)
  {
    case DataType::FLOAT32:
    {
      evalGeneric<float>(circle_reader, index);
      break;
    }
//    case DataType::U8:
//      evalQuantized();
//      break;
//    case DataType::S8:
//      evalGeneric<int8_t>();
//      break;
//    case DataType::S32:
//      evalGeneric<int32_t>();
//      break;
//    case DataType::S64:
//      evalGeneric<int64_t>();
//      break;
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

template <typename T> void Concatenation::evalGeneric(luci::CircleReader *circle_reader, int32_t index) const
{
  circle::OperatorT oper_t;
  circle_reader->operators()[index]->UnPackTo(&oper_t);
  const auto *options = oper_t.builtin_options.AsConcatenationOptions();

  const auto output_tensor_shape = luci::wrap(circle_reader->tensors()[output_ind()]->shape());
  tflite::RuntimeShape output_runtime_shape(output_tensor_shape.size());
  for (int i = 0; i < output_tensor_shape.size(); ++i)
  {
    output_runtime_shape.SetDim(i, output_tensor_shape[i]);
  }

  int axis = options->axis;
  if (axis < 0)
    axis += output_tensor_shape.size(); //output()->shape().num_dims();

  //VectorOfTensors<T, true> inputs(_inputs);

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

  tflite::ConcatenationParams params{};
  params.axis = axis;
  params.inputs_count = _inputs.size();
  throw std::runtime_error("FIX Concta here\n");
 // tflite::reference_ops::Concatenation(params, inputs_shapes, all_data.data(),
 //                                      output_runtime_shape, getTensorData<T>(output()));
}
//
//void Concatenation::evalQuantized() const
//{
//  int axis = _params.axis;
//  if (axis < 0)
//    axis += output()->shape().num_dims();
//
//  VectorOfQuantizedTensors<true> inputs(_inputs);
//  tflite::ConcatenationParams params{};
//  params.axis = axis;
//  params.input_zeropoint = inputs.zero_point();
//  params.input_scale = inputs.scale();
//  params.inputs_count = _inputs.size();
//  params.output_zeropoint = output()->zero_point();
//  params.output_scale = output()->scale();
//
//  tflite::reference_ops::ConcatenationWithScaling(params, inputs.shapes(), inputs.data(),
//                                                  getTensorShape(output()),
//                                                  getTensorData<uint8_t>(output()));
//}

} // namespace kernels
} // namespace luci_interpreter
