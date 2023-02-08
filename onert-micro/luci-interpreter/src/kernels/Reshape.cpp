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

#include "Builders.h"

#include <cassert>
#include <cstring>

namespace luci_interpreter
{


void configure_kernel_CircleReshape(const circle::Operator *cur_op,
                                       IBaseRuntimeGraph *runtime_graph)
{
  //TODO add this part
}


void execute_kernel_CircleReshape(const circle::Operator *cur_op,
                                     IBaseRuntimeGraph *runtime_graph,
                                     bool is_inplace)
{
  const auto input_index = cur_op->inputs()->operator[](0);
  const auto output_index = cur_op->outputs()->operator[](0);

  assert(input_index != -1);
  assert(output_index != -1);

  const auto input = runtime_graph->getCircleTensorByIndex(input_index);

  auto output = runtime_graph->getCircleTensorByIndex(output_index);

  if (is_inplace)
  {
    //output->set_data_buffer(const_cast<uint8_t *>(input_tensor->data<const uint8_t>()));
    runtime_graph->setNullDataByCircleTensor(input, output);
    //input_tensor->set_data_buffer(nullptr);
    return;
  }

  auto input_data = (runtime_graph->getDataByCircleTensor(input));
  auto output_data = (runtime_graph->getDataByCircleTensor(output));

  assert(input_data != nullptr);
  assert(output_data != nullptr);

  const size_t element_size = getDataTypeSize(Tensor::element_type(input));
  const int32_t num_elements = Tensor::num_elements(input);
  std::memcpy(output_data, input_data, num_elements * element_size);
}

//static Shape extractShapeFromTensor(const Tensor *tensor)
//{
//  assert(tensor->element_type() == DataType::S32);
//  Shape shape(tensor->shape().num_elements());
//  const auto *shape_data = tensor->data<int32_t>();
//  for (int i = 0; i < tensor->shape().num_elements(); ++i)
//  {
//    shape.dim(i) = shape_data[i];
//  }
//  // TODO: enable it only if kernel with dynamic shapes
//  return shape;
//}
//
//static void resolveUnknownDimension(const Shape &input_shape, Shape *output_shape)
//{
//  const int32_t num_input_elements = input_shape.num_elements();
//  int32_t num_output_elements = 1;
//  int unknown_dim_index = -1;
//  for (int i = 0; i < output_shape->num_dims(); ++i)
//  {
//    const int32_t value = output_shape->dim(i);
//    if (value == -1)
//    {
//      assert(unknown_dim_index == -1);
//      unknown_dim_index = i;
//    }
//    else
//    {
//      num_output_elements *= value;
//    }
//  }
//  if (unknown_dim_index != -1)
//  {
//    output_shape->dim(unknown_dim_index) = num_input_elements / num_output_elements;
//    num_output_elements *= output_shape->dim(unknown_dim_index);
//  }
//  assert(num_output_elements == num_input_elements);
//}
//
//Reshape::Reshape(const Tensor *input, const Tensor *shape, Tensor *output)
//  : Kernel({input, shape}, {output})
//{
//}
//
//void Reshape::configure()
//{
//  Shape output_shape = extractShapeFromTensor(shape());
//  resolveUnknownDimension(input()->shape(), &output_shape);
//  output()->resize(output_shape);
//}

//void Reshape::execute() const
//{
//  if (_is_inplace)
//  {
//    auto input_tensor = const_cast<Tensor *>(input());
//    output()->set_data_buffer(const_cast<uint8_t *>(input_tensor->data<const uint8_t>()));
//    input_tensor->set_data_buffer(nullptr);
//    return;
//  }
//
//  const auto *input_data = input()->data<void>();
//  auto *output_data = output()->data<void>();
//
//  const size_t element_size = getDataTypeSize(input()->element_type());
//  const int32_t num_elements = input()->shape().num_elements();
//  std::memcpy(output_data, input_data, num_elements * element_size);
//}

} // namespace luci_interpreter
