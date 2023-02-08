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

#include "Builders.h"
#include "kernels/Utils.h"

namespace luci_interpreter
{

void configure_kernel_CircleExpandDims(const circle::Operator *cur_op,
                                      IBaseRuntimeGraph *runtime_graph)
{
  const auto input_index = cur_op->inputs()->operator[](0);
  const auto axis_index = cur_op->inputs()->operator[](1);
  const auto output_index = cur_op->outputs()->operator[](0);

  assert(input_index != -1);
  assert(output_index != -1);

  const auto input = runtime_graph->getCircleTensorByIndex(input_index);
  auto output = runtime_graph->getCircleTensorByIndex(output_index);

  assert(input != nullptr);
  assert(output != nullptr);

  const auto axis = runtime_graph->getCircleTensorByIndex(axis_index);
  auto axis_data = runtime_graph->getDataBufferFromCircleTensor(axis);

  int32_t axis_value;

  switch (Tensor::element_type(axis))
  {
    case DataType::S32:
      axis_value = *reinterpret_cast<int32_t *>(axis_data);
      break;
    case DataType::S64:
      axis_value = static_cast<int32_t>(*reinterpret_cast<int64_t *>(axis_data));
      break;
    default:
      assert(false && "Unsupported type.");
  }

  if (axis_value < 0)
  {
    axis_value += Tensor::num_dims(input) + 1;
  }

  LUCI_INTERPRETER_CHECK(axis_value <= Tensor::num_dims(input) and axis_value >= 0);

  // TODO: check needles of it
//  Shape output_shape(input_shape.num_dims() + 1);
//  for (int32_t i = 0; i < output_shape.num_dims(); ++i)
//  {
//    if (i < axis_value)
//    {
//      output_shape.dim(i) = input_shape.dim(i);
//    }
//    else if (i == axis_value)
//    {
//      output_shape.dim(i) = 1;
//    }
//    else
//    {
//      LUCI_INTERPRETER_CHECK(i >= 1);
//      output_shape.dim(i) = input_shape.dim(i - 1);
//    }
//  }
//
//  // TODO: enable it only if kernel with dynamic shapes
//  output()->resize(output_shape);
}

void execute_kernel_CircleExpandDims(const circle::Operator *cur_op,
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
    runtime_graph->setNullDataByCircleTensor(input, output);
    return;
  }

  // Just copy input to output
  auto input_data = (runtime_graph->getDataByCircleTensor(input));
  auto output_data = (runtime_graph->getDataByCircleTensor(output));

  assert(input_data != nullptr);
  assert(output_data != nullptr);

  const size_t element_size = getDataTypeSize(Tensor::element_type(input));
  const int32_t num_elements = Tensor::num_elements(input);
  std::memcpy(output_data, input_data, num_elements * element_size);
}

//ExpandDims::ExpandDims(const Tensor *input, const Tensor *axis, Tensor *output)
//  : Kernel({input, axis}, {output})
//{
//}

//void ExpandDims::configure()
//{
////  int32_t axis_value;
////
////  switch (axis()->element_type())
////  {
////    case DataType::S32:
////      axis_value = *getTensorData<int32_t>(axis());
////      break;
////    case DataType::S64:
////      axis_value = static_cast<int32_t>(*getTensorData<int64_t>(axis()));
////      break;
////    default:
////      assert(false && "Unsupported type.");
////  }
////
////  const auto input_shape = input()->shape();
////
////  if (axis_value < 0)
////  {
////    axis_value += input_shape.num_dims() + 1;
////  }
////
////  LUCI_INTERPRETER_CHECK(axis_value <= input_shape.num_dims() and axis_value >= 0);
////
////  Shape output_shape(input_shape.num_dims() + 1);
////  for (int32_t i = 0; i < output_shape.num_dims(); ++i)
////  {
////    if (i < axis_value)
////    {
////      output_shape.dim(i) = input_shape.dim(i);
////    }
////    else if (i == axis_value)
////    {
////      output_shape.dim(i) = 1;
////    }
////    else
////    {
////      LUCI_INTERPRETER_CHECK(i >= 1);
////      output_shape.dim(i) = input_shape.dim(i - 1);
////    }
////  }
////
////  // TODO: enable it only if kernel with dynamic shapes
////  output()->resize(output_shape);
//}
//
//void ExpandDims::execute() const
//{
//  if (_is_inplace)
//  {
//    auto input_tensor = const_cast<Tensor *>(input());
//    output()->set_data_buffer(input_tensor->data<uint8_t>());
//    input_tensor->set_data_buffer(nullptr);
//    return;
//  }
//
//  // Just copy input to output
//  const auto *input_data = input()->data<void>();
//  auto *output_data = output()->data<void>();
//
//  const size_t element_size = getDataTypeSize(input()->element_type());
//  const int32_t num_elements = input()->shape().num_elements();
//  std::memcpy(output_data, input_data, num_elements * element_size);
//}

} // namespace luci_interpreter
