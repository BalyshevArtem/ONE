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

#include "Builders.h"

#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/reference/logistic.h>

namespace luci_interpreter
{
namespace
{

//
// Logistic::Logistic(const Tensor *input, Tensor *output) : Kernel({input}, {output}) {}
//
// void Logistic::configure()
//{
//  LUCI_INTERPRETER_CHECK(input()->element_type() == output()->element_type());
//
//#ifndef DIS_QUANT
//  if (input()->element_type() == DataType::U8)
//  {
//    LUCI_INTERPRETER_CHECK(output()->scale() == 1. / 256);
//  }
//#endif
//  // TODO: enable it only if kernel with dynamic shapes
//  //output()->resize(input()->shape());
//}
//
// void Logistic::execute() const
//{
//  switch (input()->element_type())
//  {
//    case DataType::FLOAT32:
//      evalFloat();
//      break;
//#ifndef DIS_QUANT
//    case DataType::U8:
//      evalQuantized();
//      break;
//#endif
//    default:
//      assert(false && "Unsupported type.");
//  }
//}
//
// void Logistic::evalFloat() const
//{
//  if (_is_inplace)
//    output()->set_data_buffer(const_cast<uint8_t *>(input()->data<uint8_t>()));
//
//  tflite::reference_ops::Logistic(getTensorShape(input()), getTensorData<float>(input()),
//                                  getTensorShape(output()), getTensorData<float>(output()));
//  if (_is_inplace)
//  {
//    auto input_tensor = const_cast<Tensor *>(input());
//    input_tensor->set_data_buffer(nullptr);
//  }
//}
//
//#ifndef DIS_QUANT
// void Logistic::evalQuantized() const
//{
//  if (_is_inplace)
//    output()->set_data_buffer(const_cast<uint8_t *>(input()->data<uint8_t>()));
//
//  tflite::reference_ops::Logistic(getTensorShape(input()), getTensorData<int8_t>(input()),
//                                  input()->scale(), input()->zero_point(),
//                                  getTensorShape(output()), getTensorData<int8_t>(output()),
//                                  output()->scale(), output()->zero_point());
//  if (_is_inplace)
//  {
//    auto input_tensor = const_cast<Tensor *>(input());
//    input_tensor->set_data_buffer(nullptr);
//  }
//}
//#endif

void evalFloat(const Tensor *input, Tensor *output, bool is_inplace)
{
  if (is_inplace)
    output->set_data_buffer(const_cast<uint8_t *>(input->data<uint8_t>()));

  tflite::reference_ops::Logistic(kernels::getTensorShape(input), kernels::getTensorData<float>(input),
                                  kernels::getTensorShape(output), kernels::getTensorData<float>(output));
  if (is_inplace)
  {
    auto input_tensor = const_cast<Tensor *>(input);
    input_tensor->set_data_buffer(nullptr);
  }
}

#ifndef DIS_QUANT
void evalQuantized() const
{
  if (_is_inplace)
    output()->set_data_buffer(const_cast<uint8_t *>(input()->data<uint8_t>()));

  tflite::reference_ops::Logistic(getTensorShape(input()), getTensorData<int8_t>(input()),
                                  input()->scale(), input()->zero_point(), getTensorShape(output()),
                                  getTensorData<int8_t>(output()), output()->scale(),
                                  output()->zero_point());
  if (_is_inplace)
  {
    auto input_tensor = const_cast<Tensor *>(input());
    input_tensor->set_data_buffer(nullptr);
  }
}
#endif

} // namespace

// TODO think how remove unused param
void configure_kernel_CircleLogistic(const circle::Operator *cur_op,
                                     IBaseRuntimeGraph *runtime_graph)
{
  const auto input_index = cur_op->inputs()->operator[](0);
  const auto output_index = cur_op->outputs()->operator[](0);

  assert(input_index != -1);
  assert(output_index != -1);

  const auto input = runtime_graph->getTensorByIndex(input_index);
  auto output = runtime_graph->getTensorByIndex(output_index);

  assert(input != nullptr);
  assert(output != nullptr);

  LUCI_INTERPRETER_CHECK(input->element_type() == output->element_type());

#ifndef DIS_QUANT
  if (input()->element_type() == DataType::U8)
  {
    LUCI_INTERPRETER_CHECK(output()->scale() == 1. / 256);
  }
#endif
  // TODO: enable it only if kernel with dynamic shapes
  // output()->resize(input()->shape());
}

// TODO think how remove unused param
void execute_kernel_CircleLogistic(const circle::Operator *cur_op,
                                   IBaseRuntimeGraph *runtime_graph,
                                   bool is_inplace)
{
  const auto input_index = cur_op->inputs()->operator[](0);
  const auto output_index = cur_op->outputs()->operator[](0);

  assert(input_index != -1);
  assert(output_index != -1);

  const auto input = runtime_graph->getTensorByIndex(input_index);
  auto output = runtime_graph->getTensorByIndex(output_index);

  assert(input != nullptr);
  assert(output != nullptr);

  switch (input->element_type())
  {
    case DataType::FLOAT32:
      evalFloat(input, output, is_inplace);
      break;
#ifndef DIS_QUANT
    case DataType::U8:
      evalQuantized();
      break;
#endif
    default:
      assert(false && "Unsupported type.");
  }
}

} // namespace luci_interpreter
