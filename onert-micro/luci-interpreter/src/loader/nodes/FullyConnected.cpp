///*
// * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
// *
// * Licensed under the Apache License, Version 2.0 (the "License");
// * you may not use this file except in compliance with the License.
// * You may obtain a copy of the License at
// *
// *    http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing, software
// * distributed under the License is distributed on an "AS IS" BASIS,
// * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// * See the License for the specific language governing permissions and
// * limitations under the License.
// */
//
//#include "Builders.h"
//
////#include "kernels/FullyConnected.h"
//
//namespace luci_interpreter
//{
//
//void configure_kernel_CircleFullyConnected(std::vector<const Tensor *> &inputs,
//                                           std::vector<Tensor *> &outputs,
//                                           const uint32_t op_index,
//                                           KernelBuilder &builder)
//{
//  assert(inputs.size() == 0);
//  const auto input = inputs.at(0);
//  const auto weights = inputs.at(1);
//  const auto bias = inputs.at(2);
//
//  auto output = outputs.at(0);
//
//  if (weights->element_type() == DataType::U8)
//  {
//    LUCI_INTERPRETER_CHECK(input->element_type() == DataType::U8);
//    LUCI_INTERPRETER_CHECK(output->element_type() == DataType::U8);
//    LUCI_INTERPRETER_CHECK(!bias || bias->element_type() == DataType::S32)
//  }
//  else if (weights->element_type() == DataType::FLOAT32)
//  {
//    LUCI_INTERPRETER_CHECK(input->element_type() == DataType::FLOAT32);
//    LUCI_INTERPRETER_CHECK(output->element_type() == DataType::FLOAT32);
//    LUCI_INTERPRETER_CHECK(!bias || bias->element_type() == DataType::FLOAT32)
//  }
//  else if (weights->element_type() == DataType::S8)
//  {
//    LUCI_INTERPRETER_CHECK(input->element_type() == DataType::S8);
//    LUCI_INTERPRETER_CHECK(output->element_type() == DataType::S8);
//    LUCI_INTERPRETER_CHECK(!bias || bias->element_type() == DataType::S32)
//  }
//  else
//  {
//    assert(false && "Unsupported type.");
//  }
//
//  //const Shape &input_shape = input()->shape();
//  //const Shape &weights_shape = weights()->shape();
//
//  LUCI_INTERPRETER_CHECK(weights->num_dims() == 2);//weights_shape.num_dims() == 2);
//  LUCI_INTERPRETER_CHECK(bias == nullptr ||
//  bias->num_elements() == weights->dim(0));
//  //bias()->shape().num_elements() == weights_shape.dim(0));
//
//  LUCI_INTERPRETER_CHECK(input->num_elements() % weights()->dim(1) == 0);
//  const int32_t batch_size = input->num_elements() / weights->dim(1);
//  const int32_t num_units = weights->dim(0);
//
//  if (bias)
//    LUCI_INTERPRETER_CHECK(bias->num_elements() == weights->dim(0));
//
//  // TODO: enable it only if kernel with dynamic shapes
//  circle::OperatorT oper_t;
//  builder.get_circle_reader()->operators()[op_index]->UnPackTo(&oper_t);
//  const auto *options = oper_t.builtin_options.AsFullyConnectedOptions();
//
//  if (!options->keep_num_dims)
//  {
//    //output()->resize({batch_size, num_units});
//  }
//  else
//  {
//    //    luci_interpreter::Shape output_shape(input_shape.num_dims());
//    //    for (int i = 0; i < input_shape.num_dims(); ++i)
//    //      output_shape.dim(i) = input_shape.dim(i);
//    //    output_shape.dim(input_shape.num_dims() - 1) = num_units;
//    //    output()->resize(output_shape);
//  }
//}
//
////std::unique_ptr<Kernel> build_kernel_CircleFullyConnected(std::vector<const Tensor *> &&inputs,
////                                                          std::vector<Tensor *> &&outputs,
////                                                          const uint32_t op_index,
////                                                          KernelBuilder &builder)
////{
////  assert(inputs.size() == 3);
////
////  const Tensor *input = inputs.at(0);
////  const Tensor *weights = inputs.at(1);
////  const Tensor *bias = inputs.at(2);
////  Tensor *output = outputs.at(0);
////
////  circle::OperatorT oper_t;
////  builder.get_circle_reader()->operators()[op_index]->UnPackTo(&oper_t);
////  const auto *options = oper_t.builtin_options.AsFullyConnectedOptions();
////
////  FullyConnectedParams params{};
////  params.activation = luci_actfunc(options->fused_activation_function);
////  params.keep_num_dims = options->keep_num_dims;
////
////  return std::make_unique<kernels::FullyConnected>(input, weights, bias, output, params);
////}
//
//} // namespace luci_interpreter
