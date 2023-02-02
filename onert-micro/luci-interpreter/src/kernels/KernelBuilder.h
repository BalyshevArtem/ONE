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

#ifndef LUCI_INTERPRETER_KERNEL_KERNELBUILDER_H
#define LUCI_INTERPRETER_KERNEL_KERNELBUILDER_H

//#include "core/Kernel.h"
#include "core/RuntimeGraph.h"

#include "luci_interpreter/core/reader/CircleMicroReader.h"

#include <memory>
#include <unordered_map>

namespace luci_interpreter
{

class KernelConfigureRegistry
{
public:
  using KernelConfigureFunc = void(const circle::Operator *,
    IBaseRuntimeGraph *);

  KernelConfigureRegistry();

  void configure_kernel(const circle::Operator *cur_op,
                        circle::BuiltinOperator opcode,
                        IBaseRuntimeGraph *runtime_graph);

private:
  std::unordered_map<int32_t, KernelConfigureFunc *> _operator_configure;

private:
  KernelConfigureFunc *get_kernel_configure_func(circle::BuiltinOperator opcode) const
  {
    return _operator_configure.at(size_t(opcode));
  }

  void register_kernel_configure(circle::BuiltinOperator id, KernelConfigureFunc *func)
  {
    _operator_configure[size_t(id)] = func;
  }
};

class KernelExecuteRegistry
{
public:
  using KernelExecuteFunc = void(const circle::Operator *, IBaseRuntimeGraph *, bool);

  KernelExecuteRegistry();

  void execute_kernel(const circle::Operator *cur_op,
                      circle::BuiltinOperator opcode,
                      IBaseRuntimeGraph *runtime_graph, bool is_inplace);

private:
  std::unordered_map<int32_t, KernelExecuteFunc *> _operator_execute;

private:
  KernelExecuteFunc *get_kernel_execute_func(circle::BuiltinOperator opcode) const
  {
    return _operator_execute.at(size_t(opcode));
  }

  void register_kernel_execute(circle::BuiltinOperator id, KernelExecuteFunc *func)
  {
    _operator_execute[size_t(id)] = func;
  }
};

//class KernelConfigureRegistry;
//class KernelExecuteRegistry;

//class KernelBuilder
//{
//public:
//  KernelBuilder() = default;
//
//  ~KernelBuilder();
//
//
//  void configure_kernel(std::vector<const Tensor *> &inputs,
//                        std::vector<Tensor *> &output,
//                        circle::BuiltinOperator opcode, int32_t op_index,
//                        luci_interpreter::CircleReader *circle_reader);
//
//  void execute_kernel(std::vector<const Tensor *> &inputs,
//                        std::vector<Tensor *> &output,
//                        circle::BuiltinOperator opcode, int32_t op_index,
//                        luci_interpreter::CircleReader *circle_reader,
//                        bool is_inplace);
//
//
//private:
//  KernelConfigureRegistry _configure_registry{};
//  KernelExecuteRegistry _execute_registry{};
//};

} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_KERNEL_KERNELBUILDER_H
