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

class KernelConfigureRegistry;
class KernelExecuteRegistry;

class KernelBuilder
{
public:
  KernelBuilder();

  ~KernelBuilder();


  void configure_kernel(std::vector<const Tensor *> &inputs,
                        std::vector<Tensor *> &output,
                        circle::BuiltinOperator opcode, int32_t op_index,
                        luci_interpreter::CircleReader *circle_reader);

  void execute_kernel(std::vector<const Tensor *> &inputs,
                        std::vector<Tensor *> &output,
                        circle::BuiltinOperator opcode, int32_t op_index,
                        luci_interpreter::CircleReader *circle_reader,
                        bool is_inplace);


private:
  std::unique_ptr<KernelConfigureRegistry> _configure_registry;
  std::unique_ptr<KernelExecuteRegistry> _execute_registry;
};

} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_KERNEL_KERNELBUILDER_H
