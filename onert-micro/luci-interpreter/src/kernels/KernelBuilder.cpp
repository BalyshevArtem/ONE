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

#include "KernelBuilder.h"
#include "Builders.h"

namespace luci_interpreter
{
enum class BuilderId
{
#define REGISTER_KERNEL(builtin_operator, name) Circle##name,
#if USE_GENERATED_LIST
#include "GeneratedKernelsToBuild.lst"
#else
#include "KernelsToBuild.lst"
#endif
  Size // casts to count of values in BuilderId enum
};
#undef REGISTER_KERNEL

/**
 * @brief Registry of kernel builders
 *
 * This class contains mapping from Opcodes to kernel builder functions
 */

class KernelConfigureRegistry
{
public:
  //  using KernelBuilderFunc = std::unique_ptr<Kernel>(std::vector<const Tensor *> &&,
  //                                                    std::vector<Tensor *> &&, const uint32_t,
  //                                                    KernelBuilder &);

  using KernelConfigureFunc = void(std::vector<const Tensor *> &, std::vector<Tensor *> &,
                                   const uint32_t, luci_interpreter::CircleReader *);

  KernelConfigureRegistry()
  {
#define REGISTER_KERNEL(builtin_operator, name)                                          \
  register_kernel_configure(circle::BuiltinOperator::BuiltinOperator_##builtin_operator, \
                            configure_kernel_Circle##name);

#if USE_GENERATED_LIST
#include "GeneratedKernelsToBuild.lst"
#else
#include "KernelsToBuild.lst"
#endif

#undef REGISTER_KERNEL
  }

  KernelConfigureFunc *get_kernel_configure_func(circle::BuiltinOperator opcode) const
  {
    return _operator_configure.at(size_t(opcode));
  }

private:
  std::map<int32_t, KernelConfigureFunc *> _operator_configure;

  void register_kernel_configure(circle::BuiltinOperator id, KernelConfigureFunc *func)
  {
    _operator_configure[size_t(id)] = func;
  }
};

class KernelExecuteRegistry
{
public:
  //  using KernelBuilderFunc = std::unique_ptr<Kernel>(std::vector<const Tensor *> &&,
  //                                                    std::vector<Tensor *> &&, const uint32_t,
  //                                                    KernelBuilder &);

  using KernelExecuteFunc = void(std::vector<const Tensor *> &, std::vector<Tensor *> &,
                                 const uint32_t, luci_interpreter::CircleReader *,
                                 bool);

  KernelExecuteRegistry()
  {
#define REGISTER_KERNEL(builtin_operator, name)                                        \
  register_kernel_execute(circle::BuiltinOperator::BuiltinOperator_##builtin_operator, \
                          execute_kernel_Circle##name);

#if USE_GENERATED_LIST
#include "GeneratedKernelsToBuild.lst"
#else
#include "KernelsToBuild.lst"
#endif

#undef REGISTER_KERNEL
  }

  KernelExecuteFunc *get_kernel_execute_func(circle::BuiltinOperator opcode) const
  {
    return _operator_execute.at(size_t(opcode));
  }

private:
  std::map<int32_t, KernelExecuteFunc *> _operator_execute;

  void register_kernel_execute(circle::BuiltinOperator id, KernelExecuteFunc *func)
  {
    _operator_execute[size_t(id)] = func;
  }
};


KernelBuilder::KernelBuilder():
_configure_registry(std::make_unique<KernelConfigureRegistry>()),
_execute_registry(std::make_unique<KernelExecuteRegistry>())
{
}

KernelBuilder::~KernelBuilder()
{
  // Need to define in this CPP to hide KernelBuilderRegistry internals.
  // This destructor deletes _builder_registry
}

void KernelBuilder::configure_kernel(std::vector<const Tensor *> &inputs,
                                     std::vector<Tensor *> &outputs,
                                     circle::BuiltinOperator opcode,
                                     int32_t op_index,
                                     luci_interpreter::CircleReader *circle_reader)
{
  auto specific_configure_func = _configure_registry->get_kernel_configure_func(opcode);
  if (specific_configure_func == nullptr)
    assert(false && "Unsupported operator");

  specific_configure_func(inputs, outputs, op_index, circle_reader);
}

void KernelBuilder::execute_kernel(std::vector<const Tensor *> &inputs,
                                     std::vector<Tensor *> &outputs,
                                     circle::BuiltinOperator opcode,
                                     int32_t op_index,
                                     luci_interpreter::CircleReader *circle_reader,
                                     bool is_inplace)
{
  auto specific_execute_func = _execute_registry->get_kernel_execute_func(opcode);
  if (specific_execute_func == nullptr)
    assert(false && "Unsupported operator");

  specific_execute_func(inputs, outputs, op_index, circle_reader, is_inplace);
}

} // namespace luci_interpreter
