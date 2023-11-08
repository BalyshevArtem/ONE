/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef ONERT_MICRO_CORE_KERNEL_TYPE_H
#define ONERT_MICRO_CORE_KERNEL_TYPE_H

namespace onert_micro
{
namespace core
{

enum OMKernelType
{
  Normal,
  WeightQuantize,
  FullQuantize,
  MixedFloatToQ8,
  MixedQ8ToFloat,
  MixedFloatToQ16,
  MixedQ16ToFloat,
};

#define REGISTER_KERNEL(builtin_operator, name) BuiltinOperator_##builtin_operator,
#define REGISTER_CUSTOM_KERNEL(name) BuiltinOperator_##name,
enum class OMBuilderID
{
#include "KernelsToBuild.lst"
  BuiltinOperatorSize,
#include "CustomKernelsToBuild.lst"
  Size // casts to count of values in BuilderId enum
};
#undef REGISTER_CUSTOM_KERNEL
#undef REGISTER_KERNEL

constexpr OMStatus getBuiltinOperatorBuilderId(circle::BuiltinOperator opcode, OMBuilderID &builderID)
{
  switch (opcode)
  {
#define REGISTER_KERNEL(builtin_operator, name)    \
  case circle::BuiltinOperator_##builtin_operator: \
    builderID = OMBuilderID::BuiltinOperator_##builtin_operator; \
    break;
#if USE_GENERATED_LIST
#include "GeneratedKernelsToBuild.lst"
#else
#include "KernelsToBuild.lst"
#endif

#undef REGISTER_KERNEL
    default:
      assert(false && "Unsupported operation");
      return UnsupportedOp;
  }
  return Ok;
}

class KernelConfigureRegistry
{
public:
  using KernelConfigureFunc = void(const circle::Operator *, OMRuntimeContext *);

  constexpr KernelConfigureRegistry() : _operator_configure()
  {
#define REGISTER_KERNEL(builtin_operator, name)                            \
  register_kernel_configure(OMBuilderID::BuiltinOperator_##builtin_operator, \
                            configure_kernel_Circle##name);

#include "KernelsToBuild.lst"

#undef REGISTER_KERNEL
  }

  void configure_kernel(const circle::Operator *cur_op, circle::BuiltinOperator opcode,
                        OMRuntimeContext *runtime_graph) const;

private:
  constexpr KernelConfigureFunc *get_kernel_configure_func(circle::BuiltinOperator opcode) const
  {
    const auto builder_id_opcode = size_t(get_builder_id(opcode));
    assert(builder_id_opcode < size_t(BuilderID::Size));
    return _operator_configure[builder_id_opcode];
  }

  constexpr void register_kernel_configure(BuilderID id, KernelConfigureFunc *func)
  {
    assert(size_t(id) < size_t(OMBuilderID::Size));
    _operator_configure[size_t(id)] = func;
  }

private:
  KernelConfigureFunc *_operator_configure[size_t(OMBuilderID::Size)];
};

//
//constexpr OMStatus getCustomOperatorBuilderId(const std::string custom_operator_name, OMBuilderID &builderID)
//{
//  switch (custom_operator_name)
//  {
//
//    default:
//      assert(false && "Unsupported operation");
//  }
//  return Ok;
//}

} // core
} // namespace onert_micro

#endif // ONERT_MICRO_CORE_KERNEL_TYPE_H
