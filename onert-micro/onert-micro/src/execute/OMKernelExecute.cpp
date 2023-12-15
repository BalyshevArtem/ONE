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

#include "execute/OMKernelExecute.h"
#include "core/OMKernelType.h"
#include "execute/OMKernelExecutionBuilder.h"

using namespace onert_micro::execute;
using namespace onert_micro;

OMStatus OMKernelExecute::executeKernel(core::OMRuntimeStorage &runtime_storage,
                                        core::OMRuntimeContext &runtime_context,
                                        core::OMBuilderID builder_id,
                                        uint16_t op_index)
{
  OMStatus status = Ok;

  KernelExecuteFunc *execute_func = nullptr;
  if (size_t(builder_id) < size_t(core::OMBuilderID::BuiltinOperatorsSize))
  {
    // Builtin operator
    status = kernel_builtin_execute.getKernelExecuteFunc(builder_id, &execute_func);
  } else
  {
    // Custom
    status = kernel_custom_execute.getKernelExecuteFunc(builder_id, &execute_func);
  }

  assert(execute_func != nullptr);

  if (status != Ok)
    return status;

  status = execute_func(runtime_storage, runtime_context, op_index);

  assert(status == Ok);

  return status;
}
