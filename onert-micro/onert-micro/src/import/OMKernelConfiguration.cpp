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

#include "import/OMKernelConfiguration.h"
#include "core/OMKernelType.h"
#include "import/OMKernelConfigureBuilder.h"

using namespace onert_micro::core;
using namespace onert_micro::import;
using namespace onert_micro;

OMStatus OMKernelConfiguration::configureKernels(core::OMRuntimeStorage &runtime_storage, core::OMRuntimeContext &runtime_context, const OMConfig &configs)
{
  OMStatus status = Ok;

  std::vector<OMKernel> &kernels = runtime_storage.getKernels();

  for (auto &cur_kernel : kernels)
  {
    const OMBuilderID builder_id = cur_kernel.getBuilderID();

    KernelConfigureFunc *configure_func = nullptr;
    if (size_t(builder_id) < size_t(core::OMBuilderID::BuiltinOperatorsSize))
    {
      // Builtin operator
      status = kernel_builtin_configure.getKernelConfigureFunc(builder_id, &configure_func);
    } else
    {
      // Custom
      status = kernel_custom_configure.getKernelConfigureFunc(builder_id, &configure_func);
    }

    assert(configure_func != nullptr);

    if (status != Ok)
      return status;

    status = configure_func(runtime_storage, runtime_context, cur_kernel, configs);
  }

  return status;
}
