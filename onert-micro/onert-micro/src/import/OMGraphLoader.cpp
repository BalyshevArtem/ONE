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

#include "import/OMGraphLoader.h"
#include "core/OMKernel.h"
#include "OMStatus.h"

using namespace onert_micro::core;
using namespace onert_micro::import;
using namespace onert_micro;


OMStatus OMGraphLoader::loadGraph(core::OMRuntimeStorage &runtime_storage, core::OMRuntimeContext &runtime_context,
                                  const OMConfig &configs)
{
  reader::OMCircleReader &reader = runtime_context.getCircleReader();
  const reader::CircleOperators *operators = reader.operators();
  for (uint16_t i = 0; i < static_cast<uint16_t>(operators->size()); ++i)
  {
    const circle::Operator *op = operators->operator[](i);
    assert(op != nullptr);
    if (op == nullptr)
      return UnknownError;

    OMKernel kernel;
    kernel.setKernelOperators({i});
    reader.builtin_code()


  }
  return Ok;
}
