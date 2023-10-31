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

#include "core/OMRuntimeModule.h"

using namespace onert_micro::core;
using namespace onert_micro;

uint32_t OMRuntimeModule::getNumberOfInputs()
{
  return _graphs.at(0).getNumberOfInputs();
}

uint32_t OMRuntimeModule::getNumberOfOutputs()
{
  return _graphs.at(0).getNumberOfOutputs();
}

uint32_t OMRuntimeModule::getInputSizeAt(uint32_t position)
{
  return _graphs.at(0).getInputSizeAt(position);
}

uint32_t OMRuntimeModule::getOutputSizeAt(uint32_t position)
{
  return _graphs.at(0).getOutputSizeAt(position);
}

void *OMRuntimeModule::getInputDataAt(uint32_t position)
{
  return _graphs.at(0).getInputDataAt(position);
}

void *OMRuntimeModule::getOutputDataAt(uint32_t position)
{
  return _graphs.at(0).getOutputDataAt(position);
}

OMStatus OMRuntimeModule::importModel(const char *model_ptr, const OMConfig &config)
{
  assert(model_ptr != nullptr && "Model ptr shouldn't be nullptr");
  if (model_ptr == nullptr)
    return UnknownError;

  // First - parse reader
  // Second - load default graph
  // Third - optimize it until can
  // Then_4 - AllocDeallocPlan creation
  // Finally_5 - KernelConfigure
  OMStatus status;
  // First
  status = _reader.parse(model_ptr);
  if (status != Ok)
    return status;



  return Ok;
}
