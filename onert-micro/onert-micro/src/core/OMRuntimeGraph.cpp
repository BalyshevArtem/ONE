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

#include "core/OMRuntimeGraph.h"
#include "execute/OMKernelExecute.h"
#include "OMStatus.h"

using namespace onert_micro::core;
using namespace onert_micro;

OMStatus OMRuntimeGraph::run()
{
  OMStatus status = Ok;

  std::vector<OMKernel> &kernels = _storage.getKernels();

  const size_t num_kernels = kernels.size();

  for (uint32_t i = 0; i < num_kernels; ++i)
  {
    status = _allocator.allocate(i, &_context, &_storage);

    if (status != Ok)
      return status;

    OMKernel &cur_kernel = kernels.at(i);

    status = execute::OMKernelExecute::executeKernel(_storage,
                                                     _context,
                                                     cur_kernel);

    if (status != Ok)
      return status;

    status = _allocator.deallocate(i, &_storage);
  }

  return status;
}

void * OMRuntimeGraph::getInputDataAt(uint32_t position)
{
  const auto input_index = _context.getGraphInputTensorIndex(position);

  uint8_t *data;
  assert(_storage.getDataByTensorIndex(&data, input_index) == Ok);
  return reinterpret_cast<void *>(data);
}

void * OMRuntimeGraph::getOutputDataAt(uint32_t position)
{
  const auto output_index = _context.getGraphOutputTensorIndex(position);

  uint8_t *data;
  assert(_storage.getDataByTensorIndex(&data, output_index) == Ok);
  return reinterpret_cast<void *>(data);
}

uint32_t OMRuntimeGraph::getOutputSizeAt(uint32_t position)
{
  const auto output_index = _context.getGraphOutputTensorIndex(position);
  const circle::Tensor *output_tensor = _context.getTensorByIndex(output_index);

  return output_tensor->shape()->size();
}
