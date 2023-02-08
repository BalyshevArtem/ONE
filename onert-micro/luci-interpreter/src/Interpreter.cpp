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

#include "luci_interpreter/Interpreter.h"
#include "memory_managers/SimpleMemoryManager.h"
//#include "memory_managers/StaticMemoryManager.h"

#include "loader/ModuleLoader.h"

namespace luci_interpreter
{

// Construct default interpreter with dynamic allocations and with input allocations
Interpreter::Interpreter(const char *model_data_raw)
{
  _runtime_module = std::make_unique<RuntimeModule>();

  _memory_manager = std::make_unique<SimpleMemoryManager>();
  //_memory_manager->is_allocate_input(true);

  //ModuleLoader loader();
  ModuleLoader::load(_runtime_module.get(), _memory_manager.get(),
                     /* is_static_allocations */ false, model_data_raw);
}

// Construct interpreter with configurations
Interpreter::Interpreter(const char *model_data_raw, const InterpreterConfigure &configuration)
{
  _runtime_module = std::make_unique<RuntimeModule>();

  // Note:
  // configuration._input_buf_size, configuration._temp_buf_size, configuration._output_buf_size
  // will be removed and will be read from circle file
//  if (configuration.isStaticManager())
//  {
//    _memory_manager = std::make_unique<StaticMemoryManager>(
//      configuration._input_buf_size, configuration._temp_buf_size, configuration._output_buf_size);
//  }
//  else
//  {
//    _memory_manager = std::make_unique<SimpleMemoryManager>();
//  }

  //_memory_manager->is_allocate_input(configuration.getAllocateInputValue());

  //ModuleLoader loader();
  ModuleLoader::load(_runtime_module.get(), _memory_manager.get(),
                     /* is_static_allocations */ configuration.isStaticManager(), model_data_raw);

//  ModuleLoader loader(_runtime_module.get(), _memory_manager.get());
//  loader.load(configuration.isStaticManager(), model_data_raw);
}

Interpreter::~Interpreter() = default;

void Interpreter::interpret() { _runtime_module->execute(); }

std::vector<const circle::Tensor *> Interpreter::getInputTensors() { return _runtime_module->getInputTensors(); }

std::vector<const circle::Tensor *> Interpreter::getOutputTensors()
{
  return _runtime_module->getOutputTensors();
}

void Interpreter::writeInputTensor(const circle::Tensor *input_tensor, const uint8_t *data, size_t data_size)
{
  auto *runtime_graph = _runtime_module->getMainGraph();
  runtime_graph->configureGraphInputs();

  auto *data_ptr = runtime_graph->getDataByCircleTensor(input_tensor);

  std::memcpy(data_ptr, data, data_size);

//  if (data != nullptr)
//    input_tensor->writeData(data, data_size);
}

void Interpreter::writeInputTensorWithoutCopy(const circle::Tensor *input_tensor, const uint8_t *data)
{
  auto *runtime_graph = _runtime_module->getMainGraph();
  runtime_graph->configureGraphInputs();
//  if (data != nullptr)
//    input_tensor->writeDataWithoutCopy(const_cast<void *>(data));
}

void Interpreter::readOutputTensor(const circle::Tensor *output_tensor, uint8_t *data, size_t data_size)
{
  auto *runtime_graph = _runtime_module->getMainGraph();

  auto *data_ptr = runtime_graph->getDataByCircleTensor(output_tensor);

  std::memcpy(data, data_ptr, data_size);

//  if (data != nullptr)
//    output_tensor->readData(data, data_size);
}

} // namespace luci_interpreter
