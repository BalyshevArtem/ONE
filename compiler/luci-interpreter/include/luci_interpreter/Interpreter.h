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

#ifndef LUCI_INTERPRETER_INTERPRETER_H
#define LUCI_INTERPRETER_INTERPRETER_H

#include "luci_interpreter/core/Tensor.h"

#include "luci_interpreter/core/Kernel.h"

#include <luci/IR/Nodes/CircleInput.h>
#include <luci/IR/Nodes/CircleOutput.h>

#include "luci_interpreter/MemoryManager.h"
#include <luci/IR/Module.h>

#include <memory>
#include <vector>
#include <unordered_map>

namespace luci_interpreter
{

//class ExecutionObserver
//{
//public:
//  virtual ~ExecutionObserver();
//
//  // Called when the value of a tensor has been updated during execution.
//  virtual void postTensorWrite(const luci::CircleNode *node, const Tensor *tensor);
//
//  // Called before / after executing an operator.
//  // Note that these methods are not called for auxiliary operators (CircleInput, CircleOutput,
//  // CircleConst and Circle*Out).
//  virtual void preOperatorExecute(const luci::CircleNode *node);
//  virtual void postOperatorExecute(const luci::CircleNode *node);
//};

class Interpreter
{
public:
  explicit Interpreter(const luci::Module *module);

  explicit Interpreter(const luci::Module *module, IMemoryManager *memory_manager);

  explicit Interpreter(std::vector<char> *model_data_raw);

  ~Interpreter();

  void writeInputTensor(const luci::CircleInput *input_node, const void *data, size_t data_size);

  void write_input_tensor(Tensor *tensor, const void *data, size_t data_size);

  Tensor *get_input_tensor_by_index(const luci::CircleInput *input_node);

  void readOutputTensor(const luci::CircleOutput *output_node, void *data, size_t data_size);

  void read_output_tensor(Tensor *tensor, void *data, size_t data_size);

  void interpret();

  const std::vector<Tensor *> &getInputTensors();
  const std::vector<Tensor *> &getOutputTensors();

  //void attachObserver(ExecutionObserver *observer);

  //const Tensor *getTensor(const loco::Node *node) { return _node_to_tensor[node]; }

private:
  // _default_memory_manager should be before _runtime_module due to
  // the order of deletion in the destructor
  std::unique_ptr<IMemoryManager> _default_memory_manager = nullptr;
  std::unique_ptr<class RuntimeModule> _runtime_module;

  std::unordered_map<Kernel *, Tensor *> _kernel_to_tensor;

  // Observer functionality support.
//  std::unique_ptr<struct RuntimeToIR> _runtime_to_ir;
//  std::unordered_map<const loco::Node *, Tensor *> _node_to_tensor;
//  std::unique_ptr<class EventNotifier> _event_notifier;
 // std::vector<ExecutionObserver *> _observers;
};

} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_INTERPRETER_H
