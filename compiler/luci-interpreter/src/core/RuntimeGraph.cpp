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

#include "core/RuntimeGraph.h"

#include "core/RuntimeModule.h"

#include <algorithm>
#include <unordered_map>

namespace luci_interpreter
{

//class RuntimeGraph::TensorAllocPlan
//{
//  //std::vector<std::vector<Tensor *>> _alloc_plan;
//  //std::vector<std::vector<Tensor *>> _dealloc_plan;
//  bool _valid = false;
//  IMemoryManager *_memory_manager;
//
//public:
//  explicit TensorAllocPlan(IMemoryManager *memory_manager);
//  void invalidate() { _valid = false; }
//  bool isValid() const { return _valid; }
//  void build(const RuntimeGraph &graph);
//  void allocate(Kernel *kernel, size_t kernel_index) const;
//  void deallocate(Kernel *kernel, size_t kernel_index) const;
//};
//
//RuntimeGraph::TensorAllocPlan::TensorAllocPlan(IMemoryManager *memory_manager)
//  : _memory_manager(memory_manager)
//{
//}

void RuntimeGraph::build()
{
  invalidate();
  using Lifetime = std::pair<size_t, size_t>;
  std::unordered_map<Tensor *, Lifetime> lifetimes;
  const size_t num_kernels = _kernels.size();
  for (size_t index = 0; index < num_kernels; ++index)
  {
    const auto &kernel = _kernels[index];
    for (auto &pair : kernel->getInputTensors())
    {
      const Tensor *tensor = pair.first;
      auto nc_tensor = const_cast<Tensor *>(tensor);
      if (lifetimes.count(nc_tensor) > 0)
        lifetimes.at(nc_tensor).second = index;
    }
    for (auto &pair : kernel->getOutputTensors())
    {
      Tensor *tensor = pair.first;
      if (lifetimes.count(tensor) == 0)
        lifetimes[tensor] = Lifetime(index, index);
    }
  }
  for (const Tensor *tensor : getOutputTensors())
  {
    auto nc_tensor = const_cast<Tensor *>(tensor);
    if (lifetimes.count(nc_tensor) > 0)
      lifetimes.at(nc_tensor).second = num_kernels;
  }
  //_alloc_plan.assign(num_kernels, std::vector<Tensor *>());
  //_dealloc_plan.assign(num_kernels + 1, std::vector<Tensor *>());
  for (const auto &item : lifetimes)
  {
    item.first->alloc_kernel = item.second.first;
    item.first->dealloc_kernel = item.second.second;

    //_alloc_plan[item.second.first].push_back(item.first);
    //_dealloc_plan[item.second.second].push_back(item.first);
  }
  _valid = true;
}

void RuntimeGraph::allocate(Kernel *kernel, size_t kernel_index) const
{
  auto input_tensors = kernel->getInputTensors();

  for (auto &pair : input_tensors)
  {
    Tensor *tensor = const_cast<Tensor *>(pair.first);

    if (tensor == nullptr)
      continue;

    if (kernel_index == tensor->alloc_kernel)
      _memory_manager->allocate_memory(*tensor);
  }

  auto output_tensors = kernel->getOutputTensors();

  for (auto &pair : output_tensors)
  {
    Tensor *tensor = const_cast<Tensor *>(pair.first);

    if (kernel_index == tensor->alloc_kernel)
      _memory_manager->allocate_memory(*tensor);
  }
}

void RuntimeGraph::deallocate(Kernel *kernel, size_t kernel_index) const
{
  auto input_tensors = kernel->getInputTensors();

  for (auto &pair : input_tensors)
  {
    Tensor *tensor = const_cast<Tensor *>(pair.first);
    if (tensor == nullptr)
      continue;

    if (kernel_index == tensor->dealloc_kernel)
      _memory_manager->release_memory(*tensor);
  }

  auto output_tensors = kernel->getOutputTensors();

  for (auto &pair : output_tensors)
  {
    Tensor *tensor = const_cast<Tensor *>(pair.first);

    if (kernel_index == tensor->dealloc_kernel)
      _memory_manager->release_memory(*tensor);
  }
}

RuntimeGraph::RuntimeGraph(RuntimeModule *runtime_module, IMemoryManager *memory_manager)
  : _memory_manager(memory_manager), _owning_module(runtime_module)
{
}

RuntimeGraph::~RuntimeGraph()
{
  for (auto &tensor : _tensors)
  {
    if (tensor->is_data_allocated())
      _memory_manager->release_memory(*tensor);
  }
}

Tensor *RuntimeGraph::addTensor(std::unique_ptr<Tensor> &&tensor)
{
  assert(tensor != nullptr);
  _tensors.push_back(std::move(tensor));
  return _tensors.back().get();
}

void RuntimeGraph::setInputTensors(const std::vector<Tensor *> &input_tensors)
{
  assert(std::all_of(input_tensors.cbegin(), input_tensors.cend(),
                     [](Tensor *tensor) { return tensor != nullptr; }));
  _input_tensors = input_tensors;
}

void RuntimeGraph::add_input_tensor(Tensor *input_tensor)
{
  _input_tensors.push_back(input_tensor);
}

void RuntimeGraph::add_output_tensor(Tensor *output_tensor)
{
  _output_tensors.push_back(output_tensor);
}

void RuntimeGraph::setOutputTensors(const std::vector<Tensor *> &output_tensors)
{
  assert(std::all_of(output_tensors.cbegin(), output_tensors.cend(),
                     [](Tensor *tensor) { return tensor != nullptr; }));
  _output_tensors = output_tensors;
}

//void RuntimeGraph::configureAllocations(Tensor *tensor)
//{
//  _memory_manager->allocate_memory(*tensor);
//}

void RuntimeGraph::addKernel(std::unique_ptr<Kernel> &&kernel)
{
  assert(kernel != nullptr);
  _kernels.push_back(std::move(kernel));
  invalidate();
}

void RuntimeGraph::make_tensor_alloca_plan()
{
  if (!isValid())
    build();
}

//void RuntimeGraph::make_tensor_configure() const
//{
//  for (size_t index = 0; index < _kernels.size(); ++index)
//  {
//    const auto &kernel = _kernels[index];
//    kernel->configure();
//  }
//}

void RuntimeGraph::execute()
{
  /*
   * Test ficha
   */
  if (!isValid())
    build();

  //EventNotifier *event_notifier = _owning_module->getEventNotifier();

  // Notify the observers that the input tensors have changed.
//  if (event_notifier != nullptr)
//  {
//    for (const Tensor *input_tensor : getInputTensors())
//    {
//      if (input_tensor->is_observable())
//        event_notifier->postTensorWrite(input_tensor);
//    }
//  }

  for (size_t index = 0; index < _kernels.size(); ++index)
  {
    const auto &kernel = _kernels[index];
//    if (event_notifier != nullptr)
//    {
//      event_notifier->preOperatorExecute(kernel.get());
//    }

    // TODO The `configure` method should only be called if the outputs of an operator need to be
    //  resized.
    /*
     * Teset ficha
     */

    circle::OperatorT oper_t;
    _owning_module->get_circle_reader()->operators()[index]->UnPackTo(&oper_t);

    kernel->configure(_owning_module->get_circle_reader(), index);

    // Preallocate outputs in advance instead of relying on automatic allocation
    allocate(kernel.get(), index);

    kernel->execute(_owning_module->get_circle_reader(), index);

//    if (event_notifier != nullptr)
//    {
//      event_notifier->postOperatorExecute(kernel.get());
//    }

//    for (const Tensor *tensor : kernel->getOutputTensors())
//    {
//      if (event_notifier != nullptr && tensor->is_observable())
//      {
//        event_notifier->postTensorWrite(tensor);
//      }
//    }
    deallocate(kernel.get(), index);
  }
//
//  for (size_t index = 0; index < _kernels.size(); ++index)
//  {
//    auto &kernel = const_cast<std::unique_ptr<Kernel>&>(_kernels[index]);
//    kernel.reset();
//  }
//
//  for (size_t index = 0; index < _tensors.size(); ++index)
//  {
//    auto &tensor = const_cast<std::unique_ptr<Tensor>&>(_tensors[index]);
//    tensor.reset();
//  }
}

} // namespace luci_interpreter
