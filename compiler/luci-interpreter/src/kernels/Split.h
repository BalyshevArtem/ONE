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

#ifndef LUCI_INTERPRETER_KERNELS_SPLIT_H
#define LUCI_INTERPRETER_KERNELS_SPLIT_H

#include "core/Kernel.h"
#include "core/KernelParams.h"
#include "luci_interpreter/core/Kernel.h"
namespace luci_interpreter
{
namespace kernels
{

class Split : public Kernel
{
public:
  Split(std::vector<std::pair<const Tensor *, int32_t>> &&inputs, std::vector<std::pair<Tensor *, int32_t>> &&outputs);

  const Tensor *axis() const { return _inputs[0].first; }
  int32_t axis_ind() const { return _inputs[0].second; }

  const Tensor *input() const { return _inputs[1].first; }
  int32_t input_ind() const { return _inputs[1].second; }

  Tensor *output(int index) const { return _outputs[index].first; }
  int32_t output_ind(int index) const { return _outputs[index].second; }

  void configure(luci::CircleReader *circle_reader, int32_t index) override;
  void execute(luci::CircleReader *circle_reader, int32_t index) const override;

private:
  template <typename T> void executeImpl(luci::CircleReader *circle_reader, int32_t index) const;
 // int32_t _axis_value{};
};

} // namespace kernels
} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_KERNELS_SPLIT_H
