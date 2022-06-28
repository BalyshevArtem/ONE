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

#ifndef LUCI_INTERPRETER_KERNELS_TANH_H
#define LUCI_INTERPRETER_KERNELS_TANH_H

#include "core/Kernel.h"
#include "luci_interpreter/core/Kernel.h"
namespace luci_interpreter
{
namespace kernels
{

class Tanh : public Kernel
{
public:
  Tanh(std::vector<std::pair<const Tensor *, int32_t>> &&inputs, std::vector<std::pair<Tensor *, int32_t>> &&outputs);

  const Tensor *input() const { return _inputs[0].first; }
  int32_t input_ind() const { return _inputs[0].second; }

  Tensor *output() const { return _outputs[0].first; }
  int32_t output_ind() const { return _outputs[0].second; }

  void configure(luci::CircleReader *circle_reader, int32_t index) override;
  void execute(luci::CircleReader *circle_reader, int32_t index) const override;

private:
  void evalFloat(luci::CircleReader *circle_reader, int32_t index) const;
//  void evalQuantized() const;
//  void populateLookupTable();
//  void setTableValue(uint8_t value, uint8_t idx) { _table[idx] = value; };
//  uint8_t getTableValue(uint8_t idx) const { return _table[idx]; };

private:
  //uint8_t _table[256]{};
};

} // namespace kernels
} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_KERNELS_TANH_H
