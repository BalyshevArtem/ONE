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

#ifndef LUCI_INTERPRETER_KERNELS_LOGISTIC_H
#define LUCI_INTERPRETER_KERNELS_LOGISTIC_H
#include "luci_interpreter/core/Kernel.h"
#include "core/Kernel.h"

namespace luci_interpreter
{
namespace kernels
{

namespace
{
struct OpDataLogistic
{
  int32_t input_zero_point;
  int32_t input_range_radius;
  int32_t input_multiplier;
  int input_left_shift;
};
} // namespace

class Logistic : public Kernel
{
public:
  Logistic(std::vector<std::pair<const Tensor *, int32_t>> &&inputs, std::vector<std::pair<Tensor *, int32_t>> &&outputs);

  const Tensor *input() const { return _inputs[0].first; }
  int32_t input_ind() const { return _inputs[0].second; }
  Tensor *output() const { return _outputs[0].first; }
  int32_t output_ind() const { return _outputs[0].second; }

  void configure(luci::CircleReader *circle_reader, int32_t index) override;
  void execute(luci::CircleReader *circle_reader, int32_t index) const override;

private:
  void evalFloat(luci::CircleReader *circle_reader) const;
  void evalQuantized() const;

  void calculateOpDataLogistic();

//private:
  //OpDataLogistic _op_data_logistic;

};

} // namespace kernels
} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_KERNELS_LOGISTIC_H
