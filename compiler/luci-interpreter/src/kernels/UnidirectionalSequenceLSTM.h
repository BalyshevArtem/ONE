/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef LUCI_INTERPRETER_KERNELS_UNIDIRECTIONAL_SEQUENCE_LSTM_H
#define LUCI_INTERPRETER_KERNELS_UNIDIRECTIONAL_SEQUENCE_LSTM_H

#include "core/Kernel.h"
#include "core/KernelParams.h"
#include "luci_interpreter/core/Kernel.h"
#include <memory>

namespace luci_interpreter
{
namespace kernels
{

class UnidirectionalSequenceLSTM : public Kernel
{
public:
  UnidirectionalSequenceLSTM(std::vector<std::pair<const Tensor *, int32_t>> &&inputs, std::vector<std::pair<Tensor *, int32_t>> &&outputs);

  const Tensor *input() const { return _inputs[0].first; }
  int32_t input_ind() const { return _inputs[0].second; }

  const Tensor *input_to_input_weights() const { return _inputs[1].first; }
  int32_t input_to_input_weights_ind() const { return _inputs[1].second; }

  const Tensor *input_to_forget_weights() const { return _inputs[2].first; }
  int32_t input_to_forget_weights_ind() const { return _inputs[2].second; }

  const Tensor *input_to_cell_weights() const { return _inputs[3].first; }
  int32_t input_to_cell_weights_ind() const { return _inputs[3].second; }

  const Tensor *input_to_output_weights() const { return _inputs[4].first; }
  int32_t input_to_output_weights_ind() const { return _inputs[4].second; }

  const Tensor *recurrent_to_input_weights() const { return _inputs[5].first; }
  int32_t recurrent_to_input_weights_ind() const { return _inputs[5].second; }

  const Tensor *recurrent_to_forget_weights() const { return _inputs[6].first; }
  int32_t recurrent_to_forget_weights_ind() const { return _inputs[6].second; }

  const Tensor *recurrent_to_cell_weights() const { return _inputs[7].first; }
  int32_t recurrent_to_cell_weights_ind() const { return _inputs[7].second; }

  const Tensor *recurrent_to_output_weights() const { return _inputs[8].first; }
  int32_t recurrent_to_output_weights_ind() const { return _inputs[8].second; }

  const Tensor *cell_to_input_weights() const { return _inputs[9].first; }
  int32_t cell_to_input_weights_ind() const { return _inputs[9].second; }

  const Tensor *cell_to_forget_weights() const { return _inputs[10].first; }
  int32_t cell_to_forget_weights_ind() const { return _inputs[10].second; }

  const Tensor *cell_to_output_weights() const { return _inputs[11].first; }
  int32_t cell_to_output_weights_ind() const { return _inputs[11].second; }

  const Tensor *input_gate_bias() const { return _inputs[12].first; }
  int32_t input_gate_bias_ind() const { return _inputs[12].second; }

  const Tensor *forget_gate_bias() const { return _inputs[13].first; }
  int32_t forget_gate_bias_ind() const { return _inputs[13].second; }

  const Tensor *cell_gate_bias() const { return _inputs[14].first; }
  int32_t cell_gate_bias_ind() const { return _inputs[14].second; }

  const Tensor *output_gate_bias() const { return _inputs[15].first; }
  int32_t output_gate_bias_ind() const { return _inputs[15].second; }

  const Tensor *projection_weights() const { return _inputs[16].first; }
  int32_t projection_weights_ind() const { return _inputs[16].second; }

  const Tensor *projection_bias() const { return _inputs[17].first; }
  int32_t projection_bias_ind() const { return _inputs[17].second; }

  const Tensor *output_state() const { return _inputs[18].first; }
  int32_t output_state_ind() const { return _inputs[18].second; }

  const Tensor *cell_state() const { return _inputs[19].first; }
  int32_t cell_state_ind() const { return _inputs[19].second; }

  const Tensor *input_layer_norm_coefficients() const { return _inputs[20].first; }
  int32_t input_layer_norm_coefficients_ind() const { return _inputs[20].second; }

  const Tensor *forget_layer_norm_coefficients() const { return _inputs[21].first; }
  int32_t forget_layer_norm_coefficients_ind() const { return _inputs[21].second; }

  const Tensor *cell_layer_norm_coefficients() const { return _inputs[22].first; }
  int32_t cell_layer_norm_coefficients_ind() const { return _inputs[22].second; }

  const Tensor *output_layer_norm_coefficients() const { return _inputs[23].first; }
  int32_t output_layer_norm_coefficients_ind() const { return _inputs[23].second; }

  Tensor *output() const { return _outputs[0].first; }
  int32_t output_ind() const { return _outputs[0].second; }

  Tensor *scratch_buffer() const { return _outputs[1].first; }
  int32_t scratch_buffer_ind() const { return _outputs[1].second; }

  // Is hybrid scratchpad tensors
  Tensor *input_quantized() const { return _outputs.size() == 13 ? _outputs[2].first : nullptr; }
  int32_t input_quantized_ind() const { return _outputs.size() == 13 ? _outputs[2].second : -1; }

  Tensor *output_state_quantized() const { return _outputs.size() == 13 ? _outputs[3].first : nullptr; }
  int32_t output_state_quantized_ind() const { return _outputs.size() == 13 ? _outputs[3].second : -1; }

  Tensor *cell_state_quantized() const { return _outputs.size() == 13 ? _outputs[4].first : nullptr; }
  int32_t cell_state_quantized_ind() const { return _outputs.size() == 13 ? _outputs[4].second : -1; }

  Tensor *input_sf() const { return _outputs.size() == 13 ? _outputs[5].first : nullptr; }
  int32_t input_sf_ind() const { return _outputs.size() == 13 ? _outputs[5].second : -1; }

  Tensor *output_state_sf() const { return _outputs.size() == 13 ? _outputs[6].first : nullptr; }
  int32_t output_state_sf_ind() const { return _outputs.size() == 13 ? _outputs[6].second : -1; }

  Tensor *prod_scaling_factors() const { return _outputs.size() == 13 ? _outputs[7].first : nullptr; }
  int32_t prod_scaling_factors_ind() const { return _outputs.size() == 13 ? _outputs[7].second : -1; }

  Tensor *recovered_cell_weights() const { return _outputs.size() == 13 ? _outputs[8].first : nullptr; }
  int32_t recovered_cell_weights_ind() const { return _outputs.size() == 13 ? _outputs[8].second : -1; }

  Tensor *accum_scratch() const { return _outputs.size() == 13 ? _outputs[9].first : nullptr; }
  int32_t accum_scratch_ind() const { return _outputs.size() == 13 ? _outputs[9].second : -1; }

  Tensor *input_zp() const { return _outputs.size() == 13 ? _outputs[10].first : nullptr; }
  int32_t input_zp_ind() const { return _outputs.size() == 13 ? _outputs[10].second : -1; }

  Tensor *output_state_zp() const { return _outputs.size() == 13 ? _outputs[11].first : nullptr; }
  int32_t output_state_zp_ind() const { return _outputs.size() == 13 ? _outputs[11].second : -1; }

  Tensor *row_sums() const { return _outputs.size() == 13 ? _outputs[12].first : nullptr; }
  int32_t row_sums_ind() const { return _outputs.size() == 13 ? _outputs[12].second : -1; }

  void configure(luci::CircleReader *circle_reader, int32_t index) override;
  void execute(luci::CircleReader *circle_reader, int32_t index) const override;

private:
  void checkInputTensorDimensions(int n_input, int n_output, int n_cell, bool is_integer) const;

  void evalFloat(luci::CircleReader *circle_reader, int32_t index) const;
  void evalHybrid(luci::CircleReader *circle_reader, int32_t index) const;

  bool _compute_row_sums = false;
};

} // namespace kernels
} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_KERNELS_UNIDIRECTIONAL_SEQUENCE_LSTM_H
