/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

#ifndef LUCI_INTERPRETER_PAL_UNIDIRECTIONAL_SEQUENCE_LSTM_H
#define LUCI_INTERPRETER_PAL_UNIDIRECTIONAL_SEQUENCE_LSTM_H

#include "PALUnidirectionalSequenceLSTMCommon.h"

#include <arm_nnfunctions.h>

namespace luci_interpreter_pal
{
namespace
{

void precomputeZeroPointTimesWeightWithBias(int32_t zero_point, const circle::Tensor *weight_tensor,
                                            const circle::Tensor *bias_tensor, std::unique_ptr<int32_t> &output,
                                            luci_interpreter::BaseRuntimeGraph *runtime_graph)
{
  if (weight_tensor == nullptr)
    return;

  const int row = luci_interpreter::Tensor::dim(weight_tensor, 0);
  const int col = luci_interpreter::Tensor::dim(weight_tensor, 1);

  {
    auto output_ptr = new int32_t[row * sizeof(int32_t)];
    output.reset(output_ptr);
  }
  if (bias_tensor == nullptr)
  {
    std::memset(output.get(), 0, row * sizeof(int32_t));
  } else
  {
    const int32_t *bias = luci_interpreter::kernels::getTensorData<int32_t>(runtime_graph->getConstDataByTensor(bias_tensor));
    std::memcpy(output.get(), bias, row * sizeof(int32_t));
  }

  if (zero_point != 0)
  {
    const int8_t *weight = luci_interpreter::kernels::getTensorData<int8_t>(runtime_graph->getConstDataByTensor(weight_tensor));
    luci_interpreter::kernels::matrixScalarMultiplyAccumulate(weight, zero_point, row, col, output.get());
  }
}

} // namespace



// Evaluate the LSTM kernel with (potential) multi-steps and multi-batch input
template <>
void evalLSTM<int8_t, int8_t, int16_t, int32_t>(
  luci_interpreter::lstm::LSTMStruct *lstm_struct,
  luci_interpreter::lstm::LSTMParameters *lstm_params,
  luci_interpreter::lstm::CellStateInfo *cell_state_info, int8_t *output_state_data,
  int16_t *cell_state_data, int16_t *scratch0, int16_t *scratch1, int16_t *scratch2,
  int16_t *scratch3, luci_interpreter::BaseRuntimeGraph *runtime_graph)
{
  cmsis_nn_lstm_context scratch_buffers;
  scratch_buffers.input_gate = scratch0;
  scratch_buffers.forget_gate = scratch1;
  scratch_buffers.cell_gate = scratch2;
  scratch_buffers.output_gate = scratch3;

  // Pre-calculate bias + zero_point * weight.
  std::unique_ptr<int32_t> input_to_forget_effective_bias;
  std::unique_ptr<int32_t> recurrent_to_forget_effective_bias;
  std::unique_ptr<int32_t> input_to_cell_effective_bias;
  std::unique_ptr<int32_t> recurrent_to_cell_effective_bias;
  std::unique_ptr<int32_t> input_to_output_effective_bias;
  std::unique_ptr<int32_t> recurrent_to_output_effective_bias;
  std::unique_ptr<int32_t> input_to_input_effective_bias;
  std::unique_ptr<int32_t> recurrent_to_input_effective_bias;

  const int32_t input_zero_point = -luci_interpreter::Tensor::zero_point(lstm_struct->input());
  const int32_t output_state_zero_point = luci_interpreter::Tensor::zero_point(lstm_struct->output_state();

  precomputeZeroPointTimesWeightWithBias(input)

  cmsis_nn_lstm_params cmsis_lstm_params;
  cmsis_lstm_params.output_state_offset = );
  //cmsis_lstm_params.i2f_effective_bias = lstm_params.
}

} // namespace luci_interpreter_pal

#endif // LUCI_INTERPRETER_PAL_UNIDIRECTIONAL_SEQUENCE_LSTM_H
