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

#include "Builders.h"
#include "kernels/Utils.h"

#include "PALUnidirectionalSequenceLSTM.h"
#include "PALApplyActivationToVector.h"
#include <tensorflow/lite/kernels/internal/tensor_utils_common.h>
#include <tensorflow/lite/kernels/internal/reference/portable_tensor_utils.h>

namespace luci_interpreter
{

namespace
{

void precompute_zero_point_times_weight_with_bias(
  int32_t zero_point, const circle::Tensor *weight_tensor, const uint8_t*weight_tensor_data, const uint8_t*bias_tensor_data,
  std::vector<int32_t> &output)
{
  if (weight_tensor == nullptr)
    return;

  LUCI_INTERPRETER_CHECK(Tensor::num_dims(weight_tensor) == 2);
  const int row = Tensor::dim(weight_tensor, 0);
  const int col = Tensor::dim(weight_tensor, 1);

  output.resize(row);

  if (bias_tensor_data == nullptr)
  {
    memset(output.data(), 0, row * sizeof(int32_t));
  }
  else
  {
    const auto *bias = kernels::getTensorData<int32_t>(bias_tensor_data);
    std::memcpy(output.data(), bias, row * sizeof(int32_t));
  }
  if (zero_point != 0)
  {
    const auto *weight = kernels::getTensorData<int8_t>(weight_tensor_data);
    luci_interpreter::kernels::matrixScalarMultiplyAccumulate(weight, zero_point, row, col, output.data());
  }
}

void populate_precomputed_zp_times_weight_with_bias(const circle::Tensor *input,

                                                    const circle::Tensor *input_to_input_weights, const circle::Tensor *input_to_forget_weights,
                                                    const circle::Tensor *input_to_cell_weights, const circle::Tensor *input_to_output_weights,

                                                    const circle::Tensor *recurrent_to_input_weights, const circle::Tensor *recurrent_to_forget_weights,
                                                    const circle::Tensor *recurrent_to_cell_weights, const circle::Tensor *recurrent_to_output_weights,

                                                    const circle::Tensor *cell_to_input_weights, const circle::Tensor *cell_to_forget_weights,
                                                    const circle::Tensor *cell_to_output_weights,

                                                    const circle::Tensor *input_gate_bias, const circle::Tensor *forget_gate_bias, const circle::Tensor *cell_gate_bias,
                                                    const circle::Tensor *output_gate_bias,

                                                    const circle::Tensor *projection_weights, const circle::Tensor *projection_bias,

                                                    const circle::Tensor *output_state, const circle::Tensor *cell_state,

                                                    const circle::Tensor *input_layer_norm_coefficients, const circle::Tensor *forget_layer_norm_coefficients,
                                                    const circle::Tensor *cell_layer_norm_coefficients, const circle::Tensor *output_layer_norm_coefficients,

                                                    const circle::Tensor *output,

                                                    const circle::Tensor *intermediate_tensor_0,

                                                    const circle::UnidirectionalSequenceLSTMOptions *options, luci_interpreter_pal::lstm::IntegerLSTMParams *integer_lstm_params,
                                                    BaseRuntimeGraph *runtime_graph)
{
  const int32_t input_zero_point = -Tensor::zero_point(input);
  const int32_t output_state_zero_point = -Tensor::zero_point(output_state);

  const int32_t hidden_zp = Tensor::zero_point(intermediate_tensor_0);

  // Get bias and perform zero point calculation.
  // When there is layer normalization, the gate bias does not apply to matmul
  // directly:
  //      y = ln(w * x + w * r + w * c) + b.

  bool is_layer_norm = (forget_layer_norm_coefficients != nullptr);

  {
    const auto forget_gate_bias_tmp_data =
      is_layer_norm ? nullptr : runtime_graph->getDataByTensor(forget_gate_bias);

    const auto input_to_forget_weights_data =
      runtime_graph->getDataByTensor(input_to_forget_weights);

    precompute_zero_point_times_weight_with_bias(
      input_zero_point, input_to_forget_weights, input_to_forget_weights_data,
      forget_gate_bias_tmp_data, integer_lstm_params->input_to_forget_effective_bias);
  }

  {
    const auto recurrent_to_forget_weights_data =
      runtime_graph->getDataByTensor(recurrent_to_forget_weights);

    precompute_zero_point_times_weight_with_bias(
      output_state_zero_point, recurrent_to_forget_weights, recurrent_to_forget_weights_data,
      nullptr, integer_lstm_params->recurrent_to_forget_effective_bias);
  }

  {
    // Modulation gate.
    const auto cell_gate_bias_tmp =
      is_layer_norm ? nullptr : runtime_graph->getDataByTensor(cell_gate_bias);

    const auto input_to_cell_weights_data = runtime_graph->getDataByTensor(input_to_cell_weights);

    precompute_zero_point_times_weight_with_bias(input_zero_point, input_to_cell_weights,
                                                 input_to_cell_weights_data, cell_gate_bias_tmp,
                                                 integer_lstm_params->input_to_cell_effective_bias);
  }

  {
    const auto recurrent_to_cell_weights_data = runtime_graph->getDataByTensor(recurrent_to_cell_weights);

    precompute_zero_point_times_weight_with_bias(
      output_state_zero_point, recurrent_to_cell_weights, recurrent_to_cell_weights_data, nullptr,
      integer_lstm_params->recurrent_to_cell_effective_bias);
  }

  {
    // Output gate.
    const auto output_gate_bias_tmp = is_layer_norm ? nullptr : runtime_graph->getDataByTensor(output_gate_bias);

    const auto input_to_output_weights_data = runtime_graph->getDataByTensor(input_to_output_weights);

    precompute_zero_point_times_weight_with_bias(
      input_zero_point, input_to_output_weights, input_to_output_weights_data, output_gate_bias_tmp,
      integer_lstm_params->input_to_output_effective_bias);
  }

  {
    const auto recurrent_to_output_weights_data = runtime_graph->getDataByTensor(recurrent_to_output_weights);

    precompute_zero_point_times_weight_with_bias(
      output_state_zero_point, recurrent_to_output_weights, recurrent_to_output_weights_data, nullptr,
      integer_lstm_params->recurrent_to_output_effective_bias);
  }

  {
    // Input gate. The calculation is only meaningful for non-cifg case.
    const auto input_gate_bias_tmp = is_layer_norm ? nullptr : runtime_graph->getDataByTensor(input_gate_bias);

    const auto input_to_input_weights_data = runtime_graph->getDataByTensor(input_to_input_weights);

    precompute_zero_point_times_weight_with_bias(
      input_zero_point, input_to_input_weights, input_to_input_weights_data, input_gate_bias_tmp,
      integer_lstm_params->input_to_input_effective_bias);
  }

  {
    const auto recurrent_to_input_weights_data = runtime_graph->getDataByTensor(recurrent_to_input_weights);

    precompute_zero_point_times_weight_with_bias(
      output_state_zero_point, recurrent_to_input_weights, recurrent_to_input_weights_data, nullptr,
      integer_lstm_params->recurrent_to_input_effective_bias);
  }

  {
    const auto projection_weights_data = runtime_graph->getDataByTensor(projection_weights);

    // Projection bias. The calculation is only meaningful for with projection.
    precompute_zero_point_times_weight_with_bias(hidden_zp, projection_weights, projection_weights_data, runtime_graph->getDataByTensor(projection_bias),
                                                 integer_lstm_params->projection_effective_bias);
  }
}

using namespace tflite;

// TODO: move this functions to another helper file (PAL, templates ?)
//void UpdateLstmCellFloat(int n_batch, int n_cell, float *cell_state, const float *input_gate,
//                         float *forget_gate, const float *cell_gate, bool use_cifg, float clip)
//{
//// NOTE tflite source is as is but will fail build with gcc-8 and above
//// TODO remove #pragma
//#pragma GCC diagnostic ignored "-Wrestrict"
//  tensor_utils::VectorVectorCwiseProduct(forget_gate, cell_state, n_batch * n_cell, cell_state);
//
//  if (use_cifg)
//  {
//    // With CIFG, input_gate = 1-forget_gate. Use the forget_gate array as
//    // scratch, as input_gate array is not allocated in this case. (Be careful
//    // not to write to the scratch before reading the forget gate data.)
//    float *scratch = forget_gate;
//    tensor_utils::Sub1Vector(forget_gate, n_batch * n_cell, scratch);
//    tensor_utils::VectorVectorCwiseProductAccumulate(cell_gate, scratch, n_batch * n_cell,
//                                                     cell_state);
//  }
//  else
//  {
//    tensor_utils::VectorVectorCwiseProductAccumulate(cell_gate, input_gate, n_batch * n_cell,
//                                                     cell_state);
//  }
//  if (clip > 0.0f)
//  {
//    tensor_utils::CwiseClipping(cell_state, n_batch * n_cell, clip);
//  }
//}
//
//void CalculateLstmOutputFloat(int n_batch, int n_cell, int n_output, const float *cell_state,
//                              const float *output_gate, TfLiteFusedActivation activation,
//                              const float *projection_weights, const float *projection_bias,
//                              const float proj_clip, float *output_state, float *scratch)
//{
//  luci_interpreter_pal::ApplyActivationToVector(cell_state, n_batch * n_cell, activation, scratch);
//  tensor_utils::VectorVectorCwiseProduct(output_gate, scratch, n_batch * n_cell, scratch);
//
//  const bool use_projection = (projection_weights != nullptr);
//  const bool use_projection_bias = (projection_bias != nullptr);
//
//  if (use_projection)
//  {
//    if (use_projection_bias)
//    {
//      tensor_utils::VectorBatchVectorAssign(projection_bias, n_output, n_batch, output_state);
//    }
//    else
//    {
//      std::fill_n(output_state, n_batch * n_output, 0.0f);
//    }
//    tensor_utils::MatrixBatchVectorMultiplyAccumulate(projection_weights, n_output, n_cell, scratch,
//                                                      n_batch, output_state);
//    if (proj_clip > 0.0f)
//    {
//      tensor_utils::CwiseClipping(output_state, n_batch * n_output, proj_clip);
//    }
//  }
//  else
//  {
//    std::copy_n(scratch, n_batch * n_output, output_state);
//  }
//}

//inline void CalculateLstmGateFloat(const float *input, const float *input_to_gate_weights,
//                                   const float *aux_input, const float *aux_input_to_gate_weights,
//                                   const float *output_state,
//                                   const float *recurrent_to_gate_weights, const float *cell_state,
//                                   const float *cell_to_gate_weights,
//                                   const float *layer_norm_coefficients, const float *gate_bias,
//                                   const int n_batch, const int n_input, const int n_aux_input,
//                                   const int n_output, const int n_cell,
//                                   const TfLiteFusedActivation activation, float *gate,
//                                   const bool is_input_all_zeros, const bool is_aux_input_all_zeros)
//{
//  const bool use_peephole = (cell_to_gate_weights != nullptr);
//  const bool use_layer_norm = (layer_norm_coefficients != nullptr);
//
//  // Initialize scratch buffers with bias for regular lstm or initialize with
//  // zero for layer norm lstm.
//  if (use_layer_norm)
//  {
//    std::fill_n(gate, n_cell * n_batch, 0.0f);
//  }
//  else
//  {
//    tensor_utils::VectorBatchVectorAssign(gate_bias, n_cell, n_batch, gate);
//  }
//  // For each batch and cell: compute input_weight * input.
//  // Skip if input is all zeros.
//  if (!is_input_all_zeros)
//  {
//    tensor_utils::MatrixBatchVectorMultiplyAccumulate(input_to_gate_weights, n_cell, n_input, input,
//                                                      n_batch, gate);
//  }
//  // For each batch and cell: compute aux_input_weight * aux_input.
//  // Skip if auxiliary input is not available or all zeros.
//  if (!is_aux_input_all_zeros)
//  {
//    tensor_utils::MatrixBatchVectorMultiplyAccumulate(aux_input_to_gate_weights, n_cell,
//                                                      n_aux_input, aux_input, n_batch, gate);
//  }
//  // For each batch and cell: compute recurrent_weight * output_state.
//  tensor_utils::MatrixBatchVectorMultiplyAccumulate(recurrent_to_gate_weights, n_cell, n_output,
//                                                    output_state, n_batch, gate);
//  // For each batch and cell: compute cell_weight .* cell_state (peephole LSTM)
//  if (use_peephole)
//  {
//    tensor_utils::VectorBatchVectorCwiseProductAccumulate(cell_to_gate_weights, n_cell, cell_state,
//                                                          n_batch, gate);
//  }
//  // Do layer normalization (if layer norm LSTM)
//  if (use_layer_norm)
//  {
//    tensor_utils::MeanStddevNormalization(gate, gate, n_cell, n_batch);
//    tensor_utils::VectorBatchVectorCwiseProduct(layer_norm_coefficients, n_cell, gate, n_batch,
//                                                gate);
//    tensor_utils::VectorBatchVectorAdd(gate_bias, n_cell, n_batch, gate);
//  }
//  // Apply activation
//  luci_interpreter_pal::ApplyActivationToVector(gate, n_batch * n_cell, activation, gate);
//}

//inline void LstmStepFloat(
//  const float *input_ptr, const float *input_to_input_weights_ptr,
//  const float *input_to_forget_weights_ptr, const float *input_to_cell_weights_ptr,
//  const float *input_to_output_weights_ptr, const float *aux_input_ptr,
//  const float *aux_input_to_input_weights_ptr, const float *aux_input_to_forget_weights_ptr,
//  const float *aux_input_to_cell_weights_ptr, const float *aux_input_to_output_weights_ptr,
//  const float *recurrent_to_input_weights_ptr, const float *recurrent_to_forget_weights_ptr,
//  const float *recurrent_to_cell_weights_ptr, const float *recurrent_to_output_weights_ptr,
//  const float *cell_to_input_weights_ptr, const float *cell_to_forget_weights_ptr,
//  const float *cell_to_output_weights_ptr, const float *input_layer_norm_coefficients_ptr,
//  const float *forget_layer_norm_coefficients_ptr, const float *cell_layer_norm_coefficients_ptr,
//  const float *output_layer_norm_coefficients_ptr, const float *input_gate_bias_ptr,
//  const float *forget_gate_bias_ptr, const float *cell_gate_bias_ptr,
//  const float *output_gate_bias_ptr, const float *projection_weights_ptr,
//  const float *projection_bias_ptr, const TfLiteLSTMParams *params, int n_batch, int n_cell,
//  int n_input, int n_aux_input, int n_output, int output_batch_leading_dim, float *output_state_ptr,
//  float *cell_state_ptr, float *scratch0, float *scratch1, float *scratch2, float *scratch3,
//  float *output_ptr)
//{
//  // Since we have already checked that weights are all there or none, we can
//  // check the existence of only one to the get the condition.
//  const bool use_cifg = (input_to_input_weights_ptr == nullptr);
//
//  // Make named scratch buffers.
//  float *input_gate_scratch = scratch0;
//  float *forget_gate_scratch = scratch1;
//  float *cell_gate_scratch = scratch2;
//  float *output_gate_scratch = scratch3;
//
//  // Check if inputs are all zeros so we can skip some computations.
//  const bool is_input_all_zeros = tensor_utils::IsZeroVector(input_ptr, n_batch * n_input);
//  const bool is_aux_input_all_zeros =
//    (aux_input_ptr == nullptr || tensor_utils::IsZeroVector(aux_input_ptr, n_batch * n_aux_input));
//  if (!use_cifg)
//  {
//    // Calculate the input gate. (If not CIFG.)
//    CalculateLstmGateFloat(input_ptr, input_to_input_weights_ptr, aux_input_ptr,
//                           aux_input_to_input_weights_ptr, output_state_ptr,
//                           recurrent_to_input_weights_ptr, cell_state_ptr,
//                           cell_to_input_weights_ptr, input_layer_norm_coefficients_ptr,
//                           input_gate_bias_ptr, n_batch, n_input, n_aux_input, n_output, n_cell,
//                           /*activation=*/kTfLiteActSigmoid, input_gate_scratch, is_input_all_zeros,
//                           is_aux_input_all_zeros);
//  }
//  // Calculate the forget gate.
//  CalculateLstmGateFloat(input_ptr, input_to_forget_weights_ptr, aux_input_ptr,
//                         aux_input_to_forget_weights_ptr, output_state_ptr,
//                         recurrent_to_forget_weights_ptr, cell_state_ptr,
//                         cell_to_forget_weights_ptr, forget_layer_norm_coefficients_ptr,
//                         forget_gate_bias_ptr, n_batch, n_input, n_aux_input, n_output, n_cell,
//                         /*activation=*/kTfLiteActSigmoid, forget_gate_scratch, is_input_all_zeros,
//                         is_aux_input_all_zeros);
//  // Calculate the cell update gate.
//  CalculateLstmGateFloat(
//    input_ptr, input_to_cell_weights_ptr, aux_input_ptr, aux_input_to_cell_weights_ptr,
//    output_state_ptr, recurrent_to_cell_weights_ptr, /*cell_state=*/nullptr,
//    /*cell_to_gate_weights=*/nullptr, cell_layer_norm_coefficients_ptr, cell_gate_bias_ptr, n_batch,
//    n_input, n_aux_input, n_output, n_cell, params->activation, cell_gate_scratch,
//    is_input_all_zeros, is_aux_input_all_zeros);
//  // Update the cell state.
//  UpdateLstmCellFloat(n_batch, n_cell, cell_state_ptr, input_gate_scratch, forget_gate_scratch,
//                      cell_gate_scratch, use_cifg, params->cell_clip);
//  // Calculate output gate.
//  CalculateLstmGateFloat(input_ptr, input_to_output_weights_ptr, aux_input_ptr,
//                         aux_input_to_output_weights_ptr, output_state_ptr,
//                         recurrent_to_output_weights_ptr, cell_state_ptr,
//                         cell_to_output_weights_ptr, output_layer_norm_coefficients_ptr,
//                         output_gate_bias_ptr, n_batch, n_input, n_aux_input, n_output, n_cell,
//                         /*activation=*/kTfLiteActSigmoid, output_gate_scratch, is_input_all_zeros,
//                         is_aux_input_all_zeros);
//  // Update the output state.
//  CalculateLstmOutputFloat(n_batch, n_cell, n_output, cell_state_ptr, output_gate_scratch,
//                           params->activation, projection_weights_ptr, projection_bias_ptr,
//                           params->proj_clip, output_state_ptr, scratch2);
//  // Copy output state to the output. Note that the output's rows may not be
//  // contiguous (output_batch_leading_dim != n_output).
//  for (int b = 0; b < n_batch; b++)
//  {
//    std::copy_n(output_state_ptr + b * n_output, n_output,
//                output_ptr + b * output_batch_leading_dim);
//  }
//}

} // namespace

//void EvalFloat(const Tensor *input,
//
//               const Tensor *input_to_input_weights, const Tensor *input_to_forget_weights,
//               const Tensor *input_to_cell_weights, const Tensor *input_to_output_weights,
//
//               const Tensor *recurrent_to_input_weights, const Tensor *recurrent_to_forget_weights,
//               const Tensor *recurrent_to_cell_weights, const Tensor *recurrent_to_output_weights,
//
//               const Tensor *cell_to_input_weights, const Tensor *cell_to_forget_weights,
//               const Tensor *cell_to_output_weights,
//
//               const Tensor *input_layer_norm_coefficients,
//               const Tensor *forget_layer_norm_coefficients,
//               const Tensor *cell_layer_norm_coefficients,
//               const Tensor *output_layer_norm_coefficients,
//
//               const Tensor *aux_input, const Tensor *aux_input_to_input_weights,
//               const Tensor *aux_input_to_forget_weights, const Tensor *aux_input_to_cell_weights,
//               const Tensor *aux_input_to_output_weights,
//
//               const Tensor *input_gate_bias, const Tensor *forget_gate_bias,
//               const Tensor *cell_gate_bias, const Tensor *output_gate_bias,
//
//               const Tensor *projection_weights, const Tensor *projection_bias,
//               const TfLiteLSTMParams *params,
//
//               bool forward_sequence, bool time_major, int output_offset,
//
//               Tensor *scratch_buffer, Tensor *output_state, Tensor *cell_state, Tensor *output)
//{
//  const Shape &input_shape = input->shape();
//  assert(input_shape.num_dims() >= 2 && input_shape.num_dims() <= 3);
//  int max_time, n_batch;
//  if (input_shape.num_dims() == 3)
//  {
//    max_time = (time_major) ? input_shape.dim(0) : input_shape.dim(1);
//    n_batch = (time_major) ? input_shape.dim(1) : input_shape.dim(0);
//  }
//  else
//  {
//    max_time = 1;
//    n_batch = input_shape.dim(0);
//  }
//  const int n_input = input_shape.dim(input_shape.num_dims() - 1);
//
//  int aux_input_temp = 0;
//  if (aux_input)
//  {
//    const Shape &aux_input_shape = aux_input->shape();
//    aux_input_temp = aux_input_shape.dim(aux_input_shape.num_dims() - 1);
//  }
//  const int aux_input_size = aux_input_temp;
//
//  // n_cell and n_output will be the same size when there is no projection.
//  const Shape &input_to_output_weights_shape = input_to_output_weights->shape();
//  const Shape &recurrent_to_output_weights_shape = recurrent_to_output_weights->shape();
//  const int n_cell = input_to_output_weights_shape.dim(0);
//  const int n_output = recurrent_to_output_weights_shape.dim(1);
//
//  // Since we have already checked that weights are all there or none, we can
//  // check the existence of only one to the get the condition.
//  const bool use_cifg = (input_to_input_weights == nullptr);
//
//  // Index the scratch buffers pointers to the global scratch buffer.
//  float *scratch_buffer_ptr = getTensorData<float>(scratch_buffer);
//  float *input_gate_scratch = nullptr;
//  float *cell_gate_scratch = nullptr;
//  float *forget_gate_scratch = nullptr;
//  float *output_gate_scratch = nullptr;
//  if (use_cifg)
//  {
//    cell_gate_scratch = scratch_buffer_ptr;
//    forget_gate_scratch = scratch_buffer_ptr + n_cell * n_batch;
//    output_gate_scratch = scratch_buffer_ptr + 2 * n_cell * n_batch;
//  }
//  else
//  {
//    input_gate_scratch = scratch_buffer_ptr;
//    cell_gate_scratch = scratch_buffer_ptr + n_cell * n_batch;
//    forget_gate_scratch = scratch_buffer_ptr + 2 * n_cell * n_batch;
//    output_gate_scratch = scratch_buffer_ptr + 3 * n_cell * n_batch;
//  }
//
//  const Shape &output_shape = output->shape();
//  const int output_batch_leading_dim = output_shape.dim(output_shape.num_dims() - 1);
//  if (time_major)
//  {
//    // Loop through the sequence.
//    const int input_step = n_batch * n_input;
//    const int output_step = n_batch * output_batch_leading_dim;
//    for (int t = 0; t < max_time; t++)
//    {
//      // If this is the forward_sequence, step forward, otherwise step
//      // backwards.
//      const int t_rel = forward_sequence ? t : max_time - t - 1;
//      const float *input_ptr = getTensorData<float>(input) + t_rel * input_step;
//      const float *aux_input_ptr = nullptr;
//      if (aux_input)
//      {
//        aux_input_ptr = getTensorData<float>(aux_input) + t_rel * input_step;
//      }
//      float *output_ptr = getTensorData<float>(output) + t_rel * output_step + output_offset;
//
//      LstmStepFloat(
//        input_ptr, getTensorData<float>(input_to_input_weights),
//        getTensorData<float>(input_to_forget_weights), getTensorData<float>(input_to_cell_weights),
//        getTensorData<float>(input_to_output_weights), aux_input_ptr,
//        getTensorData<float>(aux_input_to_input_weights),
//        getTensorData<float>(aux_input_to_forget_weights),
//        getTensorData<float>(aux_input_to_cell_weights),
//        getTensorData<float>(aux_input_to_output_weights),
//        getTensorData<float>(recurrent_to_input_weights),
//        getTensorData<float>(recurrent_to_forget_weights),
//        getTensorData<float>(recurrent_to_cell_weights),
//        getTensorData<float>(recurrent_to_output_weights),
//        getTensorData<float>(cell_to_input_weights), getTensorData<float>(cell_to_forget_weights),
//        getTensorData<float>(cell_to_output_weights),
//        getTensorData<float>(input_layer_norm_coefficients),
//        getTensorData<float>(forget_layer_norm_coefficients),
//        getTensorData<float>(cell_layer_norm_coefficients),
//        getTensorData<float>(output_layer_norm_coefficients), getTensorData<float>(input_gate_bias),
//        getTensorData<float>(forget_gate_bias), getTensorData<float>(cell_gate_bias),
//        getTensorData<float>(output_gate_bias), getTensorData<float>(projection_weights),
//        getTensorData<float>(projection_bias), params, n_batch, n_cell, n_input, aux_input_size,
//        n_output, output_batch_leading_dim, getTensorData<float>(output_state),
//        getTensorData<float>(cell_state), input_gate_scratch, forget_gate_scratch,
//        cell_gate_scratch, output_gate_scratch, output_ptr);
//    }
//  }
//  else
//  {
//    for (int b = 0; b < n_batch; b++)
//    {
//      const int input_step = n_input;
//      const int output_step = output_batch_leading_dim;
//      for (int t = 0; t < max_time; t++)
//      {
//        // If this is the forward_sequence, step forward, otherwise step
//        // backwards.
//        const int t_rel = forward_sequence ? t : max_time - t - 1;
//        const int time_offset = b * max_time + t_rel;
//        const float *input_ptr = getTensorData<float>(input) + time_offset * input_step;
//        const float *aux_input_ptr = nullptr;
//        if (aux_input)
//        {
//          aux_input_ptr = getTensorData<float>(aux_input) + time_offset * input_step;
//        }
//        float *output_ptr =
//          getTensorData<float>(output) + time_offset * output_step + output_offset;
//
//        // Offset the {output,cell}_state pointers to the right batch.
//        float *output_state_ptr = getTensorData<float>(output_state) + b * output_batch_leading_dim;
//        float *cell_state_ptr = getTensorData<float>(cell_state) + b * n_cell;
//        // Offset the scratch pointers to the right batch.
//        float *input_gate_scratch_ptr =
//          input_gate_scratch ? input_gate_scratch + b * n_cell : nullptr;
//        float *forget_gate_scratch_ptr = forget_gate_scratch + b * n_cell;
//        float *cell_gate_scratch_ptr = cell_gate_scratch + b * n_cell;
//        float *output_gate_scratch_ptr = output_gate_scratch + b * n_cell;
//
//        LstmStepFloat(
//          input_ptr, getTensorData<float>(input_to_input_weights),
//          getTensorData<float>(input_to_forget_weights),
//          getTensorData<float>(input_to_cell_weights),
//          getTensorData<float>(input_to_output_weights), aux_input_ptr,
//          getTensorData<float>(aux_input_to_input_weights),
//          getTensorData<float>(aux_input_to_forget_weights),
//          getTensorData<float>(aux_input_to_cell_weights),
//          getTensorData<float>(aux_input_to_output_weights),
//          getTensorData<float>(recurrent_to_input_weights),
//          getTensorData<float>(recurrent_to_forget_weights),
//          getTensorData<float>(recurrent_to_cell_weights),
//          getTensorData<float>(recurrent_to_output_weights),
//          getTensorData<float>(cell_to_input_weights), getTensorData<float>(cell_to_forget_weights),
//          getTensorData<float>(cell_to_output_weights),
//          getTensorData<float>(input_layer_norm_coefficients),
//          getTensorData<float>(forget_layer_norm_coefficients),
//          getTensorData<float>(cell_layer_norm_coefficients),
//          getTensorData<float>(output_layer_norm_coefficients),
//          getTensorData<float>(input_gate_bias), getTensorData<float>(forget_gate_bias),
//          getTensorData<float>(cell_gate_bias), getTensorData<float>(output_gate_bias),
//          getTensorData<float>(projection_weights), getTensorData<float>(projection_bias), params,
//          /*n_batch=*/1, n_cell, n_input, aux_input_size, n_output, output_batch_leading_dim,
//          output_state_ptr, cell_state_ptr, input_gate_scratch_ptr, forget_gate_scratch_ptr,
//          cell_gate_scratch_ptr, output_gate_scratch_ptr, output_ptr);
//      }
//    }
//  }
//}

void populate_quantized_lstm_params(const circle::Tensor *input,

                                    const circle::Tensor *input_to_input_weights, const circle::Tensor *input_to_forget_weights,
                                    const circle::Tensor *input_to_cell_weights, const circle::Tensor *input_to_output_weights,

                                    const circle::Tensor *recurrent_to_input_weights, const circle::Tensor *recurrent_to_forget_weights,
                                    const circle::Tensor *recurrent_to_cell_weights, const circle::Tensor *recurrent_to_output_weights,

                                    const circle::Tensor *cell_to_input_weights, const circle::Tensor *cell_to_forget_weights,
                                    const circle::Tensor *cell_to_output_weights,

                                    const circle::Tensor *input_gate_bias, const circle::Tensor *forget_gate_bias, const circle::Tensor *cell_gate_bias,
                                    const circle::Tensor *output_gate_bias,

                                    const circle::Tensor *projection_weights, const circle::Tensor *projection_bias,

                                    const circle::Tensor *output_state, const circle::Tensor *cell_state,

                                    const circle::Tensor *input_layer_norm_coefficients, const circle::Tensor *forget_layer_norm_coefficients,
                                    const circle::Tensor *cell_layer_norm_coefficients, const circle::Tensor *output_layer_norm_coefficients,

                                    const circle::Tensor *output,

                                    const circle::Tensor *intermediate_tensor_0,

                                    const circle::UnidirectionalSequenceLSTMOptions *options, luci_interpreter_pal::lstm::IntegerLSTMParams *integer_lstm_params)
{
  const float cell_clip = options->cell_clip();
  const float proj_clip = options->proj_clip();

  LUCI_INTERPRETER_CHECK(cell_state != nullptr);
  LUCI_INTERPRETER_CHECK(Tensor::scales(cell_state).size() > 0 and
                         Tensor::zero_points(cell_state).size() > 0);

  LUCI_INTERPRETER_CHECK(Tensor::scales(output).size() > 0 and Tensor::zero_points(output).size() > 0);

  if (cell_clip > 0.0f)
  {
    integer_lstm_params->quantized_cell_clip = static_cast<int16_t>(
      std::min(std::max(cell_clip / Tensor::scale(cell_state), -32768.0f), 32767.0f));
  }
  else
  {
    integer_lstm_params->quantized_cell_clip = 0;
  }

  if (proj_clip > 0.0f)
  {
    integer_lstm_params->quantized_proj_clip =
      static_cast<int8_t>(std::min(std::max(proj_clip / Tensor::scale(output), -128.0f), 127.0f));
  }
  else
  {
    integer_lstm_params->quantized_proj_clip = 0;
  }

  // Calculate effective scales.
  bool use_layer_norm = (forget_layer_norm_coefficients != nullptr);

  // Since we have already checked that weights are all there or none, we can
  // check the existence of only one to get the condition.
  const bool use_cifg = (input_to_input_weights == nullptr);
  const bool use_peephole = (cell_to_output_weights != nullptr);
  const bool use_projection = (projection_weights != nullptr);

  // Get intermediate scales and zero points.
  float intermediate_scale[5];
  int32_t intermediate_zp[5];

  for (int i = 0; i < 4; ++i)
  {
    if (use_layer_norm)
    {
      assert(false);
      // TODO: support it
    }
    else
    {
      intermediate_scale[i] = std::pow(2.0f, -12.0f);
      intermediate_zp[i] = 0;
    }
  }
  intermediate_scale[4] = Tensor::scale(intermediate_tensor_0);
  intermediate_zp[4] = Tensor::zero_point(intermediate_tensor_0);

  // Scales.
  const float default_scale = 1.0;
  float input_scale = default_scale;
  float input_to_input_weight_scale = default_scale;
  float recurrent_to_input_weight_scale = default_scale;
  float cell_to_input_weight_scale = default_scale;
  float input_to_forget_weight_scale = default_scale;
  float recurrent_to_forget_weight_scale = default_scale;
  float cell_to_forget_weight_scale = default_scale;
  float input_to_cell_weight_scale = default_scale;
  float recurrent_to_cell_weight_scale = default_scale;
  float input_to_output_weight_scale = default_scale;
  float recurrent_to_output_weight_scale = default_scale;
  float cell_to_output_weight_scale = default_scale;
  float projection_weight_scale = default_scale;
  float layer_norm_input_scale = default_scale;
  float layer_norm_forget_scale = default_scale;
  float layer_norm_cell_scale = default_scale;
  float layer_norm_output_scale = default_scale;
  float output_state_scale = default_scale;
  int cell_scale = 1;

  // Effective scales.
  float effective_input_to_input_scale = default_scale;
  float effective_recurrent_to_input_scale = default_scale;
  float effective_cell_to_input_scale = default_scale;
  float effective_input_to_forget_scale = default_scale;
  float effective_recurrent_to_forget_scale = default_scale;
  float effective_cell_to_forget_scale = default_scale;
  float effective_input_to_cell_scale = default_scale;
  float effective_recurrent_to_cell_scale = default_scale;
  float effective_input_to_output_scale = default_scale;
  float effective_recurrent_to_output_scale = default_scale;
  float effective_cell_to_output_scale = default_scale;
  float effective_proj_scale = default_scale;
  float effective_hidden_scale = default_scale;

  // Populate scales.
  if (!use_cifg)
  {
    input_to_input_weight_scale = Tensor::scale(input_to_input_weights);
    recurrent_to_input_weight_scale = Tensor::scale(recurrent_to_input_weights);
  }

  if (use_peephole)
  {
    if (!use_cifg)
    {
      cell_to_input_weight_scale = Tensor::scale(cell_to_input_weights);
    }
    cell_to_forget_weight_scale = Tensor::scale(cell_to_forget_weights);
    cell_to_output_weight_scale = Tensor::scale(cell_to_output_weights);
  }

  if (use_layer_norm)
  {
    if (!use_cifg)
    {
      layer_norm_input_scale = Tensor::scale(input_layer_norm_coefficients);
    }
    layer_norm_forget_scale = Tensor::scale(forget_layer_norm_coefficients);
    layer_norm_cell_scale = Tensor::scale(cell_layer_norm_coefficients);
    layer_norm_output_scale = Tensor::scale(output_layer_norm_coefficients);
  }

  if (use_projection)
  {
    projection_weight_scale = Tensor::scale(projection_weights);
  }
  output_state_scale = Tensor::scale(output_state);

  input_to_forget_weight_scale = Tensor::scale(input_to_forget_weights);
  input_to_cell_weight_scale = Tensor::scale(input_to_cell_weights);
  input_to_output_weight_scale = Tensor::scale(input_to_output_weights);
  recurrent_to_forget_weight_scale = Tensor::scale(recurrent_to_forget_weights);
  recurrent_to_cell_weight_scale = Tensor::scale(recurrent_to_cell_weights);
  recurrent_to_output_weight_scale = Tensor::scale(recurrent_to_output_weights);

  // Check cell state (already used above)
  {
    const float x_log2 = std::log(Tensor::scale(cell_state)) * (1.0f / std::log(2.0f));
    const float x_log2_rounded = std::round(x_log2);
    const float x_log2_fracpart = x_log2 - x_log2_rounded;
    cell_scale = static_cast<int>(x_log2_rounded);

    LUCI_INTERPRETER_CHECK(std::abs(x_log2_fracpart) < 1e-3f);
  }
  integer_lstm_params->cell_scale = cell_scale;
  input_scale = Tensor::scale(input);

  // Calculate effective scales.
  if (!use_cifg)
  {
    effective_input_to_input_scale =
      input_to_input_weight_scale * input_scale / intermediate_scale[0];
    effective_recurrent_to_input_scale =
      recurrent_to_input_weight_scale * output_state_scale / intermediate_scale[0];
  }
  effective_input_to_forget_scale =
    input_to_forget_weight_scale * input_scale / intermediate_scale[1];
  effective_recurrent_to_forget_scale =
    recurrent_to_forget_weight_scale * output_state_scale / intermediate_scale[1];

  effective_input_to_cell_scale = input_to_cell_weight_scale * input_scale / intermediate_scale[2];
  effective_recurrent_to_cell_scale =
    recurrent_to_cell_weight_scale * output_state_scale / intermediate_scale[2];

  effective_input_to_output_scale =
    input_to_output_weight_scale * input_scale / intermediate_scale[3];
  effective_recurrent_to_output_scale =
    recurrent_to_output_weight_scale * output_state_scale / intermediate_scale[3];

  effective_hidden_scale = std::pow(2.0f, -15.0f) / intermediate_scale[4] * std::pow(2.0f, -15.0f);

  effective_proj_scale = projection_weight_scale * intermediate_scale[4] / output_state_scale;

  if (use_peephole)
  {
    if (!use_cifg)
    {
      effective_cell_to_input_scale = std::pow(2.0f, static_cast<float>(cell_scale)) *
                                      cell_to_input_weight_scale / intermediate_scale[0];
    }
    effective_cell_to_forget_scale = std::pow(2.0f, static_cast<float>(cell_scale)) *
                                     cell_to_forget_weight_scale / intermediate_scale[1];
    effective_cell_to_output_scale = std::pow(2.0f, static_cast<float>(cell_scale)) *
                                     cell_to_output_weight_scale / intermediate_scale[3];
  }

  // Decompose scales.
  int shift_output = 0;

  kernels::quantizeMultiplier(static_cast<double>(effective_input_to_input_scale),
                     &integer_lstm_params->effective_input_to_input_scale_a, &shift_output);
  integer_lstm_params->effective_input_to_input_scale_b = static_cast<int32_t>(shift_output);

  kernels::quantizeMultiplier(static_cast<double>(effective_recurrent_to_input_scale),
                     &integer_lstm_params->effective_recurrent_to_input_scale_a, &shift_output);
  integer_lstm_params->effective_recurrent_to_input_scale_b = static_cast<int32_t>(shift_output);

  kernels::quantizeMultiplier(static_cast<double>(effective_cell_to_input_scale),
                     &integer_lstm_params->effective_cell_to_input_scale_a, &shift_output);
  integer_lstm_params->effective_cell_to_input_scale_b = static_cast<int32_t>(shift_output);

  kernels::quantizeMultiplier(static_cast<double>(effective_input_to_forget_scale),
                     &integer_lstm_params->effective_input_to_forget_scale_a, &shift_output);
  integer_lstm_params->effective_input_to_forget_scale_b = static_cast<int32_t>(shift_output);

  kernels::quantizeMultiplier(static_cast<double>(effective_recurrent_to_forget_scale),
                     &integer_lstm_params->effective_recurrent_to_forget_scale_a, &shift_output);
  integer_lstm_params->effective_recurrent_to_forget_scale_b = static_cast<int32_t>(shift_output);

  kernels::quantizeMultiplier(static_cast<double>(effective_cell_to_forget_scale),
                     &integer_lstm_params->effective_cell_to_forget_scale_a, &shift_output);
  integer_lstm_params->effective_cell_to_forget_scale_b = static_cast<int32_t>(shift_output);
  kernels::quantizeMultiplier(static_cast<double>(effective_input_to_cell_scale),
                     &integer_lstm_params->effective_input_to_cell_scale_a, &shift_output);
  integer_lstm_params->effective_input_to_cell_scale_b = static_cast<int32_t>(shift_output);

  kernels::quantizeMultiplier(static_cast<double>(effective_recurrent_to_cell_scale),
                     &integer_lstm_params->effective_recurrent_to_cell_scale_a, &shift_output);
  integer_lstm_params->effective_recurrent_to_cell_scale_b = static_cast<int32_t>(shift_output);

  kernels::quantizeMultiplier(static_cast<double>(effective_input_to_output_scale),
                     &integer_lstm_params->effective_input_to_output_scale_a, &shift_output);
  integer_lstm_params->effective_input_to_output_scale_b = static_cast<int32_t>(shift_output);

  kernels::quantizeMultiplier(static_cast<double>(effective_recurrent_to_output_scale),
                     &integer_lstm_params->effective_recurrent_to_output_scale_a, &shift_output);
  integer_lstm_params->effective_recurrent_to_output_scale_b = static_cast<int32_t>(shift_output);

  kernels::quantizeMultiplier(static_cast<double>(effective_cell_to_output_scale),
                     &integer_lstm_params->effective_cell_to_output_scale_a, &shift_output);
  integer_lstm_params->effective_cell_to_output_scale_b = static_cast<int32_t>(shift_output);

  kernels::quantizeMultiplier(static_cast<double>(effective_proj_scale),
                     &integer_lstm_params->effective_proj_scale_a, &shift_output);
  integer_lstm_params->effective_proj_scale_b = static_cast<int32_t>(shift_output);

  kernels::quantizeMultiplier(static_cast<double>(effective_hidden_scale),
                     &integer_lstm_params->effective_hidden_scale_a, &shift_output);
  integer_lstm_params->effective_hidden_scale_b = static_cast<int32_t>(shift_output);

  kernels::quantizeMultiplier(static_cast<double>(layer_norm_input_scale),
                     &integer_lstm_params->layer_norm_input_scale_a, &shift_output);
  integer_lstm_params->layer_norm_input_scale_b = static_cast<int32_t>(shift_output);

  kernels::quantizeMultiplier(static_cast<double>(layer_norm_forget_scale),
                     &integer_lstm_params->layer_norm_forget_scale_a, &shift_output);
  integer_lstm_params->layer_norm_forget_scale_b = static_cast<int32_t>(shift_output);

  kernels::quantizeMultiplier(static_cast<double>(layer_norm_cell_scale),
                     &integer_lstm_params->layer_norm_cell_scale_a, &shift_output);
  integer_lstm_params->layer_norm_cell_scale_b = static_cast<int32_t>(shift_output);

  kernels::quantizeMultiplier(static_cast<double>(layer_norm_output_scale),
                     &integer_lstm_params->layer_norm_output_scale_a, &shift_output);
  integer_lstm_params->layer_norm_output_scale_b = static_cast<int32_t>(shift_output);

  integer_lstm_params->hidden_zp = intermediate_zp[4];

  // 10000 is used to make sure the kernel logic does not overflow.
  if (!use_cifg)
  {
    integer_lstm_params->input_variance_guard =
      std::max(1, static_cast<int>(10000 * layer_norm_input_scale));
  }
  integer_lstm_params->forget_variance_guard =
    std::max(1, static_cast<int>(10000 * layer_norm_forget_scale));
  integer_lstm_params->cell_variance_guard =
    std::max(1, static_cast<int>(10000 * layer_norm_cell_scale));
  integer_lstm_params->output_variance_guard =
    std::max(1, static_cast<int>(10000 * layer_norm_output_scale));
}

void configure_kernel_CircleUnidirectionalSequenceLSTM(const circle::Operator *cur_op, BaseRuntimeGraph *runtime_graph)
{
  const auto input_index = cur_op->inputs()->operator[](0);
  const auto input_to_input_weights_index = cur_op->inputs()->operator[](1);
  const auto input_to_forget_weights_index = cur_op->inputs()->operator[](2);
  const auto input_to_cell_weights_index = cur_op->inputs()->operator[](3);
  const auto input_to_output_weights_index = cur_op->inputs()->operator[](4);
  assert(input_index != -1);
  assert(input_to_input_weights_index != -1);
  assert(input_to_forget_weights_index != -1);
  assert(input_to_cell_weights_index != -1);
  assert(input_to_output_weights_index != -1);
  const auto input = runtime_graph->getCircleTensorByIndex(input_index);
  const auto input_to_input_weights = runtime_graph->getCircleTensorByIndex(input_to_input_weights_index);
  const auto input_to_forget_weights = runtime_graph->getCircleTensorByIndex(input_to_forget_weights_index);
  const auto input_to_cell_weights = runtime_graph->getCircleTensorByIndex(input_to_cell_weights_index);
  const auto input_to_output_weights = runtime_graph->getCircleTensorByIndex(input_to_output_weights_index);

  const auto recurrent_to_input_weights_index = cur_op->inputs()->operator[](5);
  const auto recurrent_to_forget_weights_index = cur_op->inputs()->operator[](6);
  const auto recurrent_to_cell_weights_index = cur_op->inputs()->operator[](7);
  const auto recurrent_to_output_weights_index = cur_op->inputs()->operator[](8);
  assert(recurrent_to_input_weights_index != -1);
  assert(recurrent_to_forget_weights_index != -1);
  assert(recurrent_to_cell_weights_index != -1);
  assert(recurrent_to_output_weights_index != -1);
  const auto recurrent_to_input_weights = runtime_graph->getCircleTensorByIndex(recurrent_to_input_weights_index);
  const auto recurrent_to_forget_weights = runtime_graph->getCircleTensorByIndex(recurrent_to_forget_weights_index);
  const auto recurrent_to_cell_weights = runtime_graph->getCircleTensorByIndex(recurrent_to_cell_weights_index);
  const auto recurrent_to_output_weights = runtime_graph->getCircleTensorByIndex(recurrent_to_output_weights_index);

  const auto cell_to_input_weights_index = cur_op->inputs()->operator[](9);
  const auto cell_to_forget_weights_index = cur_op->inputs()->operator[](10);
  const auto cell_to_output_weights_index = cur_op->inputs()->operator[](11);
  assert(cell_to_input_weights_index != -1);
  assert(cell_to_forget_weights_index != -1);
  assert(cell_to_output_weights_index != -1);
  const auto cell_to_input_weights = runtime_graph->getCircleTensorByIndex(cell_to_input_weights_index);
  const auto cell_to_forget_weights = runtime_graph->getCircleTensorByIndex(cell_to_forget_weights_index);
  const auto cell_to_output_weights = runtime_graph->getCircleTensorByIndex(cell_to_output_weights_index);

  const auto input_gate_bias_index = cur_op->inputs()->operator[](12);
  const auto forget_gate_bias_index = cur_op->inputs()->operator[](13);
  const auto cell_gate_bias_index = cur_op->inputs()->operator[](14);
  const auto output_gate_bias_index = cur_op->inputs()->operator[](15);
  assert(input_gate_bias_index != -1);
  assert(forget_gate_bias_index != -1);
  assert(cell_gate_bias_index != -1);
  assert(output_gate_bias_index != -1);
  const auto input_gate_bias = runtime_graph->getCircleTensorByIndex(input_gate_bias_index);
  const auto forget_gate_bias = runtime_graph->getCircleTensorByIndex(forget_gate_bias_index);
  const auto cell_gate_bias = runtime_graph->getCircleTensorByIndex(cell_gate_bias_index);
  const auto output_gate_bias = runtime_graph->getCircleTensorByIndex(output_gate_bias_index);

  const auto projection_weights_index = cur_op->inputs()->operator[](16);
  const auto projection_bias_index = cur_op->inputs()->operator[](17);
  assert(projection_weights_index != -1);
  assert(projection_bias_index != -1);
  const auto projection_weights = runtime_graph->getCircleTensorByIndex(projection_weights_index);
  const auto projection_bias = runtime_graph->getCircleTensorByIndex(projection_bias_index);

  const auto output_state_index = cur_op->inputs()->operator[](18);
  const auto cell_state_index = cur_op->inputs()->operator[](19);
  assert(output_state_index != -1);
  assert(cell_state_index != -1);
  const auto output_state = runtime_graph->getCircleTensorByIndex(output_state_index);
  const auto cell_state = runtime_graph->getCircleTensorByIndex(cell_state_index);

  const auto input_layer_norm_coefficients_index = cur_op->inputs()->operator[](20);
  const auto forget_layer_norm_coefficients_index = cur_op->inputs()->operator[](21);
  const auto cell_layer_norm_coefficients_index = cur_op->inputs()->operator[](22);
  const auto output_layer_norm_coefficients_index = cur_op->inputs()->operator[](23);
  assert(input_layer_norm_coefficients_index != -1);
  assert(forget_layer_norm_coefficients_index != -1);
  assert(cell_layer_norm_coefficients_index != -1);
  assert(output_layer_norm_coefficients_index != -1);
  const auto input_layer_norm_coefficients = runtime_graph->getCircleTensorByIndex(input_layer_norm_coefficients_index);
  const auto forget_layer_norm_coefficients = runtime_graph->getCircleTensorByIndex(forget_layer_norm_coefficients_index);
  const auto cell_layer_norm_coefficients = runtime_graph->getCircleTensorByIndex(cell_layer_norm_coefficients_index);
  const auto output_layer_norm_coefficients = runtime_graph->getCircleTensorByIndex(output_layer_norm_coefficients_index);

  const auto output_index = cur_op->outputs()->operator[](0);
  assert(output_index != -1);
  const auto output = runtime_graph->getCircleTensorByIndex(output_index);

  LUCI_INTERPRETER_CHECK(Tensor::element_type(input) == DataType::FLOAT32 or
                         Tensor::element_type(input) == DataType::S8);
  const bool is_integer = Tensor::element_type(input) == DataType::S8;
  const bool use_layer_norm = (forget_layer_norm_coefficients != nullptr);

  // TODO: support it
  if (use_layer_norm and is_integer)
    assert(false && "Not supported now");

  const auto *options = cur_op->builtin_options_as_UnidirectionalSequenceLSTMOptions();

  // Inferring batch size, number of outputs and sequence length and
  // number of cells from the input tensors.
  LUCI_INTERPRETER_CHECK(Tensor::num_dims(input) > 1);
  const bool time_major = options->time_major();
  const int n_batch = time_major ? Tensor::dim(input, 1) : Tensor::dim(input, 0);
  // NOTE as dim(2) is accessed, we need to check this is valid
  LUCI_INTERPRETER_CHECK(Tensor::num_dims(input) > 2);
  const int n_input = Tensor::dim(input, 2);

  const int n_cell = Tensor::dim(input_to_output_weights, 0);
  LUCI_INTERPRETER_CHECK(Tensor::num_dims(input_to_output_weights) == 2);
  LUCI_INTERPRETER_CHECK(Tensor::dim(input_to_output_weights, 1) == n_input);

  LUCI_INTERPRETER_CHECK(Tensor::num_dims(recurrent_to_output_weights) == 2);
  LUCI_INTERPRETER_CHECK(Tensor::dim(recurrent_to_output_weights, 0) == n_cell);

  const int n_output = Tensor::dim(recurrent_to_output_weights, 1);

  // Check that input tensor dimensions matches with each other.
  //check_input_tensor_dimensions(n_input, n_output, n_cell, use_layer_norm, is_integer);
  // Making sure clipping parameters have valid values.
  // == 0 means no clipping
  //  > 0 means clipping
  LUCI_INTERPRETER_CHECK(options->cell_clip() >= 0);
  LUCI_INTERPRETER_CHECK(options->proj_clip() >= 0);

  if (input_to_input_weights != nullptr)
  {
    LUCI_INTERPRETER_CHECK(Tensor::num_dims(input_to_input_weights) == 2);
    LUCI_INTERPRETER_CHECK(Tensor::dim(input_to_input_weights,0) == n_cell);
    LUCI_INTERPRETER_CHECK(Tensor::dim(input_to_input_weights, 1) == n_input);
  }

  LUCI_INTERPRETER_CHECK(Tensor::num_dims(input_to_forget_weights) == 2);
  LUCI_INTERPRETER_CHECK(Tensor::dim(input_to_forget_weights, 0) == n_cell);
  LUCI_INTERPRETER_CHECK(Tensor::dim(input_to_forget_weights, 1) == n_input);

  LUCI_INTERPRETER_CHECK(Tensor::num_dims(input_to_cell_weights) == 2);
  LUCI_INTERPRETER_CHECK(Tensor::dim(input_to_cell_weights, 0) == n_cell);
  LUCI_INTERPRETER_CHECK(Tensor::dim(input_to_cell_weights, 1) == n_input);

  if (recurrent_to_input_weights != nullptr)
  {
    LUCI_INTERPRETER_CHECK(Tensor::num_dims(recurrent_to_input_weights) == 2);
    LUCI_INTERPRETER_CHECK(Tensor::dim(recurrent_to_input_weights, 0) == n_cell);
    LUCI_INTERPRETER_CHECK(Tensor::dim(recurrent_to_input_weights, 1) == n_output);
  }

  LUCI_INTERPRETER_CHECK(Tensor::num_dims(recurrent_to_forget_weights) == 2);
  LUCI_INTERPRETER_CHECK(Tensor::dim(recurrent_to_forget_weights, 0) == n_cell);
  LUCI_INTERPRETER_CHECK(Tensor::dim(recurrent_to_forget_weights, 1) == n_output);

  LUCI_INTERPRETER_CHECK(Tensor::num_dims(recurrent_to_cell_weights) == 2);
  LUCI_INTERPRETER_CHECK(Tensor::dim(recurrent_to_cell_weights, 0) == n_cell);
  LUCI_INTERPRETER_CHECK(Tensor::dim(recurrent_to_cell_weights, 1) == n_output);

  // We make sure the input-gate's parameters are either both present (regular
  // LSTM) or not at all (CIFG-LSTM).
  const bool cifg_weights_all_or_none =
    ((input_to_input_weights != nullptr) && (recurrent_to_input_weights != nullptr)) ||
    ((input_to_input_weights == nullptr) && (recurrent_to_input_weights == nullptr));
  LUCI_INTERPRETER_CHECK(cifg_weights_all_or_none == true);

  if (cell_to_input_weights != nullptr)
  {
    LUCI_INTERPRETER_CHECK(Tensor::num_dims(cell_to_input_weights) == 1);
    LUCI_INTERPRETER_CHECK(Tensor::dim(cell_to_input_weights, 0) == n_cell);
    LUCI_INTERPRETER_CHECK(is_integer ? Tensor::element_type(cell_to_input_weights) == DataType::S16
                                      : Tensor::element_type(cell_to_input_weights) ==
                                          Tensor::element_type(input_to_forget_weights));
  }

  if (cell_to_forget_weights != nullptr)
  {
    LUCI_INTERPRETER_CHECK(Tensor::num_dims(cell_to_forget_weights) == 1);
    LUCI_INTERPRETER_CHECK(Tensor::dim(cell_to_forget_weights, 0) == n_cell);
    LUCI_INTERPRETER_CHECK(is_integer ? Tensor::element_type(cell_to_forget_weights) == DataType::S16
                                      : Tensor::element_type(cell_to_forget_weights) ==
                             Tensor::element_type(input_to_forget_weights));
  }

  if (cell_to_output_weights != nullptr)
  {
    LUCI_INTERPRETER_CHECK(Tensor::num_dims(cell_to_output_weights) == 1);
    LUCI_INTERPRETER_CHECK(Tensor::dim(cell_to_output_weights, 0) == n_cell);
    LUCI_INTERPRETER_CHECK(is_integer ? Tensor::element_type(cell_to_output_weights) == DataType::S16
                                      : Tensor::element_type(cell_to_output_weights) ==
                             Tensor::element_type(input_to_forget_weights));
  }

  // Making sure the peephole weights are there all or none.
  const bool use_cifg = (input_to_input_weights == nullptr);
  const bool peephole_weights_all_or_none =
    ((cell_to_input_weights != nullptr || use_cifg) && (cell_to_forget_weights != nullptr) &&
     (cell_to_output_weights != nullptr)) ||
    ((cell_to_input_weights == nullptr) && (cell_to_forget_weights == nullptr) &&
     (cell_to_output_weights == nullptr));
  LUCI_INTERPRETER_CHECK(peephole_weights_all_or_none == true);

  // Make sure the input gate bias is present only when not a CIFG-LSTM.
  if (use_cifg)
  {
    LUCI_INTERPRETER_CHECK(input_gate_bias == nullptr);
  }
  else
  {
    LUCI_INTERPRETER_CHECK(Tensor::num_dims(input_gate_bias) == 1);
    LUCI_INTERPRETER_CHECK(Tensor::dim(input_gate_bias, 0) == n_cell);
    if (is_integer)
    {
      LUCI_INTERPRETER_CHECK(Tensor::element_type(input_gate_bias) == DataType::S32);
    }
    else
    {
      LUCI_INTERPRETER_CHECK(Tensor::element_type(input_gate_bias) == DataType::FLOAT32);
    }
  }

  LUCI_INTERPRETER_CHECK(Tensor::num_dims(forget_gate_bias) == 1);
  LUCI_INTERPRETER_CHECK(Tensor::dim(forget_gate_bias, 0) == n_cell);
  if (is_integer)
  {
    LUCI_INTERPRETER_CHECK(Tensor::element_type(forget_gate_bias) == DataType::S32);
  }
  else
  {
    LUCI_INTERPRETER_CHECK(Tensor::element_type(forget_gate_bias) == DataType::FLOAT32);
  }

  LUCI_INTERPRETER_CHECK(Tensor::num_dims(cell_gate_bias) == 1);
  LUCI_INTERPRETER_CHECK(Tensor::dim(cell_gate_bias, 0) == n_cell);
  if (is_integer)
  {
    LUCI_INTERPRETER_CHECK(Tensor::element_type(cell_gate_bias) == DataType::S32);
  }
  else
  {
    LUCI_INTERPRETER_CHECK(Tensor::element_type(cell_gate_bias) == DataType::FLOAT32);
  }

  LUCI_INTERPRETER_CHECK(Tensor::num_dims(output_gate_bias) == 1);
  LUCI_INTERPRETER_CHECK(Tensor::dim(output_gate_bias, 0) == n_cell);
  if (is_integer)
  {
    LUCI_INTERPRETER_CHECK(Tensor::element_type(output_gate_bias) == DataType::S32);
  }
  else
  {
    LUCI_INTERPRETER_CHECK(Tensor::element_type(output_gate_bias) == DataType::FLOAT32);
  }

  if (projection_weights != nullptr)
  {
    LUCI_INTERPRETER_CHECK(Tensor::num_dims(projection_weights) == 2);
    LUCI_INTERPRETER_CHECK(Tensor::dim(projection_weights, 0) == n_output);
    LUCI_INTERPRETER_CHECK(Tensor::dim(projection_weights, 1) == n_cell);
  }

  if (projection_bias != nullptr)
  {
    LUCI_INTERPRETER_CHECK(Tensor::num_dims(projection_bias) == 1);
    LUCI_INTERPRETER_CHECK(Tensor::dim(projection_bias, 0) == n_output);
    if (is_integer)
    {
      LUCI_INTERPRETER_CHECK(Tensor::element_type(projection_bias) == DataType::S32);
    }
    else
    {
      LUCI_INTERPRETER_CHECK(Tensor::element_type(projection_bias) == DataType::FLOAT32);
    }
  }

  // Making sure the projection tensors are consistent:
  // 1) If projection weight is not present, then projection bias should not be
  // present.
  // 2) If projection weight is present, then projection bias is optional.
  const bool projecton_tensors_consistent =
    ((projection_weights != nullptr) || (projection_bias == nullptr));
  LUCI_INTERPRETER_CHECK(projecton_tensors_consistent == true);

  if (use_layer_norm)
  {
    if (use_cifg)
    {
      LUCI_INTERPRETER_CHECK(input_layer_norm_coefficients == nullptr);
    }
    else
    {
      LUCI_INTERPRETER_CHECK(input_layer_norm_coefficients != nullptr)

      LUCI_INTERPRETER_CHECK(Tensor::num_dims(input_layer_norm_coefficients) == 1);
      LUCI_INTERPRETER_CHECK(Tensor::dim(input_layer_norm_coefficients, 0) == n_cell);
      if (is_integer)
      {
        LUCI_INTERPRETER_CHECK(Tensor::element_type(input_layer_norm_coefficients) == DataType::S16);
      }
      else
      {
        LUCI_INTERPRETER_CHECK(Tensor::element_type(input_layer_norm_coefficients) ==
                               DataType::FLOAT32);
      }
    }

    LUCI_INTERPRETER_CHECK(Tensor::num_dims(forget_layer_norm_coefficients) == 1);
    LUCI_INTERPRETER_CHECK(Tensor::dim(forget_layer_norm_coefficients, 0) == n_cell);
    if (is_integer)
    {
      LUCI_INTERPRETER_CHECK(Tensor::element_type(forget_layer_norm_coefficients) == DataType::S16);
    }
    else
    {
      LUCI_INTERPRETER_CHECK(Tensor::element_type(forget_layer_norm_coefficients) == DataType::FLOAT32);
    }

    LUCI_INTERPRETER_CHECK(Tensor::num_dims(cell_layer_norm_coefficients) == 1);
    LUCI_INTERPRETER_CHECK(Tensor::dim(cell_layer_norm_coefficients, 0) == n_cell);
    if (is_integer)
    {
      LUCI_INTERPRETER_CHECK(Tensor::element_type(cell_layer_norm_coefficients) == DataType::S16);
    }
    else
    {
      LUCI_INTERPRETER_CHECK(Tensor::element_type(cell_layer_norm_coefficients) == DataType::FLOAT32);
    }

    LUCI_INTERPRETER_CHECK(Tensor::num_dims(output_layer_norm_coefficients) == 1);
    LUCI_INTERPRETER_CHECK(Tensor::dim(output_layer_norm_coefficients, 0) == n_cell);
    if (is_integer)
    {
      LUCI_INTERPRETER_CHECK(Tensor::element_type(output_layer_norm_coefficients) == DataType::S16);
    }
    else
    {
      LUCI_INTERPRETER_CHECK(Tensor::element_type(output_layer_norm_coefficients) == DataType::FLOAT32);
    }
  }



//  // Check the shape of input state tensors.
//  // These tensor may be 1D or 2D. It's fine as long as the total size is
//  // correct.
//  const Shape &output_state_shape = output_state()->shape();
//  const Shape &cell_state_shape = cell_state()->shape();
//  LUCI_INTERPRETER_CHECK(output_state_shape.num_elements() == n_batch * n_output);
//  LUCI_INTERPRETER_CHECK(cell_state_shape.num_elements() == n_batch * n_cell);
//
//  // Resize the output tensors.
//  Shape output_shape = Shape(input_shape.num_dims());
//  for (int i = 0; i < input_shape.num_dims() - 1; i++)
//  {
//    output_shape.dim(i) = input_shape.dim(i);
//  }
//  output_shape.dim(input_shape.num_dims() - 1) = n_output;
//  output()->resize(output_shape);
//
//  const bool use_cifg = (input_to_input_weights() == nullptr);
//
//  if (not is_integer)
//  {
//    // output_state and cell_state are variable tensor; use scratchpad.
//    getOutputTensors()[1]->resize(output_state_shape);
//    getOutputTensors()[2]->resize(cell_state_shape);
//
//    if (use_cifg)
//      getOutputTensors()[3]->resize({n_batch, n_cell * 3});
//    else
//      getOutputTensors()[3]->resize({n_batch, n_cell * 4});
//
//    // hybrid not supported
//    if (input_to_output_weights()->element_type() == DataType::U8 &&
//        input()->element_type() == DataType::FLOAT32)
//    {
//      // TODO support hybrid
//      assert(false && "Hybrid type is not currently supported");
//    }
//  }
//  else
//  {
//    // Integer UnidirectionalSequenceLSTM prepare function for 8x8->16.
//    // This code path needs 5 intermediate tensors per Op.
//    // Populate quantization parameters.
//    populate_quantized_lstm_params();
//
//    LUCI_INTERPRETER_CHECK(_outputs.size() == 6 + 2);
//
//    for (int i = 1; i < 6; ++i)
//    {
//      getOutputTensors().at(i)->resize({n_batch * n_cell});
//    }
//
//    populate_precomputed_zp_times_weight_with_bias();
//  }
}

void evalInt8(const circle::Operator *cur_op, BaseRuntimeGraph *runtime_graph,
              bool in_place)
{
  const auto input_index = cur_op->inputs()->operator[](0);
  const auto input_to_input_weights_index = cur_op->inputs()->operator[](1);
  const auto input_to_forget_weights_index = cur_op->inputs()->operator[](2);
  const auto input_to_cell_weights_index = cur_op->inputs()->operator[](3);
  const auto input_to_output_weights_index = cur_op->inputs()->operator[](4);
  assert(input_index != -1);
  assert(input_to_input_weights_index != -1);
  assert(input_to_forget_weights_index != -1);
  assert(input_to_cell_weights_index != -1);
  assert(input_to_output_weights_index != -1);
  const auto input = runtime_graph->getCircleTensorByIndex(input_index);
  const auto input_to_input_weights =
    runtime_graph->getCircleTensorByIndex(input_to_input_weights_index);
  const auto input_to_forget_weights =
    runtime_graph->getCircleTensorByIndex(input_to_forget_weights_index);
  const auto input_to_cell_weights =
    runtime_graph->getCircleTensorByIndex(input_to_cell_weights_index);
  const auto input_to_output_weights =
    runtime_graph->getCircleTensorByIndex(input_to_output_weights_index);

  const auto recurrent_to_input_weights_index = cur_op->inputs()->operator[](5);
  const auto recurrent_to_forget_weights_index = cur_op->inputs()->operator[](6);
  const auto recurrent_to_cell_weights_index = cur_op->inputs()->operator[](7);
  const auto recurrent_to_output_weights_index = cur_op->inputs()->operator[](8);
  assert(recurrent_to_input_weights_index != -1);
  assert(recurrent_to_forget_weights_index != -1);
  assert(recurrent_to_cell_weights_index != -1);
  assert(recurrent_to_output_weights_index != -1);
  const auto recurrent_to_input_weights =
    runtime_graph->getCircleTensorByIndex(recurrent_to_input_weights_index);
  const auto recurrent_to_forget_weights =
    runtime_graph->getCircleTensorByIndex(recurrent_to_forget_weights_index);
  const auto recurrent_to_cell_weights =
    runtime_graph->getCircleTensorByIndex(recurrent_to_cell_weights_index);
  const auto recurrent_to_output_weights =
    runtime_graph->getCircleTensorByIndex(recurrent_to_output_weights_index);

  const auto cell_to_input_weights_index = cur_op->inputs()->operator[](9);
  const auto cell_to_forget_weights_index = cur_op->inputs()->operator[](10);
  const auto cell_to_output_weights_index = cur_op->inputs()->operator[](11);
  assert(cell_to_input_weights_index != -1);
  assert(cell_to_forget_weights_index != -1);
  assert(cell_to_output_weights_index != -1);
  const auto cell_to_input_weights =
    runtime_graph->getCircleTensorByIndex(cell_to_input_weights_index);
  const auto cell_to_forget_weights =
    runtime_graph->getCircleTensorByIndex(cell_to_forget_weights_index);
  const auto cell_to_output_weights =
    runtime_graph->getCircleTensorByIndex(cell_to_output_weights_index);

  const auto input_gate_bias_index = cur_op->inputs()->operator[](12);
  const auto forget_gate_bias_index = cur_op->inputs()->operator[](13);
  const auto cell_gate_bias_index = cur_op->inputs()->operator[](14);
  const auto output_gate_bias_index = cur_op->inputs()->operator[](15);
  assert(input_gate_bias_index != -1);
  assert(forget_gate_bias_index != -1);
  assert(cell_gate_bias_index != -1);
  assert(output_gate_bias_index != -1);
  const auto input_gate_bias = runtime_graph->getCircleTensorByIndex(input_gate_bias_index);
  const auto forget_gate_bias = runtime_graph->getCircleTensorByIndex(forget_gate_bias_index);
  const auto cell_gate_bias = runtime_graph->getCircleTensorByIndex(cell_gate_bias_index);
  const auto output_gate_bias = runtime_graph->getCircleTensorByIndex(output_gate_bias_index);

  const auto projection_weights_index = cur_op->inputs()->operator[](16);
  const auto projection_bias_index = cur_op->inputs()->operator[](17);
  assert(projection_weights_index != -1);
  assert(projection_bias_index != -1);
  const auto projection_weights = runtime_graph->getCircleTensorByIndex(projection_weights_index);
  const auto projection_bias = runtime_graph->getCircleTensorByIndex(projection_bias_index);

  const auto output_state_index = cur_op->inputs()->operator[](18);
  const auto cell_state_index = cur_op->inputs()->operator[](19);
  assert(output_state_index != -1);
  assert(cell_state_index != -1);
  const auto output_state = runtime_graph->getCircleTensorByIndex(output_state_index);
  const auto cell_state = runtime_graph->getCircleTensorByIndex(cell_state_index);

  const auto input_layer_norm_coefficients_index = cur_op->inputs()->operator[](20);
  const auto forget_layer_norm_coefficients_index = cur_op->inputs()->operator[](21);
  const auto cell_layer_norm_coefficients_index = cur_op->inputs()->operator[](22);
  const auto output_layer_norm_coefficients_index = cur_op->inputs()->operator[](23);
  assert(input_layer_norm_coefficients_index != -1);
  assert(forget_layer_norm_coefficients_index != -1);
  assert(cell_layer_norm_coefficients_index != -1);
  assert(output_layer_norm_coefficients_index != -1);
  const auto input_layer_norm_coefficients =
    runtime_graph->getCircleTensorByIndex(input_layer_norm_coefficients_index);
  const auto forget_layer_norm_coefficients =
    runtime_graph->getCircleTensorByIndex(forget_layer_norm_coefficients_index);
  const auto cell_layer_norm_coefficients =
    runtime_graph->getCircleTensorByIndex(cell_layer_norm_coefficients_index);
  const auto output_layer_norm_coefficients =
    runtime_graph->getCircleTensorByIndex(output_layer_norm_coefficients_index);

  const auto output_index = cur_op->outputs()->operator[](0);
  assert(output_index != -1);
  const auto output = runtime_graph->getCircleTensorByIndex(output_index);

  const auto *options = cur_op->builtin_options_as_UnidirectionalSequenceLSTMOptions();

  const bool use_layer_norm = (forget_layer_norm_coefficients != nullptr);
  const bool time_major = options->time_major();
  const auto output_state_zero_point = Tensor::zero_point(output_state);

  const int n_batch = time_major ? Tensor::dim(input, 1) : Tensor::dim(input, 0);
  const int n_output = Tensor::dim(recurrent_to_output_weights, 1);
  const int n_cell = Tensor::dim(input_to_output_weights, 0);
  // Check the shape of input state tensors.
  // These tensor may be 1D or 2D. It's fine as long as the total size is
  // correct.
  LUCI_INTERPRETER_CHECK(Tensor::num_elements(output_state) == n_batch * n_output);
  LUCI_INTERPRETER_CHECK(Tensor::num_elements(cell_state) == n_batch * n_cell);

  // Resize the output tensors.
  for (int i = 0; i < Tensor::num_dims(input) - 1; i++)
  {
    LUCI_INTERPRETER_CHECK(Tensor::dim(output, i) == Tensor::dim(input, i));
  }
  LUCI_INTERPRETER_CHECK(Tensor::dim(output, Tensor::num_dims(input) - 1) == n_output);

  luci_interpreter_pal::lstm::IntegerLSTMParams integer_lstm_params;

  // Get intermediates
  // TODO add more for use_layer_norm
  const auto intermediate_index_0 = cur_op->intermediates()->operator[](0);
  assert(intermediate_index_0 != -1);
  const auto intermediate_tensor_0 = runtime_graph->getCircleTensorByIndex(intermediate_index_0);
  assert(intermediate_tensor_0 != nullptr);

  const bool use_cifg = (input_to_input_weights == nullptr);
  // Integer UnidirectionalSequenceLSTM prepare function for 8x8->16.
  // This code path needs 5 intermediate tensors per Op.
  // Populate quantization parameters.
  populate_quantized_lstm_params(
    input, input_to_input_weights, input_to_forget_weights, input_to_cell_weights,
    input_to_output_weights, recurrent_to_input_weights, recurrent_to_forget_weights,
    recurrent_to_cell_weights, recurrent_to_output_weights, cell_to_input_weights,
    cell_to_forget_weights, cell_to_output_weights, input_gate_bias, forget_gate_bias,
    cell_gate_bias, output_gate_bias, projection_weights, projection_bias, output_state, cell_state,
    input_layer_norm_coefficients, forget_layer_norm_coefficients, cell_layer_norm_coefficients,
    output_layer_norm_coefficients, output, intermediate_tensor_0, options, &integer_lstm_params);

  int8_t val = 0;

  auto scratch_0_data = new uint8_t[n_batch * n_cell];
  std::fill_n(scratch_0_data, n_cell * n_batch, val);

  auto scratch_1_data = new uint8_t[n_batch * n_cell];
  std::fill_n(scratch_1_data, n_cell * n_batch, val);

  auto scratch_2_data = new uint8_t[n_batch * n_cell];
  std::fill_n(scratch_2_data, n_cell * n_batch, val);

  auto scratch_3_data = new uint8_t[n_batch * n_cell];
  std::fill_n(scratch_3_data, n_cell * n_batch, val);

  auto scratch_4_data = new uint8_t[n_batch * n_cell];
  std::fill_n(scratch_4_data, n_cell * n_batch, val);

  // Fill states tensors
  std::fill_n(runtime_graph->getDataByTensor(output_state), n_cell * n_batch, val);
  std::fill_n(runtime_graph->getDataByTensor(cell_state), n_cell * n_batch, val);

  populate_precomputed_zp_times_weight_with_bias(input, input_to_input_weights, input_to_forget_weights, input_to_cell_weights,
                                                       input_to_output_weights, recurrent_to_input_weights, recurrent_to_forget_weights,
                                                       recurrent_to_cell_weights, recurrent_to_output_weights, cell_to_input_weights,
                                                       cell_to_forget_weights, cell_to_output_weights, input_gate_bias, forget_gate_bias,
                                                       cell_gate_bias, output_gate_bias, projection_weights, projection_bias, output_state, cell_state,
                                                       input_layer_norm_coefficients, forget_layer_norm_coefficients, cell_layer_norm_coefficients,
                                                       output_layer_norm_coefficients, output, intermediate_tensor_0, options, &integer_lstm_params,
                                                       runtime_graph);

  luci_interpreter_pal::eval_integer_8x8_16_lstm(
    input, input_to_input_weights, input_to_forget_weights, input_to_cell_weights,
    input_to_output_weights, recurrent_to_input_weights, recurrent_to_forget_weights,
    recurrent_to_cell_weights, recurrent_to_output_weights, cell_to_input_weights,
    cell_to_forget_weights, cell_to_output_weights, input_layer_norm_coefficients,
    forget_layer_norm_coefficients, cell_layer_norm_coefficients, output_layer_norm_coefficients,
    input_gate_bias, forget_gate_bias, cell_gate_bias, output_gate_bias, projection_weights,
    projection_bias, options,
    /*forward_sequence=*/true, time_major, integer_lstm_params, output_state_zero_point,
    output_state, cell_state, output, kernels::getTensorData<int16_t>(scratch_0_data),
    kernels::getTensorData<int16_t>(scratch_1_data), kernels::getTensorData<int16_t>(scratch_2_data),
    kernels::getTensorData<int16_t>(scratch_3_data), kernels::getTensorData<int8_t>(scratch_4_data),
    nullptr, runtime_graph);
}

//void evalFloat(const circle::Operator *cur_op, BaseRuntimeGraph *runtime_graph,
//               bool in_place)
//{
//  const bool time_major = params().time_major;
//  const bool use_layer_norm = (forget_layer_norm_coefficients() != nullptr);
//
//  const Tensor *t_input_layer_norm_coefficients =
//    use_layer_norm ? input_layer_norm_coefficients() : nullptr;
//  const Tensor *t_forget_layer_norm_coefficients =
//    use_layer_norm ? forget_layer_norm_coefficients() : nullptr;
//  const Tensor *t_cell_layer_norm_coefficients =
//    use_layer_norm ? cell_layer_norm_coefficients() : nullptr;
//  const Tensor *t_output_layer_norm_coefficients =
//    use_layer_norm ? output_layer_norm_coefficients() : nullptr;
//
//  Tensor *sp_output_state = getOutputTensors()[1];
//  Tensor *sp_cell_state = getOutputTensors()[2];
//  Tensor *sp_scratch_buffer = getOutputTensors()[3];
//
//  // Note: it is expected that output_state input variable tensor reset to zero,
//  // also expected that this variable tensor doesn't have buffer
//  auto scratchpad_data = getTensorData<float>(sp_output_state);
//  std::fill_n(scratchpad_data, sp_output_state->shape().num_elements(), 0);
//  scratchpad_data = getTensorData<float>(sp_cell_state);
//  std::fill_n(scratchpad_data, sp_cell_state->shape().num_elements(), 0);
//  scratchpad_data = getTensorData<float>(sp_scratch_buffer);
//  std::fill_n(scratchpad_data, sp_scratch_buffer->shape().num_elements(), 0);
//
//  TfLiteLSTMParams lstm_params{};
//  lstm_params.activation = getTfLiteActivation(params().activation);
//  lstm_params.cell_clip = params().cell_clip;
//  lstm_params.proj_clip = params().proj_clip;
//  lstm_params.asymmetric_quantize_inputs = params().asymmetric_quantize_inputs;
//
//  EvalFloat(input(), input_to_input_weights(), input_to_forget_weights(),
//                  input_to_cell_weights(), input_to_output_weights(),
//
//                  recurrent_to_input_weights(), recurrent_to_forget_weights(),
//                  recurrent_to_cell_weights(), recurrent_to_output_weights(),
//
//                  cell_to_input_weights(), cell_to_forget_weights(), cell_to_output_weights(),
//
//                  t_input_layer_norm_coefficients, t_forget_layer_norm_coefficients,
//                  t_cell_layer_norm_coefficients, t_output_layer_norm_coefficients,
//                  /*aux_input=*/nullptr,
//                  /*aux_input_to_input_weights=*/nullptr,
//                  /*aux_input_to_forget_weights=*/nullptr,
//                  /*aux_input_to_cell_weights=*/nullptr,
//                  /*aux_input_to_output_weights=*/nullptr, input_gate_bias(), forget_gate_bias(),
//                  cell_gate_bias(), output_gate_bias(),
//
//                  projection_weights(), projection_bias(), &lstm_params,
//                  /*forward_sequence=*/true, time_major,
//                  /*output_offset=*/0, sp_scratch_buffer, sp_output_state, sp_cell_state, output());
//}

void execute_kernel_CircleUnidirectionalSequenceLSTM(const circle::Operator *cur_op, BaseRuntimeGraph *runtime_graph,
                                               bool in_place)
{
  const auto input_index = cur_op->inputs()->operator[](0);

  assert(input_index != -1);

  const auto input = runtime_graph->getCircleTensorByIndex(input_index);

  switch (Tensor::element_type(input))
  {
    case DataType::FLOAT32:
      //evalFloat(cur_op, runtime_graph, in_place);
      break;
    case DataType::S8:
      evalInt8(cur_op, runtime_graph, in_place);
      break;
    default:
      assert(false && "Unsupported type.");
  }
}

} // namespace luci_interpreter
