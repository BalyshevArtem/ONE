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

namespace luci_interpreter
{
namespace
{

// Create parameters for element wise multiplication that happens in a) cell
// state update ; b) hidden state update
// Note that all the output of gates are symmetrically quantized so only scales
// are required for input. However, during the hidden state update phase, the
// output is the updated hidden state, which is asymmetrically quantized. Thus
// output may require zero point
lstm::ArithmeticParams create_inter_gate_params(const float input1_scale,
                                                const float input2_scale,
                                                const float output_scale,
                                                const DataType output_type,
                                                const int output_zp)
{
  lstm::ArithmeticParams op_params;
  if (output_type == DataType::S16)
  {
    op_params.quantized_activation_min = std::numeric_limits<int16_t>::min();
    op_params.quantized_activation_max = std::numeric_limits<int16_t>::max();
  } else if (output_type == DataType::S8)
  {
    op_params.quantized_activation_min = std::numeric_limits<int8_t>::min();
    op_params.quantized_activation_max = std::numeric_limits<int8_t>::max();
  }

  op_params.input1_offset = 0;  // symmetric
  op_params.input2_offset = 0;  // symmetric
  op_params.output_offset = output_zp;

  const double input_product_scale =
    static_cast<double>(input1_scale) * static_cast<double>(input2_scale);
  double effective_scale =
    input_product_scale / static_cast<double>(output_scale);

  kernels::quantizeMultiplier(effective_scale, &op_params.output_multiplier,
                              &op_params.output_shift);
  return op_params;
}

void create_gate_params(const circle::Tensor *input, const circle::Tensor *input_weight,
                        const circle::Tensor *input_bias, const circle::Tensor *hidden_state,
                        const circle::Tensor *hidden_state_weight,
                        const float nonlinear_activation_input_scale, const DataType cell_type,
                        lstm::GateParameters *gate_params)
{
  // Input CalculateOpDataFullyConnected
  {
    lstm::FullyConnectedQuantizedParams input_gate_params;
    double real_multiplier = 0.0;
    int output_shift;
    int32_t output_activation_min;
    int32_t output_activation_max;
    int32_t output_multiplier;
    real_multiplier = kernels::getQuantizedConvolutionMultipler(
      Tensor::scale(input), Tensor::scale(input_weight), nonlinear_activation_input_scale);
    kernels::quantizeMultiplier(real_multiplier, &output_multiplier, &output_shift);
    kernels::calculateActivationRangeQuantized(FusedActFunc::NONE,
                                               0, nonlinear_activation_input_scale, cell_type, &output_activation_min,
                                               &output_activation_max);

    input_gate_params.output_shift = output_shift;
    input_gate_params.output_multiplier = output_multiplier;
    input_gate_params.quantized_activation_max = output_activation_max;
    input_gate_params.quantized_activation_min = output_activation_min;
    input_gate_params.input_offset = -Tensor::zero_point(input);
    input_gate_params.weights_offset = -Tensor::zero_point(input_weight);
    input_gate_params.output_offset = 0;

    gate_params->input_fc_params = input_gate_params;
  }

  // Recurrent CalculateOpDataFullyConnected
  {
    lstm::FullyConnectedQuantizedParams recurrent_gate_params;
    double real_multiplier = 0.0;
    int output_shift;
    int32_t output_activation_min;
    int32_t output_activation_max;
    int32_t output_multiplier;
    real_multiplier = kernels::getQuantizedConvolutionMultipler(
      Tensor::scale(hidden_state), Tensor::scale(hidden_state_weight), nonlinear_activation_input_scale);
    kernels::quantizeMultiplier(real_multiplier, &output_multiplier, &output_shift);
    kernels::calculateActivationRangeQuantized(FusedActFunc::NONE,
                                               0, nonlinear_activation_input_scale, cell_type, &output_activation_min,
                                               &output_activation_max);

    recurrent_gate_params.output_shift = output_shift;
    recurrent_gate_params.output_multiplier = output_multiplier;
    recurrent_gate_params.quantized_activation_max = output_activation_max;
    recurrent_gate_params.quantized_activation_min = output_activation_min;
    recurrent_gate_params.input_offset = -Tensor::zero_point(hidden_state);
    recurrent_gate_params.weights_offset = -Tensor::zero_point(hidden_state_weight);
    recurrent_gate_params.output_offset = 0;

    gate_params->input_fc_params = recurrent_gate_params;
  }
}

void prepare_gate_params_integer(lstm::LSTMStruct *lstm_struct, lstm::QuantLSTMParameters *quant_lstm_params)
{
  float nonlinear_input_scale = 0.00024414062;  // 2^-12 Q3.12 -> Q0.15

  create_gate_params(lstm_struct->input(), lstm_struct->input_to_forget_weights(), lstm_struct->forget_gate_bias(),
                     lstm_struct->output_state(), lstm_struct->recurrent_to_forget_weights(), nonlinear_input_scale, DataType::S16,
                     &quant_lstm_params->forget_gate_parameters);

  create_gate_params(lstm_struct->input(), lstm_struct->input_to_input_weights(), lstm_struct->input_gate_bias(),
                     lstm_struct->output_state(), lstm_struct->recurrent_to_input_weights(), nonlinear_input_scale, DataType::S16,
                     &quant_lstm_params->input_gate_parameters);

  lstm::GateParameters cell_gate_parameters;
  create_gate_params(lstm_struct->input(), lstm_struct->input_to_cell_weights(), lstm_struct->cell_gate_bias(),
                     lstm_struct->output_state(), lstm_struct->recurrent_to_cell_weights(), nonlinear_input_scale, DataType::S16,
                     &quant_lstm_params->cell_gate_parameters);

  lstm::GateParameters output_gate_parameters;
  create_gate_params(lstm_struct->input(), lstm_struct->input_to_output_weights(), lstm_struct->output_gate_bias(),
                     lstm_struct->output_state(), lstm_struct->recurrent_to_output_weights(), nonlinear_input_scale, DataType::S16,
                     &quant_lstm_params->output_gate_parameters);

  // Inter gate multiplication parameters
  float nonlinear_output_scale = 0.00003051757;  // 2^-15 Q3.12 -> Q0.15
  float cell_state_scale = Tensor::scale(lstm_struct->cell_state()); //lstm_tensors.CellStateTensor()->params.scale;
  // forget gate output (nonlinear output) x cell state -> cell state
  quant_lstm_params->inter_gate_parameters.forget_cell_mul_params = create_inter_gate_params(nonlinear_output_scale, nonlinear_output_scale, cell_state_scale, DataType::S16, 0);

  // input gate output x cell gate output -> cell state
  quant_lstm_params->inter_gate_parameters.input_mul_params = create_inter_gate_params(nonlinear_output_scale, cell_state_scale, cell_state_scale, DataType::S16, 0);

  // tanh output x output gate output -> hidden state (potentially asymmetric)
  quant_lstm_params->inter_gate_parameters.output_mul_params = create_inter_gate_params(nonlinear_output_scale, nonlinear_output_scale, Tensor::scale(lstm_struct->output_state()),
                                                                 Tensor::element_type(lstm_struct->output_state()), Tensor::zero_point(lstm_struct->output_state()));
}

void validate_weight_tensor_size(const circle::Tensor *weight_tensor, int dim1_size, int dim2_size)
{
  LUCI_INTERPRETER_CHECK(Tensor::num_dims(weight_tensor) == 2);
  LUCI_INTERPRETER_CHECK(Tensor::dim(weight_tensor, 0) == dim1_size);
  LUCI_INTERPRETER_CHECK(Tensor::dim(weight_tensor, 1) == dim2_size);
}

void validate_tensors_size(lstm::LSTMStruct *lstm_struct, const bool time_major)
{
  const auto batch_size = time_major ? Tensor::dim(lstm_struct->input(), 1) : Tensor::dim(lstm_struct->input(), 0);

  const auto input_dimension = Tensor::dim(lstm_struct->input(), 2);
  const auto state_dimension = Tensor::dim(lstm_struct->output_state(), 1);

  // Input FC weights
  for (int32_t i = 1; i < 5; i++)
  {
    validate_weight_tensor_size(lstm_struct->get_internal_tensor(i), state_dimension, input_dimension);
  }

  // Recurrent FC weights
  for (int32_t i = 5; i < 9; i++)
  {
    validate_weight_tensor_size(lstm_struct->get_internal_tensor(i), state_dimension, state_dimension);
  }

  // Biases
  for (int32_t i = 12; i < 16; i++)
  {
    LUCI_INTERPRETER_CHECK(Tensor::num_dims(lstm_struct->get_internal_tensor(i)) == 1);
    LUCI_INTERPRETER_CHECK(Tensor::dim(lstm_struct->get_internal_tensor(i), 0) == state_dimension);
  }

  // Check the shape of input state tensors.
  // These tensor may be 1D or 2D. It's fine as long as the total size is
  // correct.
  LUCI_INTERPRETER_CHECK(Tensor::num_elements(lstm_struct->output_state()) == batch_size * state_dimension);
  LUCI_INTERPRETER_CHECK(Tensor::num_elements(lstm_struct->cell_state()) == batch_size * state_dimension);

  // Check the shape of output tensor against that of input tensor
  LUCI_INTERPRETER_CHECK(Tensor::num_dims(lstm_struct->output_state()) == 3);
  LUCI_INTERPRETER_CHECK(Tensor::dim(lstm_struct->input(), 0) == Tensor::dim(lstm_struct->output(), 0));
  LUCI_INTERPRETER_CHECK(Tensor::dim(lstm_struct->input(), 1) == Tensor::dim(lstm_struct->output(), 1));
  LUCI_INTERPRETER_CHECK(Tensor::dim(lstm_struct->output(), 2) == state_dimension);
}

#ifndef DIS_QUANT
void precompute_zero_point_times_weight_with_bias(
  int32_t zero_point, const circle::Tensor *weight_tensor, const uint8_t *weight_tensor_data, const uint8_t *bias_tensor_data,
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

//void populate_precomputed_zp_times_weight_with_bias(lstm::LSTMStruct *lstm_struct,
//                                                    const circle::Tensor *intermediate_tensor_0,
//                                                    lstm::IntegerLSTMParams *integer_lstm_params,
//                                                    BaseRuntimeGraph *runtime_graph)
//{
//  const int32_t input_zero_point = -Tensor::zero_point(lstm_struct->input);
//  const int32_t output_state_zero_point = -Tensor::zero_point(lstm_struct->output_state);
//
//  const int32_t hidden_zp = Tensor::zero_point(intermediate_tensor_0);
//
//  // Get bias and perform zero point calculation.
//  // When there is layer normalization, the gate bias does not apply to matmul
//  // directly:
//  //      y = ln(w * x + w * r + w * c) + b.
//
//  bool is_layer_norm = (lstm_struct->forget_layer_norm_coefficients != nullptr);
//
//  {
//    const auto forget_gate_bias_tmp_data =
//      is_layer_norm ? nullptr : runtime_graph->getConstDataByTensor(lstm_struct->forget_gate_bias);
//
//    const auto input_to_forget_weights_data =
//      runtime_graph->getConstDataByTensor(lstm_struct->input_to_forget_weights);
//
//    precompute_zero_point_times_weight_with_bias(
//      input_zero_point, lstm_struct->input_to_forget_weights, input_to_forget_weights_data,
//      forget_gate_bias_tmp_data, integer_lstm_params->input_to_forget_effective_bias);
//  }
//
//  {
//    const auto recurrent_to_forget_weights_data =
//      runtime_graph->getConstDataByTensor(lstm_struct->recurrent_to_forget_weights);
//
//    precompute_zero_point_times_weight_with_bias(
//      output_state_zero_point, lstm_struct->recurrent_to_forget_weights, recurrent_to_forget_weights_data,
//      nullptr, integer_lstm_params->recurrent_to_forget_effective_bias);
//  }
//
//  {
//    // Modulation gate.
//    const auto cell_gate_bias_tmp =
//      is_layer_norm ? nullptr : runtime_graph->getDataByTensor(lstm_struct->cell_gate_bias);
//
//    const auto input_to_cell_weights_data = runtime_graph->getConstDataByTensor(lstm_struct->input_to_cell_weights);
//
//    precompute_zero_point_times_weight_with_bias(input_zero_point, lstm_struct->input_to_cell_weights,
//                                                 input_to_cell_weights_data, cell_gate_bias_tmp,
//                                                 integer_lstm_params->input_to_cell_effective_bias);
//  }
//
//  {
//    const auto recurrent_to_cell_weights_data = runtime_graph->getConstDataByTensor(lstm_struct->recurrent_to_cell_weights);
//
//    precompute_zero_point_times_weight_with_bias(
//      output_state_zero_point, lstm_struct->recurrent_to_cell_weights, recurrent_to_cell_weights_data, nullptr,
//      integer_lstm_params->recurrent_to_cell_effective_bias);
//  }
//
//  {
//    // Output gate.
//    const auto output_gate_bias_tmp = is_layer_norm ? nullptr : runtime_graph->getConstDataByTensor(lstm_struct->output_gate_bias);
//
//    const auto input_to_output_weights_data = runtime_graph->getConstDataByTensor(lstm_struct->input_to_output_weights);
//
//    precompute_zero_point_times_weight_with_bias(
//      input_zero_point, lstm_struct->input_to_output_weights, input_to_output_weights_data, output_gate_bias_tmp,
//      integer_lstm_params->input_to_output_effective_bias);
//  }
//
//  {
//    const auto recurrent_to_output_weights_data = runtime_graph->getConstDataByTensor(lstm_struct->recurrent_to_output_weights);
//
//    precompute_zero_point_times_weight_with_bias(
//      output_state_zero_point, lstm_struct->recurrent_to_output_weights, recurrent_to_output_weights_data, nullptr,
//      integer_lstm_params->recurrent_to_output_effective_bias);
//  }
//
//  {
//    // Input gate. The calculation is only meaningful for non-cifg case.
//    const auto input_gate_bias_tmp = is_layer_norm ? nullptr : runtime_graph->getConstDataByTensor(lstm_struct->input_gate_bias);
//
//    const auto input_to_input_weights_data = runtime_graph->getConstDataByTensor(lstm_struct->input_to_input_weights);
//
//    precompute_zero_point_times_weight_with_bias(
//      input_zero_point, lstm_struct->input_to_input_weights, input_to_input_weights_data, input_gate_bias_tmp,
//      integer_lstm_params->input_to_input_effective_bias);
//  }
//
//  {
//    const auto recurrent_to_input_weights_data = runtime_graph->getConstDataByTensor(lstm_struct->recurrent_to_input_weights);
//
//    precompute_zero_point_times_weight_with_bias(
//      output_state_zero_point, lstm_struct->recurrent_to_input_weights, recurrent_to_input_weights_data, nullptr,
//      integer_lstm_params->recurrent_to_input_effective_bias);
//  }
//
//  {
//    const auto projection_weights_data = runtime_graph->getConstDataByTensor(lstm_struct->projection_weights);
//
//    // Projection bias. The calculation is only meaningful for with projection.
//    precompute_zero_point_times_weight_with_bias(hidden_zp, lstm_struct->projection_weights, projection_weights_data, runtime_graph->getConstDataByTensor(lstm_struct->projection_bias),
//                                                 integer_lstm_params->projection_effective_bias);
//  }
//}

//void populate_quantized_lstm_params(luci_interpreter_pal::lstm::LSTMStruct *lstm_struct,
//                                    const circle::Tensor *intermediate_tensor_0,
//                                    luci_interpreter_pal::lstm::IntegerLSTMParams *integer_lstm_params)
//{
//  const float cell_clip = lstm_struct->options->cell_clip();
//  const float proj_clip = lstm_struct->options->proj_clip();
//
//  LUCI_INTERPRETER_CHECK(lstm_struct->cell_state != nullptr);
//  LUCI_INTERPRETER_CHECK(Tensor::scales(lstm_struct->cell_state).size() > 0 and
//                         Tensor::zero_points(lstm_struct->cell_state).size() > 0);
//
//  LUCI_INTERPRETER_CHECK(Tensor::scales(lstm_struct->output).size() > 0 and Tensor::zero_points(lstm_struct->output).size() > 0);
//
//  if (cell_clip > 0.0f)
//  {
//    integer_lstm_params->quantized_cell_clip = static_cast<int16_t>(
//      std::min(std::max(cell_clip / Tensor::scale(lstm_struct->cell_state), -32768.0f), 32767.0f));
//  }
//  else
//  {
//    integer_lstm_params->quantized_cell_clip = 0;
//  }
//
//  if (proj_clip > 0.0f)
//  {
//    integer_lstm_params->quantized_proj_clip =
//      static_cast<int8_t>(std::min(std::max(proj_clip / Tensor::scale(lstm_struct->output), -128.0f), 127.0f));
//  }
//  else
//  {
//    integer_lstm_params->quantized_proj_clip = 0;
//  }
//
//  // Calculate effective scales.
//  bool use_layer_norm = (lstm_struct->forget_layer_norm_coefficients != nullptr);
//
//  // Since we have already checked that weights are all there or none, we can
//  // check the existence of only one to get the condition.
//  const bool use_cifg = (lstm_struct->input_to_input_weights == nullptr);
//  const bool use_peephole = (lstm_struct->cell_to_output_weights != nullptr);
//  const bool use_projection = (lstm_struct->projection_weights != nullptr);
//
//  // Get intermediate scales and zero points.
//  float intermediate_scale[5];
//  int32_t intermediate_zp[5];
//
//  for (int i = 0; i < 4; ++i)
//  {
//    if (use_layer_norm)
//    {
//      assert(false);
//      // TODO: support it
//    }
//    else
//    {
//      intermediate_scale[i] = std::pow(2.0f, -12.0f);
//      intermediate_zp[i] = 0;
//    }
//  }
//  intermediate_scale[4] = Tensor::scale(intermediate_tensor_0);
//  intermediate_zp[4] = Tensor::zero_point(intermediate_tensor_0);
//
//  // Scales.
//  const float default_scale = 1.0;
//  float input_scale = default_scale;
//  float input_to_input_weight_scale = default_scale;
//  float recurrent_to_input_weight_scale = default_scale;
//  float cell_to_input_weight_scale = default_scale;
//  float input_to_forget_weight_scale = default_scale;
//  float recurrent_to_forget_weight_scale = default_scale;
//  float cell_to_forget_weight_scale = default_scale;
//  float input_to_cell_weight_scale = default_scale;
//  float recurrent_to_cell_weight_scale = default_scale;
//  float input_to_output_weight_scale = default_scale;
//  float recurrent_to_output_weight_scale = default_scale;
//  float cell_to_output_weight_scale = default_scale;
//  float projection_weight_scale = default_scale;
//  float layer_norm_input_scale = default_scale;
//  float layer_norm_forget_scale = default_scale;
//  float layer_norm_cell_scale = default_scale;
//  float layer_norm_output_scale = default_scale;
//  float output_state_scale = default_scale;
//  int cell_scale = 1;
//
//  // Effective scales.
//  float effective_input_to_input_scale = default_scale;
//  float effective_recurrent_to_input_scale = default_scale;
//  float effective_cell_to_input_scale = default_scale;
//  float effective_input_to_forget_scale = default_scale;
//  float effective_recurrent_to_forget_scale = default_scale;
//  float effective_cell_to_forget_scale = default_scale;
//  float effective_input_to_cell_scale = default_scale;
//  float effective_recurrent_to_cell_scale = default_scale;
//  float effective_input_to_output_scale = default_scale;
//  float effective_recurrent_to_output_scale = default_scale;
//  float effective_cell_to_output_scale = default_scale;
//  float effective_proj_scale = default_scale;
//  float effective_hidden_scale = default_scale;
//
//  // Populate scales.
//  if (!use_cifg)
//  {
//    input_to_input_weight_scale = Tensor::scale(lstm_struct->input_to_input_weights);
//    recurrent_to_input_weight_scale = Tensor::scale(lstm_struct->recurrent_to_input_weights);
//  }
//
//  if (use_peephole)
//  {
//    if (!use_cifg)
//    {
//      cell_to_input_weight_scale = Tensor::scale(lstm_struct->cell_to_input_weights);
//    }
//    cell_to_forget_weight_scale = Tensor::scale(lstm_struct->cell_to_forget_weights);
//    cell_to_output_weight_scale = Tensor::scale(lstm_struct->cell_to_output_weights);
//  }
//
//  if (use_layer_norm)
//  {
//    if (!use_cifg)
//    {
//      layer_norm_input_scale = Tensor::scale(lstm_struct->input_layer_norm_coefficients);
//    }
//    layer_norm_forget_scale = Tensor::scale(lstm_struct->forget_layer_norm_coefficients);
//    layer_norm_cell_scale = Tensor::scale(lstm_struct->cell_layer_norm_coefficients);
//    layer_norm_output_scale = Tensor::scale(lstm_struct->output_layer_norm_coefficients);
//  }
//
//  if (use_projection)
//  {
//    projection_weight_scale = Tensor::scale(lstm_struct->projection_weights);
//  }
//  output_state_scale = Tensor::scale(lstm_struct->output_state);
//
//  input_to_forget_weight_scale = Tensor::scale(lstm_struct->input_to_forget_weights);
//  input_to_cell_weight_scale = Tensor::scale(lstm_struct->input_to_cell_weights);
//  input_to_output_weight_scale = Tensor::scale(lstm_struct->input_to_output_weights);
//  recurrent_to_forget_weight_scale = Tensor::scale(lstm_struct->recurrent_to_forget_weights);
//  recurrent_to_cell_weight_scale = Tensor::scale(lstm_struct->recurrent_to_cell_weights);
//  recurrent_to_output_weight_scale = Tensor::scale(lstm_struct->recurrent_to_output_weights);
//
//  // Check cell state (already used above)
//  {
//    const float x_log2 = std::log(Tensor::scale(lstm_struct->cell_state)) * (1.0f / std::log(2.0f));
//    const float x_log2_rounded = std::round(x_log2);
//    const float x_log2_fracpart = x_log2 - x_log2_rounded;
//    cell_scale = static_cast<int>(x_log2_rounded);
//
//    LUCI_INTERPRETER_CHECK(std::abs(x_log2_fracpart) < 1e-3f);
//  }
//  integer_lstm_params->cell_scale = cell_scale;
//  input_scale = Tensor::scale(lstm_struct->input);
//
//  // Calculate effective scales.
//  if (!use_cifg)
//  {
//    effective_input_to_input_scale =
//      input_to_input_weight_scale * input_scale / intermediate_scale[0];
//    effective_recurrent_to_input_scale =
//      recurrent_to_input_weight_scale * output_state_scale / intermediate_scale[0];
//  }
//  effective_input_to_forget_scale =
//    input_to_forget_weight_scale * input_scale / intermediate_scale[1];
//  effective_recurrent_to_forget_scale =
//    recurrent_to_forget_weight_scale * output_state_scale / intermediate_scale[1];
//
//  effective_input_to_cell_scale = input_to_cell_weight_scale * input_scale / intermediate_scale[2];
//  effective_recurrent_to_cell_scale =
//    recurrent_to_cell_weight_scale * output_state_scale / intermediate_scale[2];
//
//  effective_input_to_output_scale =
//    input_to_output_weight_scale * input_scale / intermediate_scale[3];
//  effective_recurrent_to_output_scale =
//    recurrent_to_output_weight_scale * output_state_scale / intermediate_scale[3];
//
//  effective_hidden_scale = std::pow(2.0f, -15.0f) / intermediate_scale[4] * std::pow(2.0f, -15.0f);
//
//  effective_proj_scale = projection_weight_scale * intermediate_scale[4] / output_state_scale;
//
//  if (use_peephole)
//  {
//    if (!use_cifg)
//    {
//      effective_cell_to_input_scale = std::pow(2.0f, static_cast<float>(cell_scale)) *
//                                      cell_to_input_weight_scale / intermediate_scale[0];
//    }
//    effective_cell_to_forget_scale = std::pow(2.0f, static_cast<float>(cell_scale)) *
//                                     cell_to_forget_weight_scale / intermediate_scale[1];
//    effective_cell_to_output_scale = std::pow(2.0f, static_cast<float>(cell_scale)) *
//                                     cell_to_output_weight_scale / intermediate_scale[3];
//  }
//
//  // Decompose scales.
//  int shift_output = 0;
//
//  kernels::quantizeMultiplier(static_cast<double>(effective_input_to_input_scale),
//                              &integer_lstm_params->effective_input_to_input_scale_a, &shift_output);
//  integer_lstm_params->effective_input_to_input_scale_b = static_cast<int32_t>(shift_output);
//
//  kernels::quantizeMultiplier(static_cast<double>(effective_recurrent_to_input_scale),
//                              &integer_lstm_params->effective_recurrent_to_input_scale_a, &shift_output);
//  integer_lstm_params->effective_recurrent_to_input_scale_b = static_cast<int32_t>(shift_output);
//
//  kernels::quantizeMultiplier(static_cast<double>(effective_cell_to_input_scale),
//                              &integer_lstm_params->effective_cell_to_input_scale_a, &shift_output);
//  integer_lstm_params->effective_cell_to_input_scale_b = static_cast<int32_t>(shift_output);
//
//  kernels::quantizeMultiplier(static_cast<double>(effective_input_to_forget_scale),
//                              &integer_lstm_params->effective_input_to_forget_scale_a, &shift_output);
//  integer_lstm_params->effective_input_to_forget_scale_b = static_cast<int32_t>(shift_output);
//
//  kernels::quantizeMultiplier(static_cast<double>(effective_recurrent_to_forget_scale),
//                              &integer_lstm_params->effective_recurrent_to_forget_scale_a, &shift_output);
//  integer_lstm_params->effective_recurrent_to_forget_scale_b = static_cast<int32_t>(shift_output);
//
//  kernels::quantizeMultiplier(static_cast<double>(effective_cell_to_forget_scale),
//                              &integer_lstm_params->effective_cell_to_forget_scale_a, &shift_output);
//  integer_lstm_params->effective_cell_to_forget_scale_b = static_cast<int32_t>(shift_output);
//  kernels::quantizeMultiplier(static_cast<double>(effective_input_to_cell_scale),
//                              &integer_lstm_params->effective_input_to_cell_scale_a, &shift_output);
//  integer_lstm_params->effective_input_to_cell_scale_b = static_cast<int32_t>(shift_output);
//
//  kernels::quantizeMultiplier(static_cast<double>(effective_recurrent_to_cell_scale),
//                              &integer_lstm_params->effective_recurrent_to_cell_scale_a, &shift_output);
//  integer_lstm_params->effective_recurrent_to_cell_scale_b = static_cast<int32_t>(shift_output);
//
//  kernels::quantizeMultiplier(static_cast<double>(effective_input_to_output_scale),
//                              &integer_lstm_params->effective_input_to_output_scale_a, &shift_output);
//  integer_lstm_params->effective_input_to_output_scale_b = static_cast<int32_t>(shift_output);
//
//  kernels::quantizeMultiplier(static_cast<double>(effective_recurrent_to_output_scale),
//                              &integer_lstm_params->effective_recurrent_to_output_scale_a, &shift_output);
//  integer_lstm_params->effective_recurrent_to_output_scale_b = static_cast<int32_t>(shift_output);
//
//  kernels::quantizeMultiplier(static_cast<double>(effective_cell_to_output_scale),
//                              &integer_lstm_params->effective_cell_to_output_scale_a, &shift_output);
//  integer_lstm_params->effective_cell_to_output_scale_b = static_cast<int32_t>(shift_output);
//
//  kernels::quantizeMultiplier(static_cast<double>(effective_proj_scale),
//                              &integer_lstm_params->effective_proj_scale_a, &shift_output);
//  integer_lstm_params->effective_proj_scale_b = static_cast<int32_t>(shift_output);
//
//  kernels::quantizeMultiplier(static_cast<double>(effective_hidden_scale),
//                              &integer_lstm_params->effective_hidden_scale_a, &shift_output);
//  integer_lstm_params->effective_hidden_scale_b = static_cast<int32_t>(shift_output);
//
//  kernels::quantizeMultiplier(static_cast<double>(layer_norm_input_scale),
//                              &integer_lstm_params->layer_norm_input_scale_a, &shift_output);
//  integer_lstm_params->layer_norm_input_scale_b = static_cast<int32_t>(shift_output);
//
//  kernels::quantizeMultiplier(static_cast<double>(layer_norm_forget_scale),
//                              &integer_lstm_params->layer_norm_forget_scale_a, &shift_output);
//  integer_lstm_params->layer_norm_forget_scale_b = static_cast<int32_t>(shift_output);
//
//  kernels::quantizeMultiplier(static_cast<double>(layer_norm_cell_scale),
//                              &integer_lstm_params->layer_norm_cell_scale_a, &shift_output);
//  integer_lstm_params->layer_norm_cell_scale_b = static_cast<int32_t>(shift_output);
//
//  kernels::quantizeMultiplier(static_cast<double>(layer_norm_output_scale),
//                              &integer_lstm_params->layer_norm_output_scale_a, &shift_output);
//  integer_lstm_params->layer_norm_output_scale_b = static_cast<int32_t>(shift_output);
//
//  integer_lstm_params->hidden_zp = intermediate_zp[4];
//
//  // 10000 is used to make sure the kernel logic does not overflow.
//  if (!use_cifg)
//  {
//    integer_lstm_params->input_variance_guard =
//      std::max(1, static_cast<int>(10000 * layer_norm_input_scale));
//  }
//  integer_lstm_params->forget_variance_guard =
//    std::max(1, static_cast<int>(10000 * layer_norm_forget_scale));
//  integer_lstm_params->cell_variance_guard =
//    std::max(1, static_cast<int>(10000 * layer_norm_cell_scale));
//  integer_lstm_params->output_variance_guard =
//    std::max(1, static_cast<int>(10000 * layer_norm_output_scale));
//}

#endif // DIS_QUANT

//using namespace tflite;

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

void configure_kernel_CircleUnidirectionalSequenceLSTM(const circle::Operator *cur_op, BaseRuntimeGraph *runtime_graph)
{
  lstm::LSTMStruct lstm_struct(cur_op, runtime_graph);

  LUCI_INTERPRETER_CHECK(Tensor::element_type(lstm_struct.input()) == DataType::FLOAT32 or
                           Tensor::element_type(lstm_struct.input()) == DataType::S8);

  lstm_struct.validateTensorTypes();

  const bool time_major = lstm_struct.options->time_major();

  validate_tensors_size(&lstm_struct, time_major);

  // No peephole
  for (int32_t i = 9; i < 12; ++i)
    LUCI_INTERPRETER_CHECK(lstm_struct.get_internal_tensor(i) == nullptr);

  // No projection
  for (int32_t i = 16; i < 18; ++i)
    LUCI_INTERPRETER_CHECK(lstm_struct.get_internal_tensor(i) == nullptr);

  // No internal layer norm
  for (int32_t i = 20; i < 24; ++i)
    LUCI_INTERPRETER_CHECK(lstm_struct.get_internal_tensor(i) == nullptr);
}

void evalInt8(const circle::Operator *cur_op, BaseRuntimeGraph *runtime_graph,
              bool in_place)
{
  lstm::LSTMStruct lstm_struct(cur_op, runtime_graph);

  lstm::QuantLSTMParameters quant_lstm_params;
  prepare_gate_params_integer(&lstm_struct, &quant_lstm_params);

  const bool time_major = lstm_struct.options->time_major();
  const auto batch_size = time_major ? Tensor::dim(lstm_struct.input(), 1) : Tensor::dim(lstm_struct.input(), 0);
  const auto state_dimension = Tensor::dim(lstm_struct.output_state(), 1);
  const auto cell_state_type_size = getDataTypeSize(Tensor::element_type(lstm_struct.cell_state()));

  auto scratch_0_data = std::make_unique<uint8_t[]>(batch_size * state_dimension * cell_state_type_size);//new uint8_t[n_batch * n_cell];
  //std::fill_n(scratch_0_data.get(), n_cell * n_batch, 0);

  auto scratch_1_data = std::make_unique<uint8_t[]>(batch_size * state_dimension * cell_state_type_size);//new uint8_t[n_batch * n_cell];
  //std::fill_n(scratch_1_data.get(), n_cell * n_batch, 0);

  auto scratch_2_data = std::make_unique<uint8_t[]>(batch_size * state_dimension * cell_state_type_size);//new uint8_t[n_batch * n_cell];
  //std::fill_n(scratch_2_data.get(), n_cell * n_batch, 0);

  auto scratch_3_data = std::make_unique<uint8_t[]>(batch_size * state_dimension * cell_state_type_size);//new uint8_t[n_batch * n_cell];
  //std::fill_n(scratch_3_data.get(), n_cell * n_batch, 0);

  // Fill states tensors
  auto element_size = getDataTypeSize(Tensor::element_type(lstm_struct.output_state()));
  auto output_state_data = std::make_unique<int8_t[]>(Tensor::num_elements(lstm_struct.output_state()));
  std::fill_n(output_state_data.get(), Tensor::num_elements(lstm_struct.output_state()), 0);

  element_size = getDataTypeSize(Tensor::element_type(lstm_struct.cell_state()));
  auto cell_state_data = std::make_unique<int16_t[]>(Tensor::num_elements(lstm_struct.cell_state()));
  std::fill_n(cell_state_data.get(), element_size * Tensor::num_elements(lstm_struct.cell_state()), 0);

//
//
//
//  const bool use_layer_norm = (lstm_struct.forget_layer_norm_coefficients != nullptr);
//  const bool time_major = lstm_struct.options->time_major();
//  const auto output_state_zero_point = Tensor::zero_point(lstm_struct.output_state);
//
//  const int n_batch = time_major ? Tensor::dim(lstm_struct.input, 1) : Tensor::dim(lstm_struct.input, 0);
//  const int n_output = Tensor::dim(lstm_struct.recurrent_to_output_weights, 1);
//  const int n_cell = Tensor::dim(lstm_struct.input_to_output_weights, 0);
//  // Check the shape of input state tensors.
//  // These tensor may be 1D or 2D. It's fine as long as the total size is
//  // correct.
//  LUCI_INTERPRETER_CHECK(Tensor::num_elements(lstm_struct.output_state) == n_batch * n_output);
//  LUCI_INTERPRETER_CHECK(Tensor::num_elements(lstm_struct.cell_state) == n_batch * n_cell);
//
//  // Check the output tensors shape.
//  for (int i = 0; i < Tensor::num_dims(lstm_struct.input) - 1; i++)
//  {
//    LUCI_INTERPRETER_CHECK(Tensor::dim(lstm_struct.output, i) == Tensor::dim(lstm_struct.input, i));
//  }
//  LUCI_INTERPRETER_CHECK(Tensor::dim(lstm_struct.output, Tensor::num_dims(lstm_struct.input) - 1) == n_output);
//
//  luci_interpreter_pal::lstm::IntegerLSTMParams integer_lstm_params;
//
//  // Get intermediates
//  // TODO add more for use_layer_norm
//  const circle::Tensor *intermediate_tensor_0 = nullptr;
//  if (cur_op->intermediates() == nullptr)
//  {
//    intermediate_tensor_0 = runtime_graph->getIntermediateTensors()[0];
//  } else
//  {
//    const auto intermediate_index_0 = cur_op->intermediates()->operator[](0);
//    intermediate_tensor_0 = runtime_graph->getCircleTensorByIndex(intermediate_index_0);
//  }
//
//  assert(intermediate_tensor_0 != nullptr);
//
//  const bool use_cifg = (lstm_struct.input_to_input_weights == nullptr);
//  // Integer UnidirectionalSequenceLSTM prepare function for 8x8->16.
//  // This code path needs 5 intermediate tensors per Op.
//  // Populate quantization parameters.
//  populate_quantized_lstm_params(&lstm_struct, intermediate_tensor_0, &integer_lstm_params);
//
//  auto scratch_0_data = std::make_unique<int16_t[]>(n_batch * n_cell);//new uint8_t[n_batch * n_cell];
//  std::fill_n(scratch_0_data.get(), n_cell * n_batch, 0);
//
//  auto scratch_1_data = std::make_unique<int16_t[]>(n_batch * n_cell);//new uint8_t[n_batch * n_cell];
//  std::fill_n(scratch_1_data.get(), n_cell * n_batch, 0);
//
//  auto scratch_2_data = std::make_unique<int16_t[]>(n_batch * n_cell);//new uint8_t[n_batch * n_cell];
//  std::fill_n(scratch_2_data.get(), n_cell * n_batch, 0);
//
//  auto scratch_3_data = std::make_unique<int16_t[]>(n_batch * n_cell);//new uint8_t[n_batch * n_cell];
//  std::fill_n(scratch_3_data.get(), n_cell * n_batch, 0);
//
//  auto scratch_4_data = std::make_unique<int8_t[]>(n_batch * n_cell);//new uint8_t[n_batch * n_cell];
//  std::fill_n(scratch_4_data.get(), n_cell * n_batch, 0);
//
//  // Fill states tensors
//  auto element_size = getDataTypeSize(Tensor::element_type(lstm_struct.output_state));
//  auto output_state_data = std::make_unique<int8_t[]>(n_cell * n_batch);
//  std::fill_n(output_state_data.get(), n_cell * n_batch, 0);
//
//  element_size = getDataTypeSize(Tensor::element_type(lstm_struct.cell_state));
//  auto cell_state_data = std::make_unique<int16_t[]>(n_cell * n_batch);
//  std::fill_n(cell_state_data.get(), n_cell * n_batch, 0);
//
//  populate_precomputed_zp_times_weight_with_bias(&lstm_struct, intermediate_tensor_0, &integer_lstm_params,
//                                                       runtime_graph);

  luci_interpreter_pal::eval_integer_8x8_16_lstm(
    &lstm_struct, &quant_lstm_params,
    output_state_data.get(), cell_state_data.get(), kernels::getTensorData<int16_t>(scratch_0_data.get()),
    kernels::getTensorData<int16_t>(scratch_1_data.get()), kernels::getTensorData<int16_t>(scratch_2_data.get()),
    kernels::getTensorData<int16_t>(scratch_3_data.get()), runtime_graph);
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
