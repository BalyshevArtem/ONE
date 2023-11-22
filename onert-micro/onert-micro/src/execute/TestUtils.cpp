/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "execute/TestUtils.h"

#include "OMInterpreter.h"

namespace onert_micro
{
namespace execute
{
namespace testing
{

using ::testing::FloatNear;
using ::testing::Matcher;

Matcher<std::vector<float>> FloatArrayNear(const std::vector<float> &values, float max_abs_error)
{
  std::vector<Matcher<float>> matchers;
  matchers.reserve(values.size());
  for (const float v : values)
  {
    matchers.emplace_back(FloatNear(v, max_abs_error));
  }
  return ElementsAreArray(matchers);
}

template <typename T>
std::vector<T> onert_micro::execute::testing::checkSISOKernel(onert_micro::test_model::TestDataBase<T> *test_data_base)
{

  onert_micro::OMInterpreter interpreter;
  onert_micro::OMConfig config;

  interpreter.importModel(test_data_base->get_model_ptr(), config);

  T *input_data = reinterpret_cast<T *>(interpreter.getInputDataAt(0));

  // Set input data
  {
    std::copy(test_data_base->get_input_data_by_index(0).begin(),
              test_data_base->get_input_data_by_index(0).end(), input_data);
  }

  interpreter.run();

  T *output_data = reinterpret_cast<T *>(interpreter.getOutputDataAt(0));
  const size_t num_elements = interpreter.getOutputSizeAt(0);
  std::vector<T> output_data_vector(output_data, output_data + num_elements);
  return output_data_vector;
}

void checkNEGSISOKernel(onert_micro::test_model::NegTestDataBase *test_data_base)
{
  onert_micro::OMInterpreter interpreter;
  onert_micro::OMConfig config;

  interpreter.importModel(reinterpret_cast<const char *>(test_data_base->get_model_ptr()), config);
}

} // namespace testing
} // namespace execute
} // namespace onert_micro
