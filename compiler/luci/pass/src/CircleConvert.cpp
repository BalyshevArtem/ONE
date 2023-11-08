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

#include "luci/Pass/CircleConvert.h"
#include "CircleOptimizerUtils.h"

#include <luci/IR/CircleNodes.h>

namespace
{



} // namespace

namespace luci
{

/**
 * Constant Folding for Gather Op
 **/
bool CircleConvert::run(loco::Graph *g)
{
  bool changed = false;

  auto nodes = loco::active_nodes(loco::output_nodes(g));
  for (auto node : nodes)
  {
    auto first_mul = dynamic_cast<luci::CircleMul *>(node);
    if (not first_mul)
      continue;

    auto div = dynamic_cast<luci::CircleDiv *>(first_mul->y());
    if (not div)
      continue;

    auto reshape_after_div = dynamic_cast<luci::CircleReshape *>(div->y());
    if (not reshape_after_div)
      continue;

    auto mul = dynamic_cast<luci::CircleMul *>(reshape_after_div->tensor());
    if (not mul)
      continue;

    auto mean = dynamic_cast<luci::CircleMean *>(mul->x());
    if (not mean)
      continue;

    auto reshape_first = dynamic_cast<luci::CircleReshape *>(mean->input());
    if (not reshape_first)
      continue;

    auto input_node = reshape_first->tensor();

    mean->input(input_node);
    mean->shape({3, 197, 1});

    auto div_const = dynamic_cast<luci::CircleConst *>(div->x());
    auto div_value = div_const->at<loco::DataType::FLOAT32>(0);

    auto mul_const = dynamic_cast<luci::CircleConst *>(mul->y());
    auto mul_value = mul_const->at<loco::DataType::FLOAT32>(0);

    auto new_value = div_value / mul_value;

    //first_mul->y(div);
    div->y(mean);
    div_const->at<loco::DataType::FLOAT32>(0) = new_value;
    div->shape({3, 197, 197});


    first_mul->shape({3, 197, 197});
    g->outputs()->at(0)->shape({3, 197, 197});
    changed = true;
  }

  return changed;
}

} // namespace luci

