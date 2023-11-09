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

#include "luci/Pass/FuseReshapeMeanMulReshapeDivPatternPass.h"

#include <luci/IR/CircleNodes.h>

#include <luci/test/TestIOGraph.h>

#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

/**
 *  Simple graph for test
 *
 *  BEFORE
 *                [Input]
 *              (3, 197, 197)
 *                  |
 *              [Reshape]
 *             (1, 591, 197)
 *                  |
 *          [CircleMean, axis<-1>]
 *              (1, 591, 1)
 *                  |
 *          [Mul, Scalar_Const]
 *              (1, 591, 1)
 *                  |
 *              [Reshape]
 *             (1, 3, 197, 1)
 *                 |
 *           [Div, Scalar_Const]
 *             (1, 3, 197, 1)
 *                 |
 *               [Output]
 *            (1, 3, 197, 1)
 *
 *  AFTER
 *                [Input]
 *              (3, 197, 197)
 *                  |
 *          [CircleMean, axis<-1>]
 *              (3, 197, 1)
 *                  |
 *              [Reshape]
 *             (1, 3, 197, 1)
 *                 |
 *          [Div, Scalar_Const]
 *             (1, 3, 197, 1)
 *                 |
 *               [Output]
 *            (1, 3, 197, 1)
 *
 */
class PatternReshapeMeanMulReshapeDivGraphlet
{
public:
  PatternReshapeMeanMulReshapeDivGraphlet() = default;

  void init(loco::Graph *g)
  {
    _reshape_1 = g->nodes()->create<luci::CircleReshape>();
    _reshape_const_1 = g->nodes()->create<luci::CircleConst>();

    _mean = g->nodes()->create<luci::CircleMean>();
    _mean_const = g->nodes()->create<luci::CircleConst>();

    _mul = g->nodes()->create<luci::CircleMul>();
    _mul_const = g->nodes()->create<luci::CircleConst>();

    _reshape_2 = g->nodes()->create<luci::CircleReshape>();
    _reshape_const_2 = g->nodes()->create<luci::CircleConst>();

    _div = g->nodes()->create<luci::CircleDiv>();
    _div_const = g->nodes()->create<luci::CircleConst>();

    _reshape_1->name("_reshape_1");
    _reshape_const_1->name("_reshape_const_1");

    _mean->name("_mean");
    _mean_const->name("_mean_const");

    _mul->name("_mul");
    _mul_const->name("_mul_const");

    _reshape_2->name("_reshape_2");
    _reshape_const_2->name("_reshape_const_2");

    _div->name("_div");
    _div_const->name("_div_const");
  }

public:
  luci::CircleReshape *reshape_1() { return _reshape_1; }
  luci::CircleConst *reshape_const_1() { return _reshape_const_1; }
  luci::CircleReshape *reshape_2() { return _reshape_2; }
  luci::CircleConst *reshape_const_2() { return _reshape_const_2; }
  luci::CircleMean *mean() { return _mean; }
  luci::CircleConst *mean_const() { return _mean_const; }
  luci::CircleMul *mul() { return _mul; }
  luci::CircleConst *mul_const() { return _mul_const; }
  luci::CircleDiv *div() { return _div; }
  luci::CircleConst *div_const() { return _div_const; }

protected:
  luci::CircleReshape *_reshape_1 = nullptr;
  luci::CircleConst *_reshape_const_1 = nullptr;
  luci::CircleMean *_mean = nullptr;
  luci::CircleConst *_mean_const = nullptr;
  luci::CircleMul *_mul = nullptr;
  luci::CircleConst *_mul_const = nullptr;
  luci::CircleReshape *_reshape_2 = nullptr;
  luci::CircleConst *_reshape_const_2 = nullptr;
  luci::CircleDiv *_div = nullptr;
  luci::CircleConst *_div_const = nullptr;
};

class FuseReshapeMeanMulReshapeDivPatternTestGraph : public TestIOGraph, public PatternReshapeMeanMulReshapeDivGraphlet
  {
  public:
    FuseReshapeMeanMulReshapeDivPatternTestGraph() = default;

    void init(void)
    {
      TestIOGraph::init({3, 197, 197}, {1, 3, 197, 1});
      PatternReshapeMeanMulReshapeDivGraphlet::init(g());

      _indices1->rank(1);
      _indices1->dtype(loco::DataType::S32);
      _indices1->size<loco::DataType::S32>(1);
      _indices1->at<loco::DataType::S32>(0) = static_cast<int32_t>(1);
      _indices1->shape_status(luci::ShapeStatus::VALID);

      _indices2->rank(1);
      _indices2->dtype(loco::DataType::S32);
      _indices2->size<loco::DataType::S32>(1);
      _indices2->at<loco::DataType::S32>(0) = static_cast<int32_t>(2);
      _indices2->shape_status(luci::ShapeStatus::VALID);

      _mean1->input(input());
      _mean1->reduction_indices(_indices1);

      _mean2->input(_mean1);
      _mean2->reduction_indices(_indices2);

      output()->from(_mean2);
    }
  };

} // namespace

TEST(FuseMeanWithMeanPassTest, name)
{
  luci::FuseMeanWithMeanPass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST(FuseMeanWithMeanPassTest, fuse_mean_with_mean)
{
  FuseMeanWithMeanTestGraph g;
  luci::FuseMeanWithMeanPass pass;

  g.init();

  EXPECT_TRUE(pass.run(g.g()));
}

TEST(FuseMeanWithMeanPassTest, fus_mean_with_mean_NEG)
{
  FuseMeanWithMeanTestGraph g;
  luci::FuseMeanWithMeanPass pass;

  g.init();

  // Add CircleRelu operation between CircleMeans operations
  auto relu = g.g()->nodes()->create<luci::CircleRelu>();
  relu->name("relu");
  relu->features(g.mean1());
  g.mean2()->input(relu);

  // Due to the CircleRelu operation, pass will not be applied
  EXPECT_FALSE(pass.run(g.g()));
}
