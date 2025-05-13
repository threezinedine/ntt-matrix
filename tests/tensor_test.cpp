#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdio>
#define NTT_MICRO_NN_IMPLEMENTATION
#include <ntt_very_super_micro_dnn/ntt_tensor.hpp>

using namespace ntt;

TEST(TensorTest, ConstructorWithShapeOnly)
{
    Tensor tensor({2, 3});
    EXPECT_EQ(tensor.get_shape(), (shape_type{2, 3}));

    Tensor tensor1D({3});
    EXPECT_EQ(tensor1D.get_shape(), (shape_type{3}));
}

TEST(TensorTest, GetElementByIndex)
{
    Tensor tensor({2, 3});

    EXPECT_EQ(tensor.get_element({0, 0}), 0);
    EXPECT_EQ(tensor.get_element({0, 1}), 0);
    EXPECT_EQ(tensor.get_element({0, 2}), 0);
    EXPECT_EQ(tensor.get_element({1, 0}), 0);
    EXPECT_EQ(tensor.get_element({1, 1}), 0);
    EXPECT_EQ(tensor.get_element({1, 2}), 0);
}

TEST(TensorTest, ConstructorWithShapeAndDefaultValues)
{
    Tensor tensor({2, 3}, 1);

    EXPECT_EQ(tensor.get_element({0, 0}), 1);
    EXPECT_EQ(tensor.get_element({0, 1}), 1);
    EXPECT_EQ(tensor.get_element({0, 2}), 1);
    EXPECT_EQ(tensor.get_element({1, 0}), 1);
    EXPECT_EQ(tensor.get_element({1, 1}), 1);
    EXPECT_EQ(tensor.get_element({1, 2}), 1);
}

TEST(ShapeTest, ConstructorWithShape)
{
    Shape shape({2, 3});
    EXPECT_FALSE(shape.is_end());
}

TEST(ShapeTest, TestNextToReachEnd)
{
    Shape shape({2, 3});

    EXPECT_FALSE(shape.is_end()); // {0, 0}
    shape.next();                 // {0, 1}

    EXPECT_FALSE(shape.is_end());
    shape.next(); // {0, 2}

    EXPECT_FALSE(shape.is_end());
    shape.next(); // {1, 0}

    EXPECT_FALSE(shape.is_end());
    shape.next(); // {1, 1}

    EXPECT_FALSE(shape.is_end());
    shape.next(); // {1, 2}

    EXPECT_FALSE(shape.is_end());
    shape.next(); // {2, 0}

    EXPECT_TRUE(shape.is_end());

    shape.next(); // {2, 0}
    EXPECT_TRUE(shape.is_end());

    shape.next(); // {2, 0}
    EXPECT_TRUE(shape.is_end());
}

TEST(ShapeTest, TestResetShape)
{
    Shape shape({2, 2});

    shape.next();
    shape.next();

    shape.reset();
    shape.next();
    shape.next();
    EXPECT_FALSE(shape.is_end());
    shape.next();
    shape.next();
    EXPECT_TRUE(shape.is_end());
}

TEST(ShapeTest, TestNextFor3DShape)
{
    Shape shape({2, 2, 2});

    EXPECT_FALSE(shape.is_end());
    shape.next();
    shape.next();
    shape.next();
    shape.next();
    shape.next();
    shape.next();
    shape.next();
    shape.next();
    EXPECT_TRUE(shape.is_end());
    shape.next();
    shape.next();
    shape.next();
}

TEST(ShapeTest, TestShapeCurrentIndex)
{
    Shape shape({2, 2, 2, 3});

    shape.next();
    EXPECT_THAT(shape.get_current_index(), ::testing::ElementsAre(0, 0, 0, 1));

    shape.next();
    EXPECT_THAT(shape.get_current_index(), ::testing::ElementsAre(0, 0, 0, 2));

    shape.next();
    EXPECT_THAT(shape.get_current_index(), ::testing::ElementsAre(0, 0, 1, 0));
}

TEST(ShapeTest, TestIsFirstElement)
{
    Shape shape({2, 2, 2, 3});

    EXPECT_TRUE(shape.is_first_element());

    shape.next();
    EXPECT_FALSE(shape.is_first_element());

    shape.next();
    shape.next();
    shape.next();
    EXPECT_FALSE(shape.is_first_element());

    shape.reset();
    EXPECT_TRUE(shape.is_first_element());
}

TEST(TensorTest, SetElementByIndex)
{
    Tensor tensor({2, 2, 2, 3}, 1);

    tensor.set_element({0, 0, 0, 0}, 4.23);
    EXPECT_THAT(tensor.get_element({0, 0, 0, 0}), ::testing::FloatEq(4.23));
}

TEST(TensorTest, ConstructorWithPredefinedData)
{
    Tensor tensor = Tensor::from_vector({3.0, -2.1, 1.0});
    EXPECT_EQ(tensor.get_shape(), (shape_type{3}));

    EXPECT_THAT(tensor.get_element({0}), ::testing::FloatEq(3.0));
    EXPECT_THAT(tensor.get_element({1}), ::testing::FloatEq(-2.1));
    EXPECT_THAT(tensor.get_element({2}), ::testing::FloatEq(1.0));
}

TEST(TensorTest, ConstructorWithPredefinedData_2D)
{
    Tensor tensor = Tensor::from_vector({{3.0, -2.1, 1.0},
                                         {4.2, 5.0, -1.0}});
    EXPECT_EQ(tensor.get_shape(), (shape_type{2, 3}));

    EXPECT_THAT(tensor.get_element({0, 0}), ::testing::FloatEq(3.0));
    EXPECT_THAT(tensor.get_element({0, 1}), ::testing::FloatEq(-2.1));
    EXPECT_THAT(tensor.get_element({0, 2}), ::testing::FloatEq(1.0));
    EXPECT_THAT(tensor.get_element({1, 0}), ::testing::FloatEq(4.2));
    EXPECT_THAT(tensor.get_element({1, 1}), ::testing::FloatEq(5.0));
    EXPECT_THAT(tensor.get_element({1, 2}), ::testing::FloatEq(-1.0));
}

TEST(TensorTest, ConstructorWithPredefinedData_3D)
{
    Tensor tensor = Tensor::from_vector({{{4.23, 1.0f},
                                          {1.0, 1.0f}},
                                         {{-2.16f, 1.0f},
                                          {2.0, -1.0f}}});
    EXPECT_EQ(tensor.get_shape(), (shape_type{2, 2, 2}));

    EXPECT_THAT(tensor.get_element({0, 0, 0}), ::testing::FloatEq(4.23));
    EXPECT_THAT(tensor.get_element({0, 0, 1}), ::testing::FloatEq(1.0));
    EXPECT_THAT(tensor.get_element({0, 1, 0}), ::testing::FloatEq(1.0));
    EXPECT_THAT(tensor.get_element({0, 1, 1}), ::testing::FloatEq(1.0));
    EXPECT_THAT(tensor.get_element({1, 0, 0}), ::testing::FloatEq(-2.16f));
    EXPECT_THAT(tensor.get_element({1, 0, 1}), ::testing::FloatEq(1.0));
    EXPECT_THAT(tensor.get_element({1, 1, 0}), ::testing::FloatEq(2.0));
    EXPECT_THAT(tensor.get_element({1, 1, 1}), ::testing::FloatEq(-1.0));
}

TEST(TensorTest, ConstructorWithPredefinedData_4D)
{
    Tensor tensor = Tensor::from_vector({{{{4.23, 1.0f},
                                           {1.0, 1.0f}},
                                          {{-2.16f, 1.0f},
                                           {2.0, -1.0f}}},
                                         {{{4.23, 1.0f},
                                           {1.0, 1.0f}},
                                          {{-2.16f, 1.0f},
                                           {2.0, -1.0f}}}});
    EXPECT_EQ(tensor.get_shape(), (shape_type{2, 2, 2, 2}));

    EXPECT_THAT(tensor.get_element({0, 0, 0, 0}), ::testing::FloatEq(4.23));
    EXPECT_THAT(tensor.get_element({0, 0, 0, 1}), ::testing::FloatEq(1.0));
    EXPECT_THAT(tensor.get_element({0, 0, 1, 0}), ::testing::FloatEq(1.0));
    EXPECT_THAT(tensor.get_element({0, 0, 1, 1}), ::testing::FloatEq(1.0));
    EXPECT_THAT(tensor.get_element({0, 1, 0, 0}), ::testing::FloatEq(-2.16f));
    EXPECT_THAT(tensor.get_element({0, 1, 0, 1}), ::testing::FloatEq(1.0));
    EXPECT_THAT(tensor.get_element({0, 1, 1, 0}), ::testing::FloatEq(2.0));
    EXPECT_THAT(tensor.get_element({0, 1, 1, 1}), ::testing::FloatEq(-1.0));
    EXPECT_THAT(tensor.get_element({1, 0, 0, 0}), ::testing::FloatEq(4.23));
    EXPECT_THAT(tensor.get_element({1, 0, 0, 1}), ::testing::FloatEq(1.0));
    EXPECT_THAT(tensor.get_element({1, 0, 1, 0}), ::testing::FloatEq(1.0));
    EXPECT_THAT(tensor.get_element({1, 0, 1, 1}), ::testing::FloatEq(1.0));
    EXPECT_THAT(tensor.get_element({1, 1, 0, 0}), ::testing::FloatEq(-2.16f));
    EXPECT_THAT(tensor.get_element({1, 1, 0, 1}), ::testing::FloatEq(1.0));
    EXPECT_THAT(tensor.get_element({1, 1, 1, 0}), ::testing::FloatEq(2.0));
    EXPECT_THAT(tensor.get_element({1, 1, 1, 1}), ::testing::FloatEq(-1.0));
}

TEST(TensorTest, ReallocateTensorWhenANewOneIsAssigned)
{
    Tensor tensor({2, 2}, 1);
    tensor = Tensor::from_vector({3.0, -2.1, 1.0});
    EXPECT_EQ(tensor.get_shape(), (shape_type{3}));
}

TEST(TensorTest, GetElementWithExceedingIndex)
{
    Tensor tensor({2, 2}, 1);
    EXPECT_THROW(tensor.get_element({2, 2}), std::out_of_range);
    EXPECT_THROW(tensor.get_element({2, 0}), std::out_of_range);
    EXPECT_THROW(tensor.get_element({0, 2}), std::out_of_range);
}

TEST(TensorTest, SetElementWithExceedingIndex)
{
    Tensor tensor({2, 2}, 1);
    EXPECT_THROW(tensor.set_element({2, 2}, 1), std::out_of_range);
    EXPECT_THROW(tensor.set_element({2, 0}, 1), std::out_of_range);
    EXPECT_THROW(tensor.set_element({0, 2}, 1), std::out_of_range);
}

TEST(TensorTest, ReshapeFailedBecauseOfTotalElementsMismatch)
{
    Tensor tensor({2, 2}, 1);
    EXPECT_THROW(tensor.reshape({3, 3}), std::invalid_argument);
}

TEST(TensorTest, ReshapeTensor)
{
    Tensor tensor = Tensor::from_vector({2.0, 3.2, 2.1});

    tensor.reshape({1, 3});
    EXPECT_EQ(tensor.get_shape(), (shape_type{1, 3}));
    EXPECT_THAT(tensor.get_element({0, 0}), ::testing::FloatEq(2.0));
    EXPECT_THAT(tensor.get_element({0, 1}), ::testing::FloatEq(3.2));
    EXPECT_THAT(tensor.get_element({0, 2}), ::testing::FloatEq(2.1));
}

TEST(TensorTest, CopyConstructor)
{
    Tensor tensor = Tensor::from_vector({2.0, 3.2, 2.1});
    Tensor tensor2 = tensor;
    EXPECT_EQ(tensor2.get_shape(), (shape_type{3}));
    EXPECT_THAT(tensor2.get_element({0}), ::testing::FloatEq(2.0));
    EXPECT_THAT(tensor2.get_element({1}), ::testing::FloatEq(3.2));
    EXPECT_THAT(tensor2.get_element({2}), ::testing::FloatEq(2.1));
}

TEST(TensorTest, ToShapeClone)
{
    Tensor tensor = Tensor::from_vector({2.0, 3.2, 2.1});
    Tensor tensor2 = tensor.reshape_clone({1, 3});
    EXPECT_EQ(tensor2.get_shape(), (shape_type{1, 3}));
    EXPECT_THAT(tensor2.get_element({0, 0}), ::testing::FloatEq(2.0));
    EXPECT_THAT(tensor2.get_element({0, 1}), ::testing::FloatEq(3.2));
    EXPECT_THAT(tensor2.get_element({0, 2}), ::testing::FloatEq(2.1));

    EXPECT_THAT(tensor.get_element({0}), ::testing::FloatEq(2.0));
    EXPECT_THAT(tensor.get_element({1}), ::testing::FloatEq(3.2));
    EXPECT_THAT(tensor.get_element({2}), ::testing::FloatEq(2.1));
}

TEST(TensorTest, CopyAssignmentOperator)
{
    Tensor tensor1 = Tensor::from_vector({1.0, 2.0, 3.0});
    Tensor tensor2 = Tensor::from_vector({1.0, 2.0, 3.0});

    EXPECT_EQ(tensor1, tensor2);
}

TEST(TensorTest, GetElementByIndex_WithWrongShape)
{
    Tensor tensor = Tensor::from_vector({1.0, 2.0, 3.0});
    EXPECT_THROW(tensor.get_element({0, 0}), std::invalid_argument);
}

TEST(TensorTest, Addition)
{
    Tensor tensor1 = Tensor::from_vector({1.0, 2.0, 3.0});
    Tensor tensor2 = Tensor::from_vector({4.0, 5.0, 6.0});
    Tensor result = tensor1 + tensor2;

    EXPECT_EQ(result, Tensor::from_vector({5.0, 7.0, 9.0}));
}

TEST(TensorTest, AddScalar)
{
    Tensor tensor = Tensor::from_vector({1.0, 2.0, 3.0});
    Tensor result = tensor + 1.0;

    EXPECT_EQ(result, Tensor::from_vector({2.0, 3.0, 4.0}));
}

TEST(TensorTest, AdditionWithDifferentShape)
{
    Tensor tensor1 = Tensor::from_vector({1.0, 2.0, 3.0});
    Tensor tensor2 = Tensor::from_vector({4.0, 5.0, 6.0, 7.0});
    EXPECT_THROW(tensor1 + tensor2, std::invalid_argument);
}

TEST(TensorTest, Subtraction)
{
    Tensor tensor1 = Tensor::from_vector({1.0, 2.0, 3.0});
    Tensor tensor2 = Tensor::from_vector({4.0, 5.0, 6.0});
    Tensor result = tensor1 - tensor2;

    EXPECT_EQ(result, Tensor::from_vector({-3.0, -3.0, -3.0}));
}

TEST(TensorTest, SubtractionWithScalar)
{
    Tensor tensor = Tensor::from_vector({1.0, 2.0, 3.0});
    Tensor result = tensor - 1.0;

    EXPECT_EQ(result, Tensor::from_vector({0.0, 1.0, 2.0}));
}

TEST(TensorTest, SubtractionWithDifferentShape)
{
    Tensor tensor1 = Tensor::from_vector({1.0, 2.0, 3.0});
    Tensor tensor2 = Tensor::from_vector({4.0, 5.0, 6.0, 7.0});
    EXPECT_THROW(tensor1 - tensor2, std::invalid_argument);
}

TEST(TensorTest, Negative)
{
    Tensor tensor = Tensor::from_vector({1.0, 2.0, 3.0});
    Tensor result = tensor.negative();

    EXPECT_EQ(result, Tensor::from_vector({-1.0, -2.0, -3.0}));
}

TEST(TensorTest, Multiplication)
{
    Tensor tensor1 = Tensor::from_vector({1.0, 2.0, 3.0});
    Tensor result = tensor1 * 2.0;

    EXPECT_EQ(result, Tensor::from_vector({2.0, 4.0, 6.0}));
}

TEST(TensorTest, Division)
{
    Tensor tensor1 = Tensor::from_vector({1.0, 2.0, 3.0});
    Tensor result = tensor1 / 2.0;

    EXPECT_EQ(result, Tensor::from_vector({0.5, 1.0, 1.5}));
}

TEST(TensorTest, Transpose)
{
    Tensor tensor = Tensor::from_vector({{1.0, 2.0, 3.0},
                                         {4.0, 5.0, 6.0}});
    Tensor result = tensor.transpose(0, 1);

    EXPECT_EQ(result, Tensor::from_vector({{1.0, 4.0},
                                           {2.0, 5.0},
                                           {3.0, 6.0}}));
}

TEST(TensorTest, TransposeFailedBecauseOfInvalidAxis)
{
    Tensor tensor = Tensor::from_vector({{1.0, 2.0, 3.0},
                                         {4.0, 5.0, 6.0}});
    EXPECT_THROW(tensor.transpose(0, 2), std::invalid_argument);
}

TEST(TensorTest, FindMax)
{
    Tensor tensor = Tensor::from_vector({{1.0, 5.0, 3.0},
                                         {4.0, 2.0, 6.0}});
    EXPECT_THAT(tensor.max(), ::testing::FloatEq(6.0));
    EXPECT_EQ(tensor.max(0), Tensor::from_vector(tensor2d{{4.0f, 5.0f, 6.0f}}));
    EXPECT_EQ(tensor.max(1), Tensor::from_vector(tensor2d{{5.0f}, {6.0f}}));
}

TEST(TensorTest, Argmax)
{
    Tensor tensor = Tensor::from_vector({{1.0, 5.0, 6.0},
                                         {4.0, 9.0, 3.0}});
    EXPECT_EQ(tensor.argmax(), (shape_type{1, 1}));
    EXPECT_EQ(tensor.argmax(0), Tensor::from_vector(tensor2d{{1, 1, 0}}));
    EXPECT_EQ(tensor.argmax(1), Tensor::from_vector(tensor2d{{2}, {1}}));
}

TEST(NeuralNetTest, TestReLULayer)
{
    Tensor input = Tensor::from_vector({-1.0, 2.0, -3.0});
    EXPECT_EQ(ReLULayer().forward(input), Tensor::from_vector({0.0, 2.0, 0.0}));
}

TEST(NeuralNetTest, TestReLULayer_With3DInput)
{
    Tensor input = Tensor::from_vector(tensor3d{{{1.0, -2.0, -3.0},
                                                 {-4.0, 5.0, 6.0}}});
    EXPECT_EQ(ReLULayer().forward(input), Tensor::from_vector(tensor3d{{{1.0, 0.0, 0.0}, {0.0, 5.0, 6.0}}}));
}

TEST(NeuralNetTest, TestClip2D)
{
    Tensor input = Tensor::from_vector(tensor2d{{1.0, 2.0, 3.0},
                                                {0.3, -5.0, -6.0}});
    EXPECT_EQ(Clip2DLayer(-1, 1).forward(input), Tensor::from_vector(tensor2d{{1.0, 1.0, 1.0},
                                                                              {0.3, -1.0, -1.0}}));
}

TEST(NeuralNetTest, TestFullyConnectedLayer)
{
    Tensor input = Tensor::from_vector(tensor2d{{1.0},
                                                {0.3},
                                                {-0.4}}); // [3 x 1]

    Tensor weights = Tensor::from_vector(tensor2d{
        {0.1, 0.2, 0.3},
        {0.4, 0.5, 0.6}}); // [2x3]

    Tensor bias = Tensor::from_vector(tensor2d{{0.1},
                                               {0.3}}); // [2 x 1]

    Tensor result = FullyConnectedLayer(weights, bias).forward(input.reshape_clone({3, 1}));
    EXPECT_EQ(result, Tensor::from_vector(tensor2d{{0.14},
                                                   {0.61}})); // [2 x 1]
}

TEST(NeuralNetTest, TestSoftmaxLayer)
{
    Tensor input = Tensor::from_vector(tensor2d{{1.0, 2.0, 3.0}});
    EXPECT_EQ(SoftmaxLayer().forward(input),
              Tensor::from_vector(tensor2d{{0.09003057317038046, 0.24472847105479767, 0.6652409557748219}}));
}

TEST(NeuralNetTest, TestSigmoidLayer)
{
    Tensor input = Tensor::from_vector(tensor2d{{1.0, -2.0, 3.0}});
    EXPECT_EQ(SigmoidLayer().forward(input),
              Tensor::from_vector(tensor2d{{0.7310585786300049, 0.11920292202211756, 0.9525741268224334}}));
}

TEST(NeuralNetTest, TestConv2DLayer)
{
    // shape: [3, 1, 4, 4]
    Tensor input = Tensor::from_vector(tensor4d{
        {{{1.0, 2.0, 3.0, 4.0},
          {6.0, 7.0, 8.0, 9.0},
          {16.0, 17.0, 18.0, 19.0},
          {21.0, 22.0, 23.0, 24.0}}},
        {{{1.0, 2.0, 3.0, 4.0},
          {6.0, 7.0, 8.0, 9.0},
          {16.0, 17.0, 18.0, 19.0},
          {21.0, 22.0, 23.0, 24.0}}},
        {{{1.0, 2.0, 3.0, 4.0},
          {6.0, 7.0, 8.0, 9.0},
          {16.0, 17.0, 18.0, 19.0},
          {21.0, 22.0, 23.0, 24.0}}},
    });

    EXPECT_THAT(input.get_shape(), ::testing::ElementsAre(3, 1, 4, 4));

    // shape: [2, 3, 2, 2]
    Tensor weights = Tensor::from_vector(tensor4d{
        {{{1.0, 2.0},
          {7.0, 8.0}},
         {{1.0, 2.0},
          {7.0, 8.0}},
         {{-0.4, -2.3},
          {0.1, 0.2}}},
        {{{1.0, 2.0},
          {7.0, 8.0}},
         {{1.0, 2.0},
          {7.0, 8.0}},
         {{-0.4, -2.3},
          {0.1, 0.2}}}});

    EXPECT_THAT(weights.get_shape(), ::testing::ElementsAre(2, 3, 2, 2));

    // shape: [2, 1]
    Tensor bias = Tensor::from_vector(tensor2d{{0.1},
                                               {0.2}});

    Tensor result = Conv2DLayer(weights, bias, 1, false).forward(input);
}

TEST(NeuralNetTest, TestCrossCorrelation_InConv2DLayer)
{
    Tensor input = Tensor::from_vector(tensor4d{
        {{{1.0, 2.0, 3.0, 4.0},
          {6.0, 7.0, 8.0, 9.0},
          {16.0, 17.0, 18.0, 19.0},
          {21.0, 22.0, 23.0, 24.0}}},
    });

    EXPECT_THAT(input.get_shape(), ::testing::ElementsAre(1, 1, 4, 4));

    Tensor weights = Tensor::from_vector(tensor4d{
        {
            {{1.0, 1.0},
             {1.0, 1.0}},
        }});

    EXPECT_THAT(weights.get_shape(), ::testing::ElementsAre(1, 1, 2, 2));

    Tensor bias = Tensor::from_vector(tensor2d{{0.0}});

    EXPECT_EQ(Conv2DLayer(weights, bias, 2).forward(input),
              Tensor::from_vector(tensor4d{
                  {
                      {{16.0, 24.0},
                       {76.0, 84.0}},
                  }}));

    EXPECT_EQ(Conv2DLayer(weights, bias).forward(input),
              Tensor::from_vector(tensor4d{
                  {
                      {{16.0, 20.0, 24.0},
                       {46.0, 50.0, 54.0},
                       {76.0, 80.0, 84.0}},
                  }}));
}

TEST(NeuralNetTest, TestConv2DLayer_WithKernalSize1)
{
    Tensor input = Tensor::from_vector(tensor4d{
        {{{1.0, 2.0},
          {6.0, 7.0}}}});

    EXPECT_THAT(input.get_shape(), ::testing::ElementsAre(1, 1, 2, 2));

    Tensor weights({1, 1, 1, 1}, 1.0f);
    Tensor bias = Tensor::from_vector(tensor2d{{0.0}});

    EXPECT_EQ(Conv2DLayer(weights, bias).forward(input),
              Tensor::from_vector(tensor4d{
                  {
                      {{1.0, 2.0},
                       {6.0, 7.0}},
                  }}));
}

TEST(NeuralNetTest, TestDotProduct_InConv2DLayer)
{
    // input shape: [3, 1, 2, 2]
    Tensor input = Tensor::from_vector(tensor4d{
        {{{1.2, 0.3},
          {0.3, -0.8}}},
        {{{0.23, -0.12},
          {-1.2, 3.12}}},
        {{{0.23, 0.45},
          {-1.08, -1.2}}}});

    EXPECT_THAT(input.get_shape(), ::testing::ElementsAre(3, 1, 2, 2));

    Tensor weights({2, 3, 1, 1}, 1.0f);
    Tensor bias({2, 1}, 0.5f);

    // output shape: [2, 1, 2, 2]
    EXPECT_EQ(Conv2DLayer(weights, bias).forward(input),
              Tensor::from_vector(tensor4d{
                  {
                      {{2.16, 1.13},
                       {-1.48, 1.62}},
                  },
                  {
                      {{2.16, 1.13},
                       {-1.48, 1.62}},
                  }}));
}

TEST(NeuralNetTest, TestConv2DLayer_WithSamePadding)
{
    Tensor input = Tensor::from_vector(tensor4d{
        {{{1.0, 2.0, 3.0, 4.0},
          {6.0, 7.0, 8.0, 9.0},
          {16.0, 17.0, 18.0, 19.0},
          {21.0, 22.0, 23.0, 24.0}}},
    });

    EXPECT_THAT(input.get_shape(), ::testing::ElementsAre(1, 1, 4, 4));

    Tensor weights = Tensor::from_vector(tensor4d{
        {
            {{1.0, 1.0},
             {1.0, 1.0}},
        }});

    EXPECT_THAT(weights.get_shape(), ::testing::ElementsAre(1, 1, 2, 2));

    Tensor bias = Tensor::from_vector(tensor2d{{0.0}});

    EXPECT_EQ(Conv2DLayer(weights, bias, 1, 1).forward(input),
              Tensor::from_vector(tensor4d{
                  {
                      {{1.0, 3.0, 5.0, 7.0, 4.0},
                       {7.0, 16.0, 20.0, 24.0, 13.0},
                       {22.0, 46.0, 50.0, 54.0, 28.0},
                       {37.0, 76.0, 80.0, 84.0, 43.0},
                       {21.0, 43.0, 45.0, 47.0, 24.0}},
                  }}));
}

TEST(NeuralNetTest, TestFlattenLayer)
{
    Tensor input = Tensor::from_vector(tensor4d{
        {{
            {1.0, 2.0, 3.0, 4.0},
            {5.0, 6.0, 7.0, 8.0},
        }}});

    EXPECT_THAT(input.get_shape(), ::testing::ElementsAre(1, 1, 2, 4));

    EXPECT_EQ(FlattenLayer().forward(input),
              Tensor::from_vector(tensor2d{{1.0},
                                           {2.0},
                                           {3.0},
                                           {4.0},
                                           {5.0},
                                           {6.0},
                                           {7.0},
                                           {8.0}}));
}

TEST(NeuralNetTest, TestMaxPooling2DLayer)
{
    // shape [2, 1, 4, 4]
    Tensor input = Tensor::from_vector(tensor4d{
        {{
            {1.0, 2.0, 3.0, 4.0},
            {5.0, 6.0, 7.0, 8.0},
            {9.0, 10.0, 11.0, 12.0},
            {13.0, 14.0, 15.0, 16.0},
        }},
        {{
            {17.0, 18.0, 19.0, 20.0},
            {21.0, 22.0, 23.0, 24.0},
            {25.0, 26.0, 27.0, 28.0},
            {29.0, 30.0, 31.0, 32.0},
        }}});

    EXPECT_THAT(input.get_shape(), ::testing::ElementsAre(2, 1, 4, 4));

    EXPECT_EQ(MaxPooling2DLayer(2, 2).forward(input),
              Tensor::from_vector(tensor4d{
                  {
                      {{6.0, 8.0},
                       {14.0, 16.0}},
                  },
                  {
                      {{22.0, 24.0},
                       {30.0, 32.0}},
                  }}));
}

TEST(NeuralNetTest, TestMaxPoolingWithGroup)
{
    Tensor input = Tensor::from_vector(tensor4d{
        {{
            {{1.0, 2.0, 3.0, 4.0},
             {5.0, 6.0, 7.0, 8.0},
             {9.0, 10.0, 11.0, 12.0},
             {13.0, 14.0, 15.0, 16.0}},
        }},
        {{
            {{1.0, 2.0, 3.0, 4.0},
             {9.0, 10.0, 11.0, 12.0},
             {5.0, 6.0, 7.0, 8.0},
             {13.0, 14.0, 15.0, 16.0}},
        }},
        {{
            {{1.0, 2.0, 3.0, 4.0},
             {5.0, 6.0, 7.0, 8.0},
             {9.0, 10.0, 11.0, 12.0},
             {13.0, 14.0, 15.0, 16.0}},
        }},
    });
    EXPECT_THAT(input.get_shape(), ::testing::ElementsAre(3, 1, 4, 4));

    Tensor weights = Tensor::from_vector(tensor4d{
        {{
            {{0.5, 0.5},
             {0.5, 0.5}},
        }},
        {{
            {{0.5, 0.5},
             {0.5, 0.5}},
        }},
        {{
            {{0.5, 0.5},
             {0.5, 0.5}},
        }},
    });
    EXPECT_THAT(weights.get_shape(), ::testing::ElementsAre(3, 1, 2, 2));

    Tensor bias({3, 1}, 1.0f);

    EXPECT_EQ(Conv2DLayer(weights, bias, 1, 0, 3).forward(input),
              Tensor::from_vector(tensor4d{
                  {{{
                      {8.0, 10.0, 12.0},
                      {16.0, 18.0, 20.0},
                      {24.0, 26.0, 28.0},
                  }}},
                  {{
                      {{12.0, 14.0, 16.0},
                       {16.0, 18.0, 20.0},
                       {20.0, 22.0, 24.0}},
                  }},
                  {{
                      {{8.0, 10.0, 12.0},
                       {16.0, 18.0, 20.0},
                       {24.0, 26.0, 28.0}},
                  }},
              }));
}

TEST(NeuralNetTest, TestMaxPooling2DLayer_WithStride)
{
    Tensor input = Tensor::from_vector(tensor4d{
        {{
            {1.0, 2.0, 3.0, 4.0},
            {5.0, 6.0, 7.0, 8.0},
            {9.0, 10.0, 11.0, 12.0},
            {13.0, 14.0, 15.0, 16.0},
        }},
    });

    EXPECT_THAT(input.get_shape(), ::testing::ElementsAre(1, 1, 4, 4));

    EXPECT_EQ(MaxPooling2DLayer(2, 1).forward(input),
              Tensor::from_vector(tensor4d{
                  {
                      {{6.0, 7.0, 8.0},
                       {10.0, 11.0, 12.0},
                       {14.0, 15.0, 16.0}},
                  }}));
}

TEST(NeuralNetTest, TestMaxPooling2DLayer_WithPadding)
{
    Tensor input = Tensor::from_vector(tensor4d{
        {{
            {1.0, 2.0, 3.0, 4.0},
            {5.0, 6.0, 7.0, 8.0},
            {9.0, 10.0, 11.0, 12.0},
            {13.0, 14.0, 15.0, 16.0},
        }},
    });

    EXPECT_THAT(input.get_shape(), ::testing::ElementsAre(1, 1, 4, 4));

    EXPECT_EQ(MaxPooling2DLayer(2, 1, 1).forward(input),
              Tensor::from_vector(tensor4d{
                  {
                      {{1.0, 2.0, 3.0, 4.0, 4.0},
                       {5.0, 6.0, 7.0, 8.0, 8.0},
                       {9.0, 10.0, 11.0, 12.0, 12.0},
                       {13.0, 14.0, 15.0, 16.0, 16.0},
                       {13.0, 14.0, 15.0, 16.0, 16.0}},
                  }}));
}

TEST(NeuralNetTest, TestGlobalAveragePooling2DLayer)
{
    Tensor input = Tensor::from_vector(tensor4d{
        {{{{3.2, 2.1, 1.0},
           {-0.3, 0.2, 0.1},
           {0.3, 0.2, 0.1}}}},
        {{{{1.0, 1.0, 1.0},
           {1.0, 1.0, 1.0},
           {1.0, 1.0, 1.0}}}},
        {{{{1.0, 0.0, 0.0},
           {0.0, 1.0, 0.0},
           {0.0, 0.0, 1.0}}}},
    });

    EXPECT_THAT(input.get_shape(), ::testing::ElementsAre(3, 1, 3, 3));

    EXPECT_EQ(GlobalAveragePooling2DLayer().forward(input),
              Tensor::from_vector(tensor4d{
                  {{{0.7666666666666667}}},
                  {{{1.0}}},
                  {{{0.3333333333333333}}},
              }));
}

TEST(TensorTest, TestToBytes)
{
    Tensor input = Tensor::from_vector(tensor4d{
        {{{{3.2, 2.1, 1.0},
           {-0.3, 0.2, 0.1},
           {0.3, 0.2, 0.1}}}},
    });

    input.save("test.bin"); // -> shape_size (u8) + 4 * shape_siz  + 12 * 4 = 48 bytes
    Tensor output = Tensor::from_bytes("test.bin");
    EXPECT_EQ(input, output);
}