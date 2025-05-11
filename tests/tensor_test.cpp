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
