#include <gtest/gtest.h>
#define NTT_MICRO_NN_STATIC
#define NTT_MICRO_NN_I8
#define NTT_MICRO_NN_IMPLEMENTATION
#include <ntt_very_super_micro_dnn/ntt_matrix.hpp>

TEST(MatrixI8Test, DefaultConstructor)
{
    ntt::Tensor matrix(2, 2);
    EXPECT_EQ(matrix.get_rows(), 2);
    EXPECT_EQ(matrix.get_columns(), 2);
}

TEST(MatrixI8Test, ConstructorWithDefaultValue)
{
    ntt::Tensor matrix(2, 2, 1);
    EXPECT_EQ(matrix.get_element(0, 0), 1);
    EXPECT_EQ(matrix.get_element(0, 1), 1);
}

TEST(MatrixI8Test, CopyConstructor)
{
    ntt::Tensor matrix(2, 2, 1);
    ntt::Tensor matrix2(matrix);
    EXPECT_EQ(matrix2.get_element(0, 0), 1);
    EXPECT_EQ(matrix2.get_element(0, 1), 1);
}

TEST(MatrixI8Test, DotProduct)
{
    ntt::Tensor matrix(2, 2);
    matrix.set_element(0, 0, 1);
    matrix.set_element(0, 1, 2);
    matrix.set_element(1, 0, 3);
    matrix.set_element(1, 1, 4);
}

TEST(MatrixI8Test, Equality)
{
    ntt::Tensor matrix(2, 2);
    matrix.set_element(0, 0, 1);
    matrix.set_element(0, 1, 2);
    matrix.set_element(1, 0, 3);
}

TEST(MatrixI8Test, Inequality)
{
    ntt::Tensor matrix(2, 2);
    matrix.set_element(0, 0, 1);
    matrix.set_element(0, 1, 2);
    matrix.set_element(1, 0, 3);
}
