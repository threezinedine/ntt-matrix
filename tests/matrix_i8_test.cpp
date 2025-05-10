#include <gtest/gtest.h>
#define NTT_MATRIX_STATIC
#define NTT_MATRIX_I8
#define NTT_MATRIX_IMPLEMENTATION
#include <ntt_matrix.hpp>

TEST(MatrixI8Test, DefaultConstructor)
{
    ntt::Matrix matrix(2, 2);
    EXPECT_EQ(matrix.get_rows(), 2);
    EXPECT_EQ(matrix.get_columns(), 2);
}

TEST(MatrixI8Test, ConstructorWithDefaultValue)
{
    ntt::Matrix matrix(2, 2, 1);
    EXPECT_EQ(matrix.get_element(0, 0), 1);
    EXPECT_EQ(matrix.get_element(0, 1), 1);
}

TEST(MatrixI8Test, CopyConstructor)
{
    ntt::Matrix matrix(2, 2, 1);
    ntt::Matrix matrix2(matrix);
    EXPECT_EQ(matrix2.get_element(0, 0), 1);
    EXPECT_EQ(matrix2.get_element(0, 1), 1);
}

TEST(MatrixI8Test, DotProduct)
{
    ntt::Matrix matrix(2, 2);
    matrix.set_element(0, 0, 1);
    matrix.set_element(0, 1, 2);
    matrix.set_element(1, 0, 3);
    matrix.set_element(1, 1, 4);
}

TEST(MatrixI8Test, Equality)
{
    ntt::Matrix matrix(2, 2);
    matrix.set_element(0, 0, 1);
    matrix.set_element(0, 1, 2);
    matrix.set_element(1, 0, 3);
}

TEST(MatrixI8Test, Inequality)
{
    ntt::Matrix matrix(2, 2);
    matrix.set_element(0, 0, 1);
    matrix.set_element(0, 1, 2);
    matrix.set_element(1, 0, 3);
}
