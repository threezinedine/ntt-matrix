#include <gtest/gtest.h>
#include <gmock/gmock.h>

#define NTT_MATRIX_IMPLEMENTATION
#include <ntt_matrix.hpp>

TEST(MatrixTest, DefaultConstructor)
{
    ntt::Matrix matrix(3, 3);
    EXPECT_EQ(matrix.get_rows(), 3);
    EXPECT_EQ(matrix.get_columns(), 3);
    EXPECT_EQ(matrix.get_element(0, 0), 0);
    EXPECT_EQ(matrix.get_element(0, 1), 0);
    EXPECT_EQ(matrix.get_element(0, 2), 0);
    EXPECT_EQ(matrix.get_element(1, 0), 0);
    EXPECT_EQ(matrix.get_element(1, 1), 0);
    EXPECT_EQ(matrix.get_element(1, 2), 0);
    EXPECT_EQ(matrix.get_element(2, 0), 0);
    EXPECT_EQ(matrix.get_element(2, 1), 0);
    EXPECT_EQ(matrix.get_element(2, 2), 0);
}

TEST(MatrixTest, ConstructorWithDefaultValue)
{
    ntt::Matrix matrix(3, 3, 1);
    EXPECT_EQ(matrix.get_element(0, 0), 1);
    EXPECT_EQ(matrix.get_element(0, 1), 1);
    EXPECT_EQ(matrix.get_element(0, 2), 1);
    EXPECT_EQ(matrix.get_element(1, 0), 1);
    EXPECT_EQ(matrix.get_element(1, 1), 1);
    EXPECT_EQ(matrix.get_element(1, 2), 1);
    EXPECT_EQ(matrix.get_element(2, 0), 1);
    EXPECT_EQ(matrix.get_element(2, 1), 1);
    EXPECT_EQ(matrix.get_element(2, 2), 1);
}

TEST(MatrixTest, SetElement)
{
    ntt::Matrix matrix(3, 3, 1);
    matrix.set_element(0, 0, 2);
    EXPECT_EQ(matrix.get_element(0, 0), 2);
}

TEST(MatrixTest, CopyConstructor)
{
    ntt::Matrix matrix(3, 3, 1);
    ntt::Matrix matrix2(matrix);

    EXPECT_EQ(matrix.get_rows(), matrix2.get_rows());
    EXPECT_EQ(matrix.get_columns(), matrix2.get_columns());

    EXPECT_EQ(matrix2.get_element(0, 0), matrix.get_element(0, 0));
    EXPECT_EQ(matrix2.get_element(0, 1), matrix.get_element(0, 1));
    EXPECT_EQ(matrix2.get_element(0, 2), matrix.get_element(0, 2));
    EXPECT_EQ(matrix2.get_element(1, 0), matrix.get_element(1, 0));
    EXPECT_EQ(matrix2.get_element(1, 1), matrix.get_element(1, 1));
    EXPECT_EQ(matrix2.get_element(1, 2), matrix.get_element(1, 2));
    EXPECT_EQ(matrix2.get_element(2, 0), matrix.get_element(2, 0));
    EXPECT_EQ(matrix2.get_element(2, 1), matrix.get_element(2, 1));
    EXPECT_EQ(matrix2.get_element(2, 2), matrix.get_element(2, 2));
}
