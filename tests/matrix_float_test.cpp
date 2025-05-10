#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdio>

#define NTT_MATRIX_STATIC
#define NTT_MATRIX_IMPLEMENTATION
#include <ntt_matrix.hpp>

TEST(MatrixFloatTest, DefaultConstructor)
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

TEST(MatrixFloatTest, ConstructorWithDefaultValue)
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

TEST(MatrixFloatTest, SetElement)
{
    ntt::Matrix matrix(3, 3, 1);
    matrix.set_element(0, 0, 2);
    EXPECT_EQ(matrix.get_element(0, 0), 2);
}

TEST(MatrixFloatTest, CopyConstructor)
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

TEST(MatrixFloatTest, TestCreateMatrixFromVectorVector)
{
    std::vector<std::vector<float>> vector = {{0.35, 0.45},
                                              {0.4, 0.5}};
    ntt::Matrix matrix = ntt::Matrix::create_from_vector_vector(vector);

    EXPECT_EQ(matrix.get_rows(), 2);
    EXPECT_EQ(matrix.get_columns(), 2);
    EXPECT_THAT(matrix.get_element(0, 0), testing::FloatEq(0.35));
    EXPECT_THAT(matrix.get_element(0, 1), testing::FloatEq(0.45));
    EXPECT_THAT(matrix.get_element(1, 0), testing::FloatEq(0.4));
    EXPECT_THAT(matrix.get_element(1, 1), testing::FloatEq(0.5));
}

TEST(MatrixFloatTest, DotProduct)
{
    ntt::Matrix matrix = ntt::Matrix::create_from_vector_vector({{0.35, 0.45},
                                                                 {0.4, 0.5}});

    ntt::Matrix matrix2 = ntt::Matrix::create_from_vector_vector({{1, 1},
                                                                  {1, 1}});

    // results: [[0.8, 0.8],
    //           [0.9, 0.9]]
    ntt::Matrix expectedResult = ntt::Matrix::create_from_vector_vector({{0.8, 0.8},
                                                                         {0.9, 0.9}});

    ntt::Matrix result = matrix.dot(matrix2);
    EXPECT_TRUE(result == expectedResult);
}

TEST(MatrixFloatTest, Equality)
{
    ntt::Matrix matrix = ntt::Matrix::create_from_vector_vector({{0.35, 0.45},
                                                                 {0.4, 0.5}});

    ntt::Matrix matrix2 = ntt::Matrix::create_from_vector_vector({{0.35, 0.45},
                                                                  {0.4, 0.5}});

    EXPECT_TRUE(matrix == matrix2);
}

TEST(MatrixFloatTest, Inequality)
{
    ntt::Matrix matrix = ntt::Matrix::create_from_vector_vector({{0.35, 0.45},
                                                                 {0.4, 0.5}});

    ntt::Matrix matrix2 = ntt::Matrix::create_from_vector_vector({{0.35, 0.45},
                                                                  {0.4, 0.0}});

    EXPECT_FALSE(matrix == matrix2);
}

TEST(MatrixFloatTest, TestNonMatchedSizeDotProduct)
{
    ntt::Matrix matrix = ntt::Matrix::create_from_vector_vector({{0.35},
                                                                 {0.4}});
    ntt::Matrix matrix2 = ntt::Matrix::create_from_vector_vector({{1, 1, 1},
                                                                  {1, 1, 1}});

    EXPECT_THROW(matrix.dot(matrix2), std::invalid_argument);
}
