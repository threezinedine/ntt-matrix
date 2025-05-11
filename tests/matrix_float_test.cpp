#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdio>

#define NTT_MICRO_NN_STATIC
#define NTT_MICRO_NN_FLOAT
#define NTT_MICRO_NN_IMPLEMENTATION
#include <ntt_very_super_micro_dnn/ntt_matrix.hpp>

TEST(MatrixFloatTest, DefaultConstructor)
{
    ntt::Tensor matrix(3, 3);
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
    ntt::Tensor matrix(3, 3, 1);
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
    ntt::Tensor matrix(3, 3, 1);
    matrix.set_element(0, 0, 2);
    EXPECT_EQ(matrix.get_element(0, 0), 2);
}

TEST(MatrixFloatTest, CopyConstructor)
{
    ntt::Tensor matrix(3, 3, 1);
    ntt::Tensor matrix2(matrix);

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
    ntt::Tensor matrix = ntt::Tensor::create_from_vector_vector({{0.35, 0.45},
                                                                 {0.4, 0.5}});

    EXPECT_EQ(matrix.get_rows(), 2);
    EXPECT_EQ(matrix.get_columns(), 2);
    EXPECT_THAT(matrix.get_element(0, 0), testing::FloatEq(0.35));
    EXPECT_THAT(matrix.get_element(0, 1), testing::FloatEq(0.45));
    EXPECT_THAT(matrix.get_element(1, 0), testing::FloatEq(0.4));
    EXPECT_THAT(matrix.get_element(1, 1), testing::FloatEq(0.5));
}

TEST(MatrixFloatTest, DotProduct)
{
    ntt::Tensor matrix = ntt::Tensor::create_from_vector_vector({{0.35, 0.45},
                                                                 {0.4, 0.5}});

    ntt::Tensor matrix2 = ntt::Tensor::create_from_vector_vector({{1, 1},
                                                                  {1, 1}});

    // results: [[0.8, 0.8],
    //           [0.9, 0.9]]
    ntt::Tensor expectedResult = ntt::Tensor::create_from_vector_vector({{0.8, 0.8},
                                                                         {0.9, 0.9}});

    ntt::Tensor result = matrix.dot(matrix2);
    EXPECT_TRUE(result == expectedResult);
}

TEST(MatrixFloatTest, Equality)
{
    ntt::Tensor matrix = ntt::Tensor::create_from_vector_vector({{0.35, 0.45},
                                                                 {0.4, 0.5}});

    ntt::Tensor matrix2 = ntt::Tensor::create_from_vector_vector({{0.35, 0.45},
                                                                  {0.4, 0.5}});

    EXPECT_TRUE(matrix == matrix2);
}

TEST(MatrixFloatTest, Inequality)
{
    ntt::Tensor matrix = ntt::Tensor::create_from_vector_vector({{0.35, 0.45},
                                                                 {0.4, 0.5}});

    ntt::Tensor matrix2 = ntt::Tensor::create_from_vector_vector({{0.35, 0.45},
                                                                  {0.4, 0.0}});

    EXPECT_FALSE(matrix == matrix2);
}

TEST(MatrixFloatTest, TestMutiply)
{
    ntt::Tensor matrix = ntt::Tensor::create_from_vector_vector({{0.35, 0.45},
                                                                 {0.4, 0.5}});

    ntt::Tensor expectedResult = ntt::Tensor::create_from_vector_vector({{0.7, 0.9},
                                                                         {0.8, 1}});

    ntt::Tensor result = matrix * 2;
    EXPECT_TRUE(result == expectedResult);
}

TEST(MatrixFloatTest, TestNonMatchedSizeDotProduct)
{
    ntt::Tensor matrix = ntt::Tensor::create_from_vector_vector({{0.35},
                                                                 {0.4}});
    ntt::Tensor matrix2 = ntt::Tensor::create_from_vector_vector({{1, 1, 1},
                                                                  {1, 1, 1}});

    EXPECT_THROW(matrix.dot(matrix2), std::invalid_argument);
}

TEST(MatrixFloatTest, TestTranspose)
{
    ntt::Tensor matrix = ntt::Tensor::create_from_vector_vector({{0.35, 0.45},
                                                                 {0.4, 0.5}});

    ntt::Tensor expectedResult = ntt::Tensor::create_from_vector_vector({{0.35, 0.4},
                                                                         {0.45, 0.5}});

    ntt::Tensor result = matrix.transpose();
    EXPECT_TRUE(result == expectedResult);
}

TEST(MatrixFloatTest, TestVectorTranspose)
{
    ntt::Tensor matrix = ntt::Tensor::create_from_vector_vector({{0.35, 0.45}});

    ntt::Tensor expectedResult = ntt::Tensor::create_from_vector_vector({{0.35},
                                                                         {0.45}});

    ntt::Tensor result = matrix.transpose();
    EXPECT_TRUE(result == expectedResult);
}

TEST(MatrixFloatTest, AddAnotherMatrix)
{
    ntt::Tensor matrix = ntt::Tensor::create_from_vector_vector({{0.35, 0.45},
                                                                 {0.4, 0.5}});

    ntt::Tensor matrix2 = ntt::Tensor::create_from_vector_vector({{1, 1},
                                                                  {1, 1}});

    ntt::Tensor expectedResult = ntt::Tensor::create_from_vector_vector({{1.35, 1.45},
                                                                         {1.4, 1.5}});

    ntt::Tensor result = matrix + matrix2;
    EXPECT_TRUE(result == expectedResult);
}

TEST(MatrixFloatTest, TestNonMatchedSizeAdd)
{
    ntt::Tensor matrix = ntt::Tensor::create_from_vector_vector({{0.35, 0.45},
                                                                 {0.4, 0.5}});

    ntt::Tensor matrix2 = ntt::Tensor::create_from_vector_vector({{1, 1, 1},
                                                                  {1, 1, 1}});

    EXPECT_THROW(matrix + matrix2, std::invalid_argument);
}

TEST(MatrixFloatTest, AddMatrixWithNumber)
{
    ntt::Tensor matrix = ntt::Tensor::create_from_vector_vector({{0.35, 0.45},
                                                                 {0.4, 0.5}});

    ntt::Tensor expectedResult = ntt::Tensor::create_from_vector_vector({{1.35, 1.45},
                                                                         {1.4, 1.5}});

    ntt::Tensor result = matrix + 1;
    EXPECT_TRUE(result == expectedResult);
}

TEST(MatrixFloatTest, TestSubtractAnotherMatrix)
{
    ntt::Tensor matrix = ntt::Tensor::create_from_vector_vector({{0.35, 0.45},
                                                                 {0.4, 0.5}});

    ntt::Tensor matrix2 = ntt::Tensor::create_from_vector_vector({{1, 1},
                                                                  {1, 1}});
}

TEST(MatrixFloatTest, TestSubtractMatrixWithNumber)
{
    ntt::Tensor matrix = ntt::Tensor::create_from_vector_vector({{1.35, 1.45},
                                                                 {1.4, 1.5}});

    ntt::Tensor expectedResult = ntt::Tensor::create_from_vector_vector({{0.35, 0.45},
                                                                         {0.4, 0.5}});

    ntt::Tensor result = matrix - 1;
    EXPECT_TRUE(result == expectedResult);
}

TEST(MatrixFloatTest, TestNonMatchedSizeSubtract)
{
    ntt::Tensor matrix = ntt::Tensor::create_from_vector_vector({{0.35, 0.45},
                                                                 {0.4, 0.5}});

    ntt::Tensor matrix2 = ntt::Tensor::create_from_vector_vector({{1, 1, 1},
                                                                  {1, 1, 1}});

    EXPECT_THROW(matrix - matrix2, std::invalid_argument);
}

TEST(MatrixFloatTest, TestNegative)
{
    ntt::Tensor matrix = ntt::Tensor::create_from_vector_vector({{0.35, 0.45},
                                                                 {0.4, 0.5}});

    ntt::Tensor expectedResult = ntt::Tensor::create_from_vector_vector({{-0.35, -0.45},
                                                                         {-0.4, -0.5}});

    ntt::Tensor result = matrix.negative();
    EXPECT_TRUE(result == expectedResult);
}

TEST(MatrixFloatTest, TestSubtract)
{
    ntt::Tensor matrix = ntt::Tensor::create_from_vector_vector({{0.35, 0.45},
                                                                 {0.4, 0.5}});

    ntt::Tensor matrix2 = ntt::Tensor::create_from_vector_vector({{0.2, 0.12},
                                                                  {0.8, 1.5}});

    ntt::Tensor expectedResult = ntt::Tensor::create_from_vector_vector({{0.15, 0.33},
                                                                         {-0.4, -1}});

    ntt::Tensor result = matrix - matrix2;
    EXPECT_TRUE(result == expectedResult);
}

TEST(MatrixFloatTest, CreateIdentityMatrix)
{
    ntt::Tensor matrix = ntt::Tensor::create_identity_matrix(3);
    ntt::Tensor matrix2 = ntt::Tensor::create_from_vector_vector({{3, -2.3, 0},
                                                                  {2.1, 0, 0.2},
                                                                  {0, 0, 0}});

    ntt::Tensor expectedResult = ntt::Tensor::create_from_vector_vector({{1, 0, 0},
                                                                         {0, 1, 0},
                                                                         {0, 0, 1}});

    EXPECT_TRUE(matrix == expectedResult);
    EXPECT_TRUE(matrix2.dot(matrix) == matrix2);
}

TEST(MatrixFloatTest, TestDivide)
{
    ntt::Tensor matrix = ntt::Tensor::create_from_vector_vector({{1, 2, 3},
                                                                 {4, 5, 6},
                                                                 {7, 8, 9}});

    ntt::Tensor expectedResult = ntt::Tensor::create_from_vector_vector({{0.5, 1, 1.5},
                                                                         {2, 2.5, 3},
                                                                         {3.5, 4, 4.5}});

    ntt::Tensor result = matrix / 2;
    EXPECT_TRUE(result == expectedResult);
}

TEST(MatrixFloatTest, TestAddPadding)
{
    ntt::Tensor matrix = ntt::Tensor::create_from_vector_vector({{1, 0, 0},
                                                                 {0, 1, 0},
                                                                 {0, 0, 1}});

    ntt::Tensor expectedResult = ntt::Tensor::create_from_vector_vector({{0, 0, 0, 0, 0},
                                                                         {0, 1, 0, 0, 0},
                                                                         {0, 0, 1, 0, 0},
                                                                         {0, 0, 0, 1, 0},
                                                                         {0, 0, 0, 0, 0}});

    ntt::Tensor result = matrix.add_padding(1);
    EXPECT_TRUE(result == expectedResult);
}

TEST(MatrixFloatTest, TestSliding)
{
    // create random 4x4 matrix
    ntt::Tensor matrix = ntt::Tensor::create_from_vector_vector({{1, 2, 3, 4},
                                                                 {5, 6, 7, 8},
                                                                 {9, 10, 11, 12},
                                                                 {13, 14, 15, 16}});

    struct MaxPoolingData
    {
        ntt::Tensor result;

        MaxPoolingData(size_t rows, size_t columns) : result(rows, columns) {}
    };

    ntt::sliding_callback callback = [](size_t startRow,
                                        size_t startColumn,
                                        size_t endRow,
                                        size_t endColumn,
                                        ntt::Tensor &matrix,
                                        void *data)
    {
        MaxPoolingData *maxPoolingData = (MaxPoolingData *)data;
        float max = matrix.get_element(startRow, startColumn);
        for (size_t i = startRow; i < endRow; i++)
        {
            for (size_t j = startColumn; j < endColumn; j++)
            {
                if (matrix.get_element(i, j) > max)
                {
                    max = matrix.get_element(i, j);
                }
            }
        }
        maxPoolingData->result.set_element(startRow / 2, startColumn / 2, max);
    };

    MaxPoolingData maxPoolingData(matrix.get_rows() / 2, matrix.get_columns() / 2);
    matrix.sliding(callback, 2, 2, 2, 2, &maxPoolingData);

    ntt::Tensor expectedResult = ntt::Tensor::create_from_vector_vector({{6, 8},
                                                                         {14, 16}});

    EXPECT_TRUE(maxPoolingData.result == expectedResult);
}

TEST(MatrixFloatTest, MatrixFloatTest_Reshape_Test)
{
    ntt::Tensor matrix = ntt::Tensor::create_from_vector_vector({{1, 2, 3, 4},
                                                                 {5, 6, 7, 8}});

    ntt::Tensor expectedResult = ntt::Tensor::create_from_vector_vector({{1, 2},
                                                                         {3, 4},
                                                                         {5, 6},
                                                                         {7, 8}});

    matrix.reshape(4, 2);
    EXPECT_TRUE(matrix == expectedResult);
}

TEST(MatrixFloatTest, MatrixFloatTest_ToShape_Test)
{
    ntt::Tensor matrix = ntt::Tensor::create_from_vector_vector({{1, 2, 3, 4},
                                                                 {5, 6, 7, 8}});

    ntt::Tensor expectedResult = ntt::Tensor::create_from_vector_vector({{1, 2},
                                                                         {3, 4},
                                                                         {5, 6},
                                                                         {7, 8}});

    ntt::Tensor result = matrix.toShape(4, 2);
    EXPECT_TRUE(result == expectedResult);
}

TEST(MatrixFloatTest, MatrixFloatTest_Max_Test)
{
    ntt::Tensor matrix = ntt::Tensor::create_from_vector_vector({{1, 2, 3, 4},
                                                                 {5, 10, 7, 8}});

    ntt::Tensor expectedMatrixResult = ntt::Tensor::create_from_vector_vector({{10}});
    ntt::Tensor expectedRowResult = ntt::Tensor::create_from_vector_vector({{4},
                                                                            {10}});
    ntt::Tensor expectedColumnResult = ntt::Tensor::create_from_vector_vector({{5, 10, 7, 8}});

    EXPECT_TRUE(matrix.max(ntt::Tensor::Axis::MATRIX) == expectedMatrixResult);
    EXPECT_TRUE(matrix.max(ntt::Tensor::Axis::ROW) == expectedRowResult);
    EXPECT_TRUE(matrix.max(ntt::Tensor::Axis::COLUMN) == expectedColumnResult);
}

TEST(MatrixFloatTest, MatrixFloatTest_Argmax_Test)
{
    ntt::Tensor matrix = ntt::Tensor::create_from_vector_vector({{1, 2, 3, 4}});

    EXPECT_THAT(matrix.argmax(), 3);
}

TEST(MatrixFloatTest, MatrixFloatTest_Argmax_Test_2)
{
    ntt::Tensor matrix = ntt::Tensor::create_from_vector_vector({{1},
                                                                 {3},
                                                                 {-1}});

    EXPECT_THAT(matrix.argmax(), 1);
}

TEST(MatrixFloatTest, MatrixFloatTest_CrossCorrelation_Test)
{
    ntt::Tensor matrix = ntt::Tensor::create_from_vector_vector({{1, 2, 3, 4},
                                                                 {5, 6, 7, 8},
                                                                 {6, 2, 3, 1},
                                                                 {1, 2, 3, 4}});

    ntt::Tensor kernel = ntt::Tensor::create_from_vector_vector({{1, 1},
                                                                 {1, 1}});

    ntt::Tensor expectedResult = ntt::Tensor::create_from_vector_vector({{14, 18, 22},
                                                                         {19, 18, 19},
                                                                         {11, 10, 11}});

    ntt::Tensor result = matrix.cross_correlation(kernel);
    EXPECT_TRUE(result == expectedResult);
}

TEST(MatrixFloatTest, MatrixFloatTest_CrossCorrelation_Test_2)
{
    ntt::Tensor matrix = ntt::Tensor::create_from_vector_vector({{1, 2, 3, 4},
                                                                 {5, 6, 7, 8},
                                                                 {6, 2, 3, 1},
                                                                 {1, 2, 3, 4}});

    ntt::Tensor kernel = ntt::Tensor::create_from_vector_vector({{1, 0.5},
                                                                 {0.5, 1}});

    ntt::Tensor expectedResult = ntt::Tensor::create_from_vector_vector({{10.5, 16.5},
                                                                         {9.5, 9}});

    ntt::Tensor result = matrix.cross_correlation(kernel, 2);
    EXPECT_TRUE(result == expectedResult);
}

TEST(MatrixFloatTest, MatrixFloatTest_Clip_Test)
{
    ntt::Tensor matrix = ntt::Tensor::create_from_vector_vector({{1, 2, 3, 4},
                                                                 {5, 6, 7, 8}});

    ntt::Tensor expectedResult = ntt::Tensor::create_from_vector_vector({{3, 3, 3, 4},
                                                                         {5, 6, 6, 6}});

    ntt::Tensor result = ntt::ClipLayer(3, 6).forward(matrix);
    EXPECT_TRUE(result == expectedResult);
}
