#include <gtest/gtest.h>
#include <gmock/gmock.h>
#define NTT_MICRO_NN_STATIC
#define NTT_MICRO_NN_FLOAT
#define NTT_MICRO_NN_IMPLEMENTATION
#include <ntt_very_super_micro_dnn/ntt_matrix.hpp>

TEST(NNTest, FullyConnectedLayer)
{
    ntt::Tensor input = ntt::Tensor::create_from_vector_vector({{1.0f}, {2.0f}});

    ntt::Tensor weights = ntt::Tensor::create_from_vector_vector({{1.0f, 2.0f},
                                                                  {3.0f, 4.0f}});

    ntt::FullyConnectedLayer layer(weights, 2.0f);

    ntt::Tensor output = layer.forward(input);

    ntt::Tensor expectedOutput = ntt::Tensor::create_from_vector_vector(
        {{7},
         {13}});

    EXPECT_TRUE(output == expectedOutput);
}

TEST(NNTest, ReLU)
{
    ntt::Tensor input = ntt::Tensor::create_from_vector_vector({{1.0f}, {2.0f}, {-1.0f}, {-2.0f}});

    ntt::ReLU layer;

    ntt::Tensor output = layer.forward(input);

    ntt::Tensor expectedOutput = ntt::Tensor::create_from_vector_vector({{1.0f}, {2.0f}, {0.0f}, {0.0f}});

    EXPECT_TRUE(output == expectedOutput);
}

TEST(NNTest, Sigmoid)
{
    ntt::Tensor input = ntt::Tensor::create_from_vector_vector({{1.0f, -0.5f}});

    ntt::Sigmoid layer;

    ntt::Tensor output = layer.forward(input);

    ntt::Tensor expectedOutput = ntt::Tensor::create_from_vector_vector({{0.7310585786300049, 0.3775406687981454}});

    EXPECT_TRUE(output == expectedOutput);
}

TEST(NNTest, Softmax)
{
    ntt::Tensor input = ntt::Tensor::create_from_vector_vector({{1.0f},
                                                                {2.0f},
                                                                {3.0f}});

    ntt::Softmax layer;

    ntt::Tensor output = layer.forward(input);

    ntt::Tensor expectedOutput = ntt::Tensor::create_from_vector_vector({{0.09003057317038046},
                                                                         {0.24472847105479767},
                                                                         {0.6652409557748219}});

    printf("output: %s\n", output.to_string().c_str());
    printf("expectedOutput: %s\n", expectedOutput.to_string().c_str());

    EXPECT_TRUE(output == expectedOutput);
}
