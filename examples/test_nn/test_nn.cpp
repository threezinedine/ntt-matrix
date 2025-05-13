#include <cstdio>
#define NTT_MICRO_NN_IMPLEMENTATION
#include <ntt_very_super_micro_dnn/ntt_matrix.hpp>

using namespace ntt;

int main(void)
{
    Matrix input = Matrix::create_from_vector_vector({{1.0f},
                                                      {2.0f},
                                                      {3.0f}});

    Matrix weights = Matrix::create_from_vector_vector({{1.0f, 2.0f, 3.0f}});
    Matrix biases = Matrix::create_from_vector_vector({{1.0f}});

    Matrix output = FullyConnectedLayer(weights, biases).forward(input);

    printf("output: %s\n", output.to_string().c_str());

    return 0;
}