#include <cstdio>

#define NTT_MICRO_NN_IMPLEMENTATION
#include <ntt_very_super_micro_dnn/ntt_matrix.hpp>

using namespace ntt;

void print_matrix(const Tensor &matrix);

int main(void)
{
    printf("Tensor 1:\n");
    ntt::Tensor matrix(3, 3);
    print_matrix(matrix);

    printf("Tensor 2:\n");
    ntt::Tensor matrix2(3, 3, 1);
    print_matrix(matrix2);

    // dot product

    return 0;
}

void print_matrix(const Tensor &matrix)
{
    for (size_t i = 0; i < matrix.get_rows(); i++)
    {
        for (size_t j = 0; j < matrix.get_columns(); j++)
        {
            printf("%f ", matrix.get_element(i, j));
        }

        printf("\n");
    }
}
