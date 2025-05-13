#include <cstdio>
#define NTT_MICRO_NN_IMPLEMENTATION
#include <ntt_very_super_micro_dnn/ntt_tensor.hpp>

using namespace ntt;

int main(void)
{
    Tensor tensor({2, 2, 2}, 1.0f);
    printf("tensor.to_string(): %s\n", tensor.to_string().c_str());
    printf("tensor.flatten(): %s\n", tensor.flatten().c_str());

    tensor.set_element({0, 0, 0}, 4.23);
    printf("tensor.to_string(): %s\n", tensor.to_string().c_str());
    printf("tensor.flatten(): %s\n", tensor.flatten().c_str());

    tensor.set_element({1, 0, 0}, -2.16f);
    printf("tensor.to_string(): %s\n", tensor.to_string().c_str());
    printf("tensor.flatten(): %s\n", tensor.flatten().c_str());

    tensor.set_element({1, 1, 1}, -1.0f);
    printf("tensor.to_string(): %s\n", tensor.to_string().c_str());
    printf("tensor.flatten(): %s\n", tensor.flatten().c_str());

    tensor.set_element({1, 1, 0}, 2.0f);
    printf("tensor.to_string(): %s\n", tensor.to_string().c_str());
    printf("tensor.flatten(): %s\n", tensor.flatten().c_str());

    return 0;
}
