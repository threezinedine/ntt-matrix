#include <cstdio>

#define NTT_MICRO_NN_IMPLEMENTATION
#include <ntt_very_super_micro_dnn/ntt_matrix.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace ntt;

int main(void)
{
#include "fc1_weight.tasm"
#include "fc1_bias.tasm"
#include "fc2_weight.tasm"
#include "fc2_bias.tasm"
#include "fc3_weight.tasm"
#include "fc3_bias.tasm"

    // Matrix input = Matrix::create_from_vector_vector({{1.0f, 2.0f, 3.0f}}).toShape(3, 1);
    int width, height, channels;
    unsigned char *data = stbi_load(
        // "C:/Users/Acer/Project/ntt-very-super-micro-dnn/examples/test_idx_2691_label_8.png",
        "C:/Users/Acer/Project/ntt-very-super-micro-dnn/examples/test_idx_9915_label_4.png",
        &width, &height, &channels, 0);
    Matrix inputMatrix(height, width);
    if (data)
    {
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                inputMatrix.set_element(i, j, data[i * height + j]);
            }
        }
    }
    else
    {
        printf("Error vcb\n");
        exit(-1);
    }

    printf("Width: %d, Height: %d, Channel: %d", width, height, channels);
    printf("Matrix: %s", inputMatrix.to_string().c_str());

    FullyConnectedLayer fc1(fc1_weight, fc1_bias.transpose());
    ReLU relu1 = ReLU();
    FullyConnectedLayer fc2(fc2_weight, fc2_bias.transpose());
    ReLU relu2 = ReLU();
    FullyConnectedLayer fc3(fc3_weight, fc3_bias.transpose());

    std::vector<Layer *> layers = {&fc1, &relu1, &fc2, &relu2, &fc3};

    Matrix output = inputMatrix.toShape(width * height, 1);

    for (auto const &layer : layers)
    {
        output = layer->forward(output);
    }

    printf("output: %s\n", output.to_string().c_str());
    printf("Finished\n");
    stbi_image_free(data);
    return 0;
}
