#include <cstdio>

#define NTT_MICRO_NN_IMPLEMENTATION
#include <ntt_very_super_micro_dnn/ntt_tensor.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace ntt;

int main(void)
{
#include "conv2d1_weight.tasm"
#include "conv2d1_bias.tasm"
#include "fc4_weight.tasm"
#include "fc4_bias.tasm"

    // Matrix input = Matrix::create_from_vector_vector({{1.0f, 2.0f, 3.0f}}).toShape(3, 1);
    int width, height, channels;
    unsigned char *data = stbi_load(
        "C:/Users/Acer/Project/ntt-very-super-micro-dnn/examples/test_idx_2691_label_8.png",
        // "C:/Users/Acer/Project/ntt-very-super-micro-dnn/examples/test_idx_9915_label_4.png",
        &width, &height, &channels, 0);

    Tensor inputMatrix({static_cast<size_t>(height), static_cast<size_t>(width)});
    if (data)
    {
        for (size_t i = 0; i < height; i++)
        {
            for (size_t j = 0; j < width; j++)
            {
                inputMatrix.set_element({i, j}, data[i * height + j]);
            }
        }
    }
    else
    {
        printf("Error vcb\n");
        exit(-1);
    }

    Conv2DLayer conv2d1(conv2d1_weight, conv2d1_bias.reshape_clone({16, 1}), 1, 1);
    FlattenLayer flattenLayer;
    FullyConnectedLayer fc4(fc4_weight, fc4_bias.reshape_clone({10, 1}));
    SoftmaxLayer softmaxLayer;

    std::vector<Layer *> layers = {&conv2d1, &flattenLayer, &fc4, &softmaxLayer};

    Tensor output = inputMatrix.reshape_clone({1, 1, static_cast<size_t>(height), static_cast<size_t>(width)});
    output = output / 255.0f;

    for (Layer *layer : layers)
    {
        output = layer->forward(output);
    }

    output.reshape({output.getTotalElements()});
    printf("output : %s\n", output.to_string().c_str());
    printf("output max : %s\n", Shape::convert_shape_to_string(output.argmax()).c_str());

    printf("Finished\n");
    stbi_image_free(data);
    return 0;
}
