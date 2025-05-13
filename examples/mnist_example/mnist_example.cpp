#include <cstdio>

#define NTT_MICRO_NN_IMPLEMENTATION
#include <ntt_very_super_micro_dnn/ntt_tensor.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace ntt;

int main(void)
{
    Tensor fc1_weight = Tensor::from_bytes("C:/Users/Acer/Project/ntt-very-super-micro-dnn/examples/mnist_example/fc1_weight.bin");
    Tensor fc1_bias = Tensor::from_bytes("C:/Users/Acer/Project/ntt-very-super-micro-dnn/examples/mnist_example/fc1_bias.bin");
    Tensor fc2_weight = Tensor::from_bytes("C:/Users/Acer/Project/ntt-very-super-micro-dnn/examples/mnist_example/fc2_weight.bin");
    Tensor fc2_bias = Tensor::from_bytes("C:/Users/Acer/Project/ntt-very-super-micro-dnn/examples/mnist_example/fc2_bias.bin");
    Tensor fc3_weight = Tensor::from_bytes("C:/Users/Acer/Project/ntt-very-super-micro-dnn/examples/mnist_example/fc3_weight.bin");
    Tensor fc3_bias = Tensor::from_bytes("C:/Users/Acer/Project/ntt-very-super-micro-dnn/examples/mnist_example/fc3_bias.bin");

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

    printf("Width: %d, Height: %d, Channel: %d", width, height, channels);
    inputMatrix = inputMatrix / 255.0f;

    FullyConnectedLayer fc1(fc1_weight, fc1_bias.reshape_clone({fc1_bias.get_shape()[0], 1}));
    ReLULayer relu1 = ReLULayer();
    FullyConnectedLayer fc2(fc2_weight, fc2_bias.reshape_clone({fc2_bias.get_shape()[0], 1}));
    ReLULayer relu2 = ReLULayer();
    FullyConnectedLayer fc3(fc3_weight, fc3_bias.reshape_clone({fc3_bias.get_shape()[0], 1}));
    SoftmaxLayer softmax = SoftmaxLayer();

    std::vector<Layer *> layers = {&fc1, &relu1, &fc2, &relu2, &fc3, &softmax};

    {
        Tensor output = inputMatrix.reshape_clone({static_cast<size_t>(width * height), 1});

        for (auto const &layer : layers)
        {
            output = layer->forward(output);
        }

        // output.reshape({output.getTotalElements()});
        printf("output: %s\n", output.to_string().c_str());
        printf("max: %f\n", output.max());
        printf("number: %s\n", Shape::convert_shape_to_string(output.argmax()).c_str());
    }
    printf("Finished\n");
    stbi_image_free(data);
    return 0;
}
