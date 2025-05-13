#include <cstdio>
#include <opencv2/opencv.hpp>
#define NTT_MICRO_NN_IMPLEMENTATION
#include <ntt_very_super_micro_dnn/ntt_tensor.hpp>

using namespace ntt;

int main(void)
{
    // Tensor conv1_weight = Tensor::load();

    Conv2DLayer conv1(conv1_weight, conv1_bias.reshape_clone({conv1_bias.getTotalElements(), 1}), 2, 1, 1);
    Clip2DLayer clip1(0, 6);
    Conv2DLayer conv2(conv2_weight, conv2_bias.reshape_clone({conv2_bias.getTotalElements(), 1}), 1, 1, 24);
    Clip2DLayer clip2(0, 6);
    Conv2DLayer conv3(conv3_weight, conv3_bias.reshape_clone({conv3_bias.getTotalElements(), 1}), 1, 0, 1);
    Conv2DLayer conv4(conv4_weight, conv4_bias.reshape_clone({conv4_bias.getTotalElements(), 1}), 1, 0, 1);
    Clip2DLayer clip4(0, 6);
    Conv2DLayer conv5(conv5_weight, conv5_bias.reshape_clone({conv5_bias.getTotalElements(), 1}), 2, 1, 64);
    Clip2DLayer clip5(0, 6);
    Conv2DLayer conv6(conv6_weight, conv6_bias.reshape_clone({conv6_bias.getTotalElements(), 1}), 1, 0, 64);

    std::vector<Layer *> chunk1 = {&conv1, &clip1, &conv2, &clip2, &conv3, &conv4, &clip4, &conv5, &clip5, &conv6};

    Conv2DLayer conv7(conv7_weight, conv7_bias.reshape_clone({conv7_bias.getTotalElements(), 1}), 1, 0, 1);
    Clip2DLayer clip7(0, 6);
    Conv2DLayer conv8(conv8_weight, conv8_bias.reshape_clone({conv8_bias.getTotalElements(), 1}), 1, 1, 144);
    Clip2DLayer clip8(0, 6);
    Conv2DLayer conv9(conv9_weight, conv9_bias.reshape_clone({conv9_bias.getTotalElements(), 1}), 1, 0, 1);

    std::vector<Layer *> chunk2 = {&conv7, &clip7, &conv8, &clip8, &conv9};

    Conv2DLayer conv10(conv10_weight, conv10_bias.reshape_clone({conv10_bias.getTotalElements(), 1}), 1, 0, 1);
    Clip2DLayer clip10(0, 6);
    Conv2DLayer conv11(conv11_weight, conv11_bias.reshape_clone({conv11_bias.getTotalElements(), 1}), 2, 2, 144);
    Clip2DLayer clip11(0, 6);
    Conv2DLayer conv12(conv12_weight, conv12_bias.reshape_clone({conv12_bias.getTotalElements(), 1}), 1, 0, 1);

    std::vector<Layer *> chunk3 = {&conv10, &clip10, &conv11, &clip11, &conv12};

    Conv2DLayer conv13(conv13_weight, conv13_bias.reshape_clone({conv13_bias.getTotalElements(), 1}), 1, 0, 1);
    Clip2DLayer clip13(0, 6);
    Conv2DLayer conv14(conv14_weight, conv14_bias.reshape_clone({conv14_bias.getTotalElements(), 1}), 1, 2, 240);
    Clip2DLayer clip14(0, 6);
    Conv2DLayer conv15(conv15_weight, conv15_bias.reshape_clone({conv15_bias.getTotalElements(), 1}), 1, 0, 1);

    std::vector<Layer *> chunk4 = {&conv13, &clip13, &conv14, &clip14, &conv15};

    Conv2DLayer conv16(conv16_weight, conv16_bias.reshape_clone({conv16_bias.getTotalElements(), 1}), 1, 0, 1);
    Clip2DLayer clip16(0, 6);
    Conv2DLayer conv17(conv17_weight, conv17_bias.reshape_clone({conv17_bias.getTotalElements(), 1}), 2, 1, 240);
    Clip2DLayer clip17(0, 6);
    Conv2DLayer conv18(conv18_weight, conv18_bias.reshape_clone({conv18_bias.getTotalElements(), 1}), 1, 0, 1);

    std::vector<Layer *> chunk5 = {&conv16, &clip16, &conv17, &clip17, &conv18};

    Conv2DLayer conv19(conv19_weight, conv19_bias.reshape_clone({conv19_bias.getTotalElements(), 1}), 1, 0, 1);
    Clip2DLayer clip19(0, 6);
    Conv2DLayer conv20(conv20_weight, conv20_bias.reshape_clone({conv20_bias.getTotalElements(), 1}), 1, 1, 480);
    Clip2DLayer clip20(0, 6);
    Conv2DLayer conv21(conv21_weight, conv21_bias.reshape_clone({conv21_bias.getTotalElements(), 1}), 1, 0, 1);

    std::vector<Layer *> chunk6 = {&conv19, &clip19, &conv20, &clip20, &conv21};

    Conv2DLayer conv22(conv22_weight, conv22_bias.reshape_clone({conv22_bias.getTotalElements(), 1}), 1, 0, 1);
    Clip2DLayer clip22(0, 6);
    Conv2DLayer conv23(conv23_weight, conv23_bias.reshape_clone({conv23_bias.getTotalElements(), 1}), 1, 1, 480);
    Clip2DLayer clip23(0, 6);
    Conv2DLayer conv24(conv24_weight, conv24_bias.reshape_clone({conv24_bias.getTotalElements(), 1}), 1, 0, 1);

    std::vector<Layer *> chunk7 = {&conv22, &clip22, &conv23, &clip23, &conv24};

    Conv2DLayer conv25(conv25_weight, conv25_bias.reshape_clone({conv25_bias.getTotalElements(), 1}), 1, 0, 1);
    Clip2DLayer clip25(0, 6);
    Conv2DLayer conv26(conv26_weight, conv26_bias.reshape_clone({conv26_bias.getTotalElements(), 1}), 1, 2, 480);
    Clip2DLayer clip26(0, 6);
    Conv2DLayer conv27(conv27_weight, conv27_bias.reshape_clone({conv27_bias.getTotalElements(), 1}), 1, 0, 1);

    std::vector<Layer *> chunk8 = {&conv25, &clip25, &conv26, &clip26, &conv27};

    Conv2DLayer conv28(conv28_weight, conv28_bias.reshape_clone({conv28_bias.getTotalElements(), 1}), 1, 0, 1);
    Clip2DLayer clip28(0, 6);
    Conv2DLayer conv29(conv29_weight, conv29_bias.reshape_clone({conv29_bias.getTotalElements(), 1}), 1, 2, 672);
    Clip2DLayer clip29(0, 6);
    Conv2DLayer conv30(conv30_weight, conv30_bias.reshape_clone({conv30_bias.getTotalElements(), 1}), 1, 0, 1);

    std::vector<Layer *> chunk9 = {&conv28, &clip28, &conv29, &clip29, &conv30};

    Conv2DLayer conv31(conv31_weight, conv31_bias.reshape_clone({conv31_bias.getTotalElements(), 1}), 1, 0, 1);
    Clip2DLayer clip31(0, 6);
    Conv2DLayer conv32(conv32_weight, conv32_bias.reshape_clone({conv32_bias.getTotalElements(), 1}), 1, 2, 672);
    Clip2DLayer clip32(0, 6);
    Conv2DLayer conv33(conv33_weight, conv33_bias.reshape_clone({conv33_bias.getTotalElements(), 1}), 1, 0, 1);

    std::vector<Layer *> chunk10 = {&conv31, &clip31, &conv32, &clip32, &conv33};

    Conv2DLayer conv34(conv34_weight, conv34_bias.reshape_clone({conv34_bias.getTotalElements(), 1}), 1, 0, 1);
    Clip2DLayer clip34(0, 6);
    Conv2DLayer conv35(conv35_weight, conv35_bias.reshape_clone({conv35_bias.getTotalElements(), 1}), 2, 2, 672);
    Clip2DLayer clip35(0, 6);
    Conv2DLayer conv36(conv36_weight, conv36_bias.reshape_clone({conv36_bias.getTotalElements(), 1}), 1, 0, 1);

    std::vector<Layer *> chunk11 = {&conv34, &clip34, &conv35, &clip35, &conv36};

    Conv2DLayer conv37(conv37_weight, conv37_bias.reshape_clone({conv37_bias.getTotalElements(), 1}), 1, 0, 1);
    Clip2DLayer clip37(0, 6);
    Conv2DLayer conv38(conv38_weight, conv38_bias.reshape_clone({conv38_bias.getTotalElements(), 1}), 1, 2, 1152);
    Clip2DLayer clip38(0, 6);
    Conv2DLayer conv39(conv39_weight, conv39_bias.reshape_clone({conv39_bias.getTotalElements(), 1}), 1, 0, 1);

    std::vector<Layer *> chunk12 = {&conv37, &clip37, &conv38, &clip38, &conv39};

    Conv2DLayer conv40(conv40_weight, conv40_bias.reshape_clone({conv40_bias.getTotalElements(), 1}), 1, 0, 1);
    Clip2DLayer clip40(0, 6);
    Conv2DLayer conv41(conv41_weight, conv41_bias.reshape_clone({conv41_bias.getTotalElements(), 1}), 1, 2, 1152);
    Clip2DLayer clip41(0, 6);
    Conv2DLayer conv42(conv42_weight, conv42_bias.reshape_clone({conv42_bias.getTotalElements(), 1}), 1, 0, 1);

    std::vector<Layer *> chunk13 = {&conv40, &clip40, &conv41, &clip41, &conv42};

    Conv2DLayer conv43(conv43_weight, conv43_bias.reshape_clone({conv43_bias.getTotalElements(), 1}), 1, 0, 1);
    Clip2DLayer clip43(0, 6);
    Conv2DLayer conv44(conv44_weight, conv44_bias.reshape_clone({conv44_bias.getTotalElements(), 1}), 1, 2, 1152);
    Clip2DLayer clip44(0, 6);
    Conv2DLayer conv45(conv45_weight, conv45_bias.reshape_clone({conv45_bias.getTotalElements(), 1}), 1, 0, 1);

    std::vector<Layer *> chunk14 = {&conv43, &clip43, &conv44, &clip44, &conv45};

    Conv2DLayer conv46(conv46_weight, conv46_bias.reshape_clone({conv46_bias.getTotalElements(), 1}), 1, 0, 1);
    Clip2DLayer clip46(0, 6);
    Conv2DLayer conv47(conv47_weight, conv47_bias.reshape_clone({conv47_bias.getTotalElements(), 1}), 1, 1, 1152);
    Clip2DLayer clip47(0, 6);
    GlobalAveragePooling2DLayer gap47;

    std::vector<Layer *> chunk15 = {&conv46, &clip46, &conv47, &clip47, &gap47};

    return 0;
}