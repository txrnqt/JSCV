#pragma once

#include <cstddef>
#include <cstdint>

class CudaStream;

namespace preprocess {

    __global__ void colorToGray(
        const unsigned char *rgb,
        unsigned char *gray,
        int rows,
        int cols);

    void decimate(
        const uint8_t *gray_image,
        uint8_t *decimated_image,
        uint8_t *unfiltered_minmax_image,
        uint8_t *minmax_image,
        uint8_t *thresholded_image,
        size_t width,
        size_t height,
        size_t min_white_black_diff,
        CudaStream *stream);

    __global__ void adaptiveThresholdSauvola(
        const uint8_t *input_image,
        uint8_t *output_image,
        size_t width,
        size_t height,
        int neighborhood_size,
        float k,
        float R);

    __global__ void connectedBlobLabeling(
        const uint8_t *binary_image,
        uint32_t *labeled_image,
        size_t width,
        size_t height);

    __global__ void mergeLabels(
        uint32_t *labeled_image,
        size_t width,
        size_t height);

}