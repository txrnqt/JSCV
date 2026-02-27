#pragma once

#define CHANNELS 3
#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>

class CudaStream;

namespace preprocess {
__global__ void colorToGray(
    const unsigned char *rgb,
    unsigned char *gray,
    int rows,
    int cols) {

    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (col >= cols || row >= rows) return;

    int gray_offset = row * cols + col;
    int rgb_offset = gray_offset * CHANNELS;

    unsigned char r = rgb[rgb_offset + 0];
    unsigned char g = rgb[rgb_offset + 1];
    unsigned char b = rgb[rgb_offset + 2];

    gray[gray_offset] =
        static_cast<unsigned char>(0.299f * r +
                                   0.587f * g +
                                   0.114f * b);
}

void decimate(
    const uint8_t *gray_image,
    uint8_t *decimated_image,
    uint8_t *unfiltered_minmax_image,
    uint8_t *minmax_image,
    uint8_t *thresholded_image,
    size_t width,
    size_t height,
    size_t min_white_black_diff,
    CudaStream *stream) {

    CHECK((width % 8) == 0);
    CHECK((height % 8) == 0);

    constexpr size_t kThreads = 256;

    {
        size_t kBlocks =
            (width * height + kThreads - 1) / kThreads / 4;

        InternalCudaDecimate<<<kBlocks, kThreads, 0,
                               stream->get()>>>(
            gray_image,
            decimated_image,
            width,
            height);

        MaybeCheckAndSynchronize();
    }

    size_t decimated_width = width / 2;
    size_t decimated_height = height / 2;

    {
        dim3 threads(16, 16, 1);
        dim3 blocks((decimated_width / 4 + 15) / 16,
                    (decimated_height / 4 + 15) / 16, 1);

        InternalBlockMinMax<<<blocks, threads, 0,
                              stream->get()>>>(
            decimated_image,
            reinterpret_cast<uchar2 *>(
                unfiltered_minmax_image),
            decimated_width / 4,
            decimated_height / 4);

        MaybeCheckAndSynchronize();

        InternalBlockFilter<<<blocks, threads, 0,
                              stream->get()>>>(
            reinterpret_cast<uchar2 *>(
                unfiltered_minmax_image),
            reinterpret_cast<uchar2 *>(minmax_image),
            decimated_width / 4,
            decimated_height / 4);

        MaybeCheckAndSynchronize();
    }

    {
        size_t kBlocks =
            (width * height / 4 + kThreads - 1) /
            kThreads / 4;

        InternalThreshold<<<kBlocks, kThreads, 0,
                            stream->get()>>>(
            decimated_image,
            reinterpret_cast<uchar2 *>(minmax_image),
            thresholded_image,
            decimated_width,
            decimated_height,
            min_white_black_diff);

        MaybeCheckAndSynchronize();
    }
}

__global__ void adaptiveThresholdSauvola(
    const uint8_t *input_image,
    uint8_t *output_image,
    size_t width,
    size_t height,
    int neighborhood_size,
    float k,
    float R) {

    size_t x =
        blockIdx.x * blockDim.x + threadIdx.x;
    size_t y =
        blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int offset = neighborhood_size / 2;
    int sum = 0, sum_sq = 0, count = 0;

    for (int dy = -offset; dy <= offset; ++dy) {
        for (int dx = -offset; dx <= offset; ++dx) {

            int nx = static_cast<int>(x) + dx;
            int ny = static_cast<int>(y) + dy;

            if (nx >= 0 && nx < (int)width &&
                ny >= 0 && ny < (int)height) {

                uint8_t pixel =
                    input_image[ny * width + nx];

                sum += pixel;
                sum_sq += pixel * pixel;
                count++;
            }
        }
    }

    float mean = (float)sum / count;
    float variance =
        ((float)sum_sq / count) -
        (mean * mean);

    float std_dev =
        sqrtf(fmaxf(0.0f, variance));

    float threshold =
        mean * (1.0f +
                k * ((std_dev / R) - 1.0f));

    output_image[y * width + x] =
        (input_image[y * width + x] >
         threshold)
            ? 255
            : 0;
}

__global__ void connectedComponetLabeling(
    const uint8_t *binary_image,
    uint32_t *labeled_image,
    size_t width,
    size_t height) {

    size_t x =
        blockIdx.x * blockDim.x + threadIdx.x;
    size_t y =
        blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    size_t idx = y * width + x;

    if (binary_image[idx] == 0) {
        labeled_image[idx] = 0;
        return;
    }

    labeled_image[idx] =
        static_cast<uint32_t>(idx + 1);
}

__global__ void mergeLabels(
    uint32_t *labeled_image,
    size_t width,
    size_t height) {

    size_t x =
        blockIdx.x * blockDim.x + threadIdx.x;
    size_t y =
        blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    size_t idx = y * width + x;
    uint32_t current =
        labeled_image[idx];

    if (current == 0) return;

    uint32_t min_label = current;

    if (x > 0 &&
        labeled_image[idx - 1] != 0) {
        min_label =
            min(min_label,
                labeled_image[idx - 1]);
    }

    if (y > 0 &&
        labeled_image[idx - width] != 0) {
        min_label =
            min(min_label,
                labeled_image[idx - width]);
    }

    labeled_image[idx] = min_label;

}} // namespace preprocess