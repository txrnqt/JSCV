//
// Created by Jace Rodgers on 2/21/26.
//

#pragma once

#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>

namespace detect {

    __global__ void detectCorners(
        const uint8_t *input_image,
        float *corner_response,
        size_t width,
        size_t height
    );

    __global__ void NMS(
        const float *corner_response,
        uint8_t *corners,
        float threshold,
        size_t width,
        size_t height
    );

    __global__ void linesAccumulate(
        const uint8_t *edges,
        const float *edge_direction,
        int *hough_space,
        size_t width,
        size_t height,
        int num_angles,
        int max_distance
    );

    __global__ void linePeaks(
        const int *hough_space,
        uint8_t *line_peaks,
        size_t width,
        size_t height,
        int num_angles,
        int max_distance,
        int threshold
    );

    __global__ void ExtractLines(
        const uint8_t *line_peaks,
        float4 *lines,
        int *line_count,
        int num_angles,
        int max_distance,
        size_t img_width,
        size_t img_height
    );

    __global__ void CornerToPointList(
        const uint8_t *corners,
        float2 *corner_points,
        int *corner_count,
        size_t width,
        size_t height
    );

} // namespace detect}