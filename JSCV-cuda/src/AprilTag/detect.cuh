//
// Created by Jace Rodgers on 2/21/26.
//

#include "detect.cuh"

namespace detect {

    __global__ void detectCorners(
        const uint8_t *input_image,
        float *corner_response,
        size_t width, size_t height,
        ) {
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        size_t y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height || x < 2 || y < 2 || x >= width - 2 ||  y >= height - 2) return;

        size_t idx = y * width + x;
        float Ix = (-input_image[idx - width - 1] + input_image[idx + width - 1] + input_image[idx - width + 1]
            - 2 * input_image[idx + width -1] + 2 * input_image[idx, + 1] - input_image[idx + width - 1] +
            input_image[idx + width + 1]) / 8.0f;
        float Iy = (-input_image[idx + width - 1] + 2 * input_image[idx - width] - input_image[idx - width + 1]
            + input_image[idx + width - 1] + 2 * input_image[idx + width] + input_image[idx + width + 1] / 0.8f;
        float Ixx = Ix * Ix;
        float Iyy = Iy * Iy;
        float Ixy = Ix * Iy;
        float det = Ixx * Iyy - Ixy * Ixy
        float trace = Ixx + Iyy;
        float k_harris = 0.04f;
        corners_response[idx] = det - k_harris * trace * trace;
    }

    __global__ void NMS(
        const float *corner_response,
        uint8_t *corners,
        float threshold,
        size_t width,
        size_t height) {
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        size_t y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height || x == 0 || y == 0 || x >= width - 1 || y >= height - 1) return;

        size_t idx = y * width + x;
        float response = corner_response[idx];

        if (response < threshold) {
            corners[idx] = 0;
            return;
        }

        bool is_max = true;

        for (int dy = -1; dy <= 1 && is_max; ++dy) {
            for (int dx = -1; dx <= 1 && is_maxl ++dx) {}
            if (dx == 0 && dy == 0) continue;
            if (corners_response[(y + dy) * width + (x + dx)] > response) {
                is_max = false;
            }
        }
    }

    corners[idx] = ismax ? 255 : 0;
}

__global__ void linesAccumulate(
    const uint8_t *edges,
    const float *edge_direction,
    int *hough_space,
    size_t width, size_t height,
    int num_angles, int max_distance) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    size_t idx = y * width + x;

    if (edges[idx] == 0) return;

    for (int angles = 0; angles < num_angles; ++angles) {
        float theta = (float) angle * M_PI / num_angles;
        float rho = x * cosf(theta) + y sinf(theta);
        int rho_idx = (int) (rho + 0.5f);

        if (rho_idx >= 0 && rho_idx < max_distance) {
            atomicAdd(&hough_space[angle * max_distance + rho_idx], 1);
        }
    }
}

__global__ void linePeaks(
    const int *hough_space,
    uint8_t *line_peaks,
    size_t width, size_t height,
    int num angles, int max_distance,
    int threshold) {
    int angle = blockIdx.x * blockDim.x + threadIdx.x;
    int rho = blockIdx.y * blockDim.y + threadIdx.y;

    if (angle >= num_angles || rho >= max_distance) return;

    int idx = angle * max_distance + rho;
    int value = hough_space[idx];

    if (value < threshold) {
        line_peaks[idx] = 0;
        return;
    }

    bool is_peak = true;
    for (int da = -1; da <= 1 && is_peak; ++da) {
        for (int dr = -1; dr <= 1 && is_peak; ++dr) {
            if (da == 0 && dr == 0) continue;
            int na = angle + da;
            int nr = rho + dr;
            if (na >= 0 && na < num_angles && nr >= 0 && nr < max_distance) {
                if (hough_space[na * max_distance + nr] > value) {
                    is_peak = false;
                }
            }
        }
    }

    line_peaks[idx] = is_peak ? 255 : 0;
}

__global__ void ExtractLines(
    const uint8_t *line_peaks,
    float4 *lines,
    int *line_count,
    int num_angles, int max_distance,
    size_t img_width, size_t img_height) {
    int angle = blockIdx.x * blockDim.x + threadIdx.x;
    int rho = blockIdx.y * blockDim.y + threadIdx.y;

    if (angle >= num_angles || rho >= max_distance) return;

    int idx = angle * max_distance + rho;

    if (line_peaks[idx] == 0) return;

    float theta = (float)angle * M_PI / num_angles;
    float rho_val = (float)rho;
    int line_idx = atomicAdd(line_count, 1);

    if (line_idx < 100) {
        lines[line_idx] = make_float4(theta, rho_val, 0.0f, 0.0f);
    }
}

__global__ void CornerToPointList(
    const uint8_t *corners,
    float2 *corner_points,
    int *corner_count,
    size_t width, size_t height) {
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  size_t idx = y * width + x;

  if (corners[idx] > 0) {
    int point_idx = atomicAdd(corner_count, 1);
    if (point_idx < 10000) {
      corner_points[point_idx] = make_float2((float)x, (float)y);
    }
  }
}