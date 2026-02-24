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

__global__ void QuadFitting(
    const float2 *corner_points,
    int num_corners,
    float4 *quads,
    int *quad_count,
    float min_quad_area) {
  int quad_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (quad_idx >= min((int)(num_corners * num_corners / 100), 1000)) return;

  int corner_a = (quad_idx / (num_corners * num_corners)) % num_corners;
  int corner_b = (quad_idx / num_corners) % num_corners;
  int corner_c = quad_idx % num_corners;
  int corner_d = (quad_idx + 1) % num_corners;

  if (corner_a == corner_b || corner_b == corner_c || corner_c == corner_d) return;

  float2 p1 = corner_points[corner_a];
  float2 p2 = corner_points[corner_b];
  float2 p3 = corner_points[corner_c];
  float2 p4 = corner_points[corner_d];
  float dx1 = p2.x - p1.x, dy1 = p2.y - p1.y;
  float dx2 = p3.x - p2.x, dy2 = p3.y - p2.y;
  float cross = dx1 * dy2 - dy1 * dx2;
  float area = fabsf(cross) / 2.0f;

  if (area < min_quad_area) return;

  float d12 = sqrtf(dx1*dx1 + dy1*dy1);
  float d23 = sqrtf(dx2*dx2 + dy2*dy2);
  float aspect = fmaxf(d12, d23) / fminf(d12, d23 + 1e-6f);

  if (aspect > 2.0f) return;

  int stored_idx = atomicAdd(quad_count, 1);
  if (stored_idx < 1000) {
    quads[stored_idx * 2] = make_float4(p1.x, p1.y, p2.x, p2.y);
    quads[stored_idx * 2 + 1] = make_float4(p3.x, p3.y, p4.x, p4.y);
  }
}

__global__ void RefineQuadCorners(
    const uint8_t *edges,
    float4 *quads,
    int num_quads,
    size_t width, size_t height,
    int search_radius) {

  int quad_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (quad_idx >= num_quads) return;

  for (int corner = 0; corner < 4; ++corner) {
    float2 current;
    if (corner < 2) {
      current = make_float2(quads[quad_idx * 2].x, quads[quad_idx * 2].y);
      if (corner == 1) {
        current = make_float2(quads[quad_idx * 2].z, quads[quad_idx * 2].w);
      }
    } else {
      current = make_float2(quads[quad_idx * 2 + 1].x, quads[quad_idx * 2 + 1].y);
      if (corner == 3) {
        current = make_float2(quads[quad_idx * 2 + 1].z, quads[quad_idx * 2 + 1].w);
      }
    }

    float best_dist = FLT_MAX;
    float2 best_pos = current;

    for (int dy = -search_radius; dy <= search_radius; ++dy) {
      for (int dx = -search_radius; dx <= search_radius; ++dx) {
        int nx = (int)current.x + dx;
        int ny = (int)current.y + dy;

        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
          if (edges[ny * width + nx] > 0) {
            float dist = dx*dx + dy*dy;
            if (dist < best_dist) {
              best_dist = dist;
              best_pos = make_float2((float)nx, (float)ny);
            }
          }
        }
      }
    }

    if (corner == 0) {
      quads[quad_idx * 2].x = best_pos.x;
      quads[quad_idx * 2].y = best_pos.y;
    } else if (corner == 1) {
      quads[quad_idx * 2].z = best_pos.x;
      quads[quad_idx * 2].w = best_pos.y;
    } else if (corner == 2) {
      quads[quad_idx * 2 + 1].x = best_pos.x;
      quads[quad_idx * 2 + 1].y = best_pos.y;
    } else {
      quads[quad_idx * 2 + 1].z = best_pos.x;
      quads[quad_idx * 2 + 1].w = best_pos.y;
    }
  }
}