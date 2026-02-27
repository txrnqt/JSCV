//
// Created by Jace Rodgers on 2/25/26.
//

#pragma once

#include <vector>
#include <cuda_runtime.h>

namespace detect {

    struct Point2f {
        float x, y;

        Point2f() = default;
        Point2f(float x_, float y_) : x(x_), y(y_) {}

        Point2f operator+(const Point2f& p) const { return Point2f(x + p.x, y + p.y); }
        Point2f operator-(const Point2f& p) const { return Point2f(x - p.x, y - p.y); }
        Point2f operator*(float s) const { return Point2f(x * s, y * s); }

        float dot(const Point2f& p) const { return x * p.x + y * p.y; }
        float cross(const Point2f& p) const { return x * p.y - y * p.x; }
        float distance(const Point2f& p) const {
            Point2f d = *this - p;
            return sqrtf(d.dot(d));
        }
        float length() const { return sqrtf(x * x + y * y); }
    };

    struct Quad {
        Point2f corners[4];
        float area;

        Quad() : area(0.0f) {}

        float computeArea() const {
            float a = 0.5f * fabsf(
                corners[0].cross(corners[1]) +
                corners[1].cross(corners[2]) +
                corners[2].cross(corners[3]) +
                corners[3].cross(corners[0])
            );
            return a;
        }

        inline bool isValid(float min_area = 100.0f) const {
            if (area < min_area) return false;

            for (int i = 0; i < 4; ++i) {
                for (int j = i + 1; j < 4; ++j) {
                    if (corners[i].distance(corners[j]) < 5.0f) return false;
                }
            }

            return true;
        }
    };

    class QuadDetector {
    public:
        QuadDetector(float max_corner_distance = 50.0f);

        std::vector<Quad> detectQuads(
            const float2 *corner_points,
            int corner_count,
            size_t image_width,
            size_t image_height
        );

    private:
        float max_corner_distance_;

        bool tryFormQuad(
            const Point2f& p0, const Point2f& p1,
            const Point2f& p2, const Point2f& p3,
            Quad& out_quad
        );

        bool orderQuadPoints(Point2f points[4]);

        bool isValidQuadGeometry(const Point2f points[4]);
    };

}  // namespace detect