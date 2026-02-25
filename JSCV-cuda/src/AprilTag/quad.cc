//
// Created by Jace Rodgers on 2/25/26.
//

#include "quad.hh"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace detect {

    QuadDetector::QuadDetector(float max_corner_distance)
        : max_corner_distance_(max_corner_distance) {}

    std::vector<Quad> QuadDetector::detectQuads(
        const float2 *corner_points,
        int corner_count,
        size_t image_width,
        size_t image_height) {

        std::vector<Quad> quads;

        if (corner_count < 4) {
            std::cerr << "Not enough corners detected: " << corner_count << std::endl;
            return quads;
        }

        std::vector<Point2f> points(corner_count);
        for (int i = 0; i < corner_count; ++i) {
            points[i] = Point2f(corner_points[i].x, corner_points[i].y);
        }

        for (int i = 0; i < corner_count - 3; ++i) {
            for (int j = i + 1; j < corner_count - 2; ++j) {
                for (int k = j + 1; k < corner_count - 1; ++k) {
                    for (int l = k + 1; l < corner_count; ++l) {
                        Quad quad;
                        if (tryFormQuad(points[i], points[j], points[k], points[l], quad)) {
                            quads.push_back(quad);
                        }
                    }
                }
            }
        }

        std::sort(quads.begin(), quads.end(),
                  [](const Quad& a, const Quad& b) { return a.area > b.area; });

        if (quads.size() > 10) {
            quads.resize(10);
        }

        std::cout << "Detected " << quads.size() << " quads from "
                  << corner_count << " corners" << std::endl;

        return quads;
    }

    bool QuadDetector::tryFormQuad(
        const Point2f& p0, const Point2f& p1,
        const Point2f& p2, const Point2f& p3,
        Quad& out_quad) {

        Point2f pts[4] = {p0, p1, p2, p3};

        if (!orderQuadPoints(pts)) {
            return false;
        }

        if (!isValidQuadGeometry(pts)) {
            return false;
        }

        for (int i = 0; i < 4; ++i) {
            out_quad.corners[i] = pts[i];
        }
        out_quad.area = out_quad.computeArea();

        return out_quad.isValid();
    }

    bool QuadDetector::orderQuadPoints(Point2f points[4]) {
        Point2f centroid(0, 0);
        for (int i = 0; i < 4; ++i) {
            centroid = centroid + points[i];
        }
        centroid = centroid * 0.25f;

        std::sort(points, points + 4, [&centroid](const Point2f& a, const Point2f& b) {
            float angle_a = atan2f(a.y - centroid.y, a.x - centroid.x);
            float angle_b = atan2f(b.y - centroid.y, b.x - centroid.x);
            return angle_a < angle_b;
        });

        int min_idx = 0;
        float min_y = points[0].y;
        for (int i = 1; i < 4; ++i) {
            if (points[i].y < min_y ||
                (points[i].y == min_y && points[i].x < points[min_idx].x)) {
                min_y = points[i].y;
                min_idx = i;
            }
        }

        Point2f temp[4];
        for (int i = 0; i < 4; ++i) {
            temp[i] = points[(min_idx + i) % 4];
        }
        for (int i = 0; i < 4; ++i) {
            points[i] = temp[i];
        }

        return true;
    }

    bool QuadDetector::isValidQuadGeometry(const Point2f points[4]) {

        float sides[4];
        for (int i = 0; i < 4; ++i) {
            sides[i] = points[i].distance(points[(i + 1) % 4]);
        }

        float min_side = *std::min_element(sides, sides + 4);
        float max_side = *std::max_element(sides, sides + 4);

        if (max_side > 2.0f * min_side) {
            return false;
        }

        // Check that diagonals exist and make sense
        float diag1 = points[0].distance(points[2]);
        float diag2 = points[1].distance(points[3]);

        if (fabsf(diag1 - diag2) > max_corner_distance_) {
            return false;
        }

        if (diag1 < 1.2f * max_side || diag2 < 1.2f * max_side) {
            return false;
        }

        return true;
    }

}  // namespace detect