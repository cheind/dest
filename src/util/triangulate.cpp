/**
    This file is part of Deformable Shape Tracking (DEST).

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/

#include <dest/core/config.h>
#ifdef DEST_WITH_OPENCV

#include <dest/core/shape.h>
#include <dest/util/triangulate.h>
#include <opencv2/imgproc.hpp>

namespace dest {
    namespace util {
        
        std::vector<core::Shape::Index> triangulateShape(const core::Shape & s)
        {
            Eigen::Vector2f minC = s.rowwise().minCoeff();
            Eigen::Vector2f maxC = s.rowwise().maxCoeff();

            // Don't make the bounds too tight.
            cv::Rect_<float> bounds(std::floor(minC.x() - 1.f),
                std::floor(minC.y() - 1.f),
                std::ceil(maxC.x() - minC.x() + 2.f),
                std::ceil(maxC.y() - minC.y() + 2.f));

            cv::Subdiv2D subdiv(bounds);

            std::vector<cv::Point2f> controlPoints;

            for (dest::core::Shape::Index i = 0; i < s.cols(); ++i) {
                cv::Point2f c(s(0, i), s(1, i));
                subdiv.insert(c);
                controlPoints.push_back(c);
            }

            std::vector<cv::Vec6f> triangleList;
            subdiv.getTriangleList(triangleList);

            std::vector<core::Shape::Index> triangleIds(triangleList.size() * 3);

            int validTris = 0;
            for (size_t i = 0; i < triangleList.size(); i++)
            {
                cv::Vec6f t = triangleList[i];

                cv::Point2f p0(t[0], t[1]);
                cv::Point2f p1(t[2], t[3]);
                cv::Point2f p2(t[4], t[5]);

                if (bounds.contains(p0) && bounds.contains(p1) && bounds.contains(p2)) {

                    auto iter0 = std::find(controlPoints.begin(), controlPoints.end(), p0);
                    auto iter1 = std::find(controlPoints.begin(), controlPoints.end(), p1);
                    auto iter2 = std::find(controlPoints.begin(), controlPoints.end(), p2);

                    triangleIds[validTris * 3 + 0] = (core::Shape::Index)std::distance(controlPoints.begin(), iter0);
                    triangleIds[validTris * 3 + 1] = (core::Shape::Index)std::distance(controlPoints.begin(), iter1);
                    triangleIds[validTris * 3 + 2] = (core::Shape::Index)std::distance(controlPoints.begin(), iter2);

                    ++validTris;
                }
            }

            return std::vector<core::Shape::Index>(triangleIds.begin(), triangleIds.begin() + validTris * 3);
        }
    }
}

#endif