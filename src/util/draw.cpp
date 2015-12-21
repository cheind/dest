/**
 This file is part of Deformable Shape Tracking (DEST).
 
 Copyright Christoph Heindl 2015
 
 Deformable Shape Tracking is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 
 Deformable Shape Tracking is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with Deformable Shape Tracking. If not, see <http://www.gnu.org/licenses/>.
 */

#include <dest/util/draw.h>
#include <dest/util/convert.h>

#ifdef DEST_WITH_OPENCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

namespace dest {
    namespace util {
        
        void drawShape(cv::Mat &img, const core::Shape &s, const cv::Scalar &color) {
            
            for (core::Shape::Index i = 0; i < s.cols(); ++i) {
                cv::circle(img, cv::Point2f(s(0, i), s(1 ,i)), 2.f, color, -1, CV_AA);
            }
            
        }
        
        cv::Mat drawShape(const core::Image &img, const core::Shape &s, const cv::Scalar &color)
        {
            cv::Mat tmp, gray, tmp2;

            util::toCVHeaderOnly(img, tmp);
            tmp.convertTo(gray, CV_8U);
            cv::cvtColor(gray, tmp2, CV_GRAY2BGR);
            
            drawShape(tmp2, s, color);

            return tmp2;
        }
        
    }
}

#endif