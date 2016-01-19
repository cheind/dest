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

#ifndef DEST_CONVERT_H
#define DEST_CONVERT_H

#include <dest/core/config.h>
#if !defined(DEST_WITH_OPENCV)
#error OpenCV is required for this part of DEST.
#endif

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace dest {
    namespace util {

        /**
            Convert OpenCV image to DEST.
        */
        inline void toDest(const cv::Mat &src, core::Image &dst) {
            dst.resize(src.rows, src.cols);

            cv::Mat singleChannel;
            if (src.channels() == 3) {
                cv::cvtColor(src, singleChannel, CV_BGR2GRAY);
            } else {
                singleChannel = src;
            }

            const int outerStride = static_cast<int>(singleChannel.step[0] / sizeof(unsigned char));

            typedef Eigen::Map<const core::Image, 0, Eigen::OuterStride<Eigen::Dynamic> > MapType;

            MapType map(singleChannel.ptr<unsigned char>(), singleChannel.rows, singleChannel.cols, Eigen::OuterStride<Eigen::Dynamic>(outerStride));
            dst = map;
        }

        /**
            Convert DEST image to OpenCV header.
        */
        inline void toCVHeaderOnly(const core::Image &src, cv::Mat &dst) {

            const int rows = static_cast<int>(src.rows());
            const int cols = static_cast<int>(src.cols());

            dst = cv::Mat(rows, cols, CV_8UC1, const_cast<unsigned char*>(src.data()));
        }

        /**
            Convert DEST image to OpenCV.
        */
        inline void toCV(const core::Image &src, cv::Mat &dst) {

            cv::Mat hdr;
            toCVHeaderOnly(src, hdr);
            hdr.copyTo(dst);
        }
        
        /**
            Convert OpenCV rectangle to DEST.
        */
        inline void toDest(const cv::Rect &src, core::Rect &dst) {
            dst = core::createRectangle(Eigen::Vector2f(src.tl().x, src.tl().y), Eigen::Vector2f(src.br().x, src.br().y));
        }
        
        /**
            Convert DEST rectangle to OpenCV.
        */
        inline void toCV(const core::Rect &src, cv::Rect_<float> &dst) {
            dst.x = src(0,0);
            dst.y = src(1,0);
            dst.width = src(0,3) - src(0,0);
            dst.height = src(1,3) - src(1,0);
        }

    }
}

#endif