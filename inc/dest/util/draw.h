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

#ifndef DEST_DRAW_H
#define DEST_DRAW_H

#include <dest/core/config.h>
#include <dest/core/shape.h>
#include <dest/core/image.h>

#ifdef DEST_WITH_OPENCV
#include <opencv2/core/core.hpp>

namespace dest {
    namespace util {
        
        void drawShape(cv::Mat &img, const core::Shape &s, const cv::Scalar &color);
        
        cv::Mat drawShape(const core::Image &img, const core::Shape &s, const cv::Scalar &color);
        
    }
}

#endif

#endif