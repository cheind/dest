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
        
        /**
            Draw shape to OpenCV image.

            \param img Image to draw to.
            \param s Shape to draw.
            \param color Color to use.
        */
        void drawShape(cv::Mat &img, const core::Shape &s, const cv::Scalar &color);

        /**
            Draw shape to OpenCV image using different colors for each landmark.

            \param img Image to draw to.
            \param s Shape to draw.
            \param colormap OpenCV colormap see cv::applyColorMap.
        */
        void drawShape(cv::Mat &img, const core::Shape &s, int colormap);

        /**
            Draw landmark ids rendered as text.

            \param img Image to draw to.
            \param s Shape to draw.
            \param color Color to use.
        */
        void drawShapeText(cv::Mat &img, const core::Shape &s, const cv::Scalar &color);

        /**
            Draw rectangle.

            \param img Image to draw to.
            \param r Rectangle to draw.
            \param color Color to use.
        */
        void drawRect(cv::Mat &img, const core::Rect &r, const cv::Scalar &color);
        
        /**
            Draw shape.

            Usually used as a kick-starter to get initial OpenCV image with rendered shape.

            \param img Image to draw to. Copy will be made and returned by this method.
            \param s Shape to draw.
            \param color Color to use.
        */
        cv::Mat drawShape(const core::Image &img, const core::Shape &s, const cv::Scalar &color);
        
    }
}

#endif

#endif