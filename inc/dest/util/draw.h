/**
    This file is part of Deformable Shape Tracking (DEST).

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/

#ifndef DEST_DRAW_H
#define DEST_DRAW_H

#include <dest/core/config.h>
#if !defined(DEST_WITH_OPENCV)
#error OpenCV is required for this part of DEST.
#endif

#include <dest/core/shape.h>
#include <dest/core/image.h>
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