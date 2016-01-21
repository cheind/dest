/**
    This file is part of Deformable Shape Tracking (DEST).

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/

#include <dest/core/config.h>
#ifdef DEST_WITH_OPENCV

#include <dest/util/draw.h>
#include <dest/util/log.h>
#include <dest/util/convert.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

namespace dest {
    namespace util {
        
        void drawShape(cv::Mat &img, const core::Shape &s, const cv::Scalar &color) {
            
            for (core::Shape::Index i = 0; i < s.cols(); ++i) {
                cv::circle(img, cv::Point2f(s(0, i), s(1 ,i)), 1.f, color, -1, CV_AA);
            }
            
        }
        
        void drawShape(cv::Mat &img, const core::Shape &s, int colormap) {
            
            cv::Mat values(1, static_cast<int>(s.cols()), CV_8UC1);
            for (int i = 0; i < (int)s.cols(); ++i)
                values.at<uchar>(0, i) = static_cast<uchar>(i);
            cv::normalize(values, values, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            
            cv::Mat colors;
            cv::applyColorMap(values, colors, colormap);
            
            for (core::Shape::Index i = 0; i < s.cols(); ++i) {
                cv::Scalar color = colors.at<cv::Vec3b>(0, static_cast<int>(i));
                cv::circle(img, cv::Point2f(s(0, i), s(1 ,i)), 1.f, color, -1, CV_AA);
            }
        }
        
        void drawShapeText(cv::Mat &img, const core::Shape &s, const cv::Scalar &color) {
            for (core::Shape::Index i = 0; i < s.cols(); ++i) {
                std::ostringstream str;
                str << i;
                cv::putText(img, str.str(), cv::Point2f(s(0, i), s(1 ,i)), CV_FONT_HERSHEY_PLAIN, 0.7f, color);
            }

        }
        
        void drawRect(cv::Mat &img, const core::Rect &r, const cv::Scalar &color) {
            cv::line(img, cv::Point2f(r(0,0), r(1,0)), cv::Point2f(r(0,1), r(1,1)), color, 1, CV_AA);
            cv::line(img, cv::Point2f(r(0,1), r(1,1)), cv::Point2f(r(0,3), r(1,3)), color, 1, CV_AA);
            cv::line(img, cv::Point2f(r(0,3), r(1,3)), cv::Point2f(r(0,2), r(1,2)), color, 1, CV_AA);
            cv::line(img, cv::Point2f(r(0,2), r(1,2)), cv::Point2f(r(0,0), r(1,0)), color, 1, CV_AA);
        }
        
        cv::Mat drawShape(const core::Image &img, const core::Shape &s, const cv::Scalar &color)
        {
            cv::Mat tmp, tmp2;

            util::toCVHeaderOnly(img, tmp);
            cv::cvtColor(tmp, tmp2, CV_GRAY2BGR);
            
            drawShape(tmp2, s, color);

            return tmp2;
        }
        
    }
}

#endif