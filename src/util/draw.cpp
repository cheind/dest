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
                cv::Vec3b temp = colors.at<cv::Vec3b>(0, static_cast<int>(i));
                cv::Scalar color(temp[0], temp[1], temp[2]);
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

        void drawShapeTriangles(cv::Mat & img, const core::Shape & s, const std::vector<core::Shape::Index>& tris, cv::Scalar color)
        {
            for (size_t i = 0; i < tris.size() / 3; ++i) {

                cv::line(img,
                    cv::Point2f(s(0, tris[i * 3 + 0]), s(1, tris[i * 3 + 0])),
                    cv::Point2f(s(0, tris[i * 3 + 1]), s(1, tris[i * 3 + 1])),
                    color, 1, CV_AA);

                cv::line(img,
                    cv::Point2f(s(0, tris[i * 3 + 1]), s(1, tris[i * 3 + 1])),
                    cv::Point2f(s(0, tris[i * 3 + 2]), s(1, tris[i * 3 + 2])),
                    color, 1, CV_AA);

                cv::line(img,
                    cv::Point2f(s(0, tris[i * 3 + 2]), s(1, tris[i * 3 + 2])),
                    cv::Point2f(s(0, tris[i * 3 + 0]), s(1, tris[i * 3 + 0])),
                    color, 1, CV_AA);
            }
        }

        void pawShapeTexture(const cv::Mat & src, cv::Mat & dst, const core::Shape & srcShape, const core::Shape & dstShape, const std::vector<core::Shape::Index>& tris)
        {
            cv::Mat warp(2, 3, CV_32FC1);
            cv::Mat warpImg = cv::Mat::zeros(dst.rows, dst.cols, dst.type());
            cv::Mat warpMask = cv::Mat::zeros(dst.rows, dst.cols, CV_8UC1);

            std::vector<cv::Point2f> sp(3), dp(3);
            std::vector<cv::Point> dpi(3);
            const int shift = 4;
            const float multiplier = (float)(1 << shift);
            for (size_t i = 0; i < tris.size() / 3; ++i) {
                sp[0].x = srcShape(0, tris[i * 3 + 0]);
                sp[0].y = srcShape(1, tris[i * 3 + 0]);
                sp[1].x = srcShape(0, tris[i * 3 + 1]);
                sp[1].y = srcShape(1, tris[i * 3 + 1]);
                sp[2].x = srcShape(0, tris[i * 3 + 2]);
                sp[2].y = srcShape(1, tris[i * 3 + 2]);

                dp[0].x = dstShape(0, tris[i * 3 + 0]);
                dp[0].y = dstShape(1, tris[i * 3 + 0]);
                dp[1].x = dstShape(0, tris[i * 3 + 1]);
                dp[1].y = dstShape(1, tris[i * 3 + 1]);
                dp[2].x = dstShape(0, tris[i * 3 + 2]);
                dp[2].y = dstShape(1, tris[i * 3 + 2]);

                cv::Rect roiSrc = cv::boundingRect(sp);
                cv::Rect roiDst = cv::boundingRect(dp);

                // Relax bounds
                roiSrc.x -= 2; roiSrc.y -= 2;
                roiSrc.width += 4; roiSrc.height += 4;

                roiDst.x -= 2; roiDst.y -= 2;
                roiDst.width += 4; roiDst.height += 4;

                // Correct offsets
                sp[0].x -= roiSrc.x; sp[0].y -= roiSrc.y;
                sp[1].x -= roiSrc.x; sp[1].y -= roiSrc.y;
                sp[2].x -= roiSrc.x; sp[2].y -= roiSrc.y;

                dp[0].x -= roiDst.x; dp[0].y -= roiDst.y;
                dp[1].x -= roiDst.x; dp[1].y -= roiDst.y;
                dp[2].x -= roiDst.x; dp[2].y -= roiDst.y;

                warp = cv::getAffineTransform(sp, dp);

                cv::warpAffine(src(roiSrc), warpImg(roiDst), warp, roiDst.size());

                warpMask.setTo(0);
                cv::Mat warpMaskRoi = warpMask(roiDst);

                dpi[0] = dp[0] * multiplier;
                dpi[1] = dp[1] * multiplier;
                dpi[2] = dp[2] * multiplier;
                cv::fillConvexPoly(warpMaskRoi, &dpi[0], 3, cv::Scalar(255), -1, shift);

                warpImg(roiDst).copyTo(dst(roiDst), warpMaskRoi);
            }
        }
        
    }
}

#endif