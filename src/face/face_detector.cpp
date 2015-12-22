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

#include <dest/face/face_detector.h>
#include <dest/util/convert.h>
#include <opencv2/opencv.hpp>

namespace dest {
    namespace face {
        
        struct FaceDetector::data {
            cv::CascadeClassifier classifierFace;
            cv::CascadeClassifier classifierEyes;
            bool withEyes;
            cv::Mat gray;
        };

        FaceDetector::FaceDetector()
            :_data(new data())
        {}

        FaceDetector::~FaceDetector()
        {}

        bool FaceDetector::loadClassifiers(const std::string &frontalFaceClassifier, const std::string &eyeClassifier)
        {
            FaceDetector::data &data = *_data;

            data.withEyes = !eyeClassifier.empty();

            if (!data.classifierFace.load(frontalFaceClassifier))
                return false;

            if (data.withEyes && !data.classifierEyes.load(eyeClassifier))
                return false;

            return true;
        }

        bool FaceDetector::detectSingleFace(const cv::Mat &img, cv::Rect &face) const
        {
            FaceDetector::data &data = *_data;

            std::vector<cv::Rect> faces;

            if (img.channels() == 3 || img.channels() == 4) {
                cv::cvtColor(img, data.gray, CV_BGR2GRAY);
            }
            else {
                img.copyTo(data.gray);
            }
            cv::equalizeHist(data.gray, data.gray);

            //-- Detect faces
            data.classifierFace.detectMultiScale(data.gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));

            if (faces.empty()) {
                return false;
            }
            else {
                std::sort(faces.begin(), faces.end(), [](const cv::Rect &a, const cv::Rect &b) { return a.area() > b.area(); });
                if (data.withEyes) {
                    cv::Mat roi = data.gray(faces.front());
                    std::vector<cv::Rect> eyes;
                    data.classifierEyes.detectMultiScale(roi, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));
                    if (eyes.size() == 0)
                        return false;
                }              

                face = faces.front();
                return true;
            }
        }

        bool FaceDetector::detectSingleFace(const core::Image &img, core::Rect &face) const
        {
            cv::Mat hdr, u8;

            util::toCVHeaderOnly(img, hdr);
            hdr.convertTo(u8, CV_8U);

            cv::Rect r;
            if (!detectSingleFace(u8, r))
                return false;

            face = core::createRectangle(Eigen::Vector2f(r.tl().x, r.tl().y), Eigen::Vector2f(r.br().x, r.br().y));
            return true;
        }
    }
}