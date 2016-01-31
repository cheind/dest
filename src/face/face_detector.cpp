/**
    This file is part of Deformable Shape Tracking (DEST).

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/


#include <dest/core/config.h>
#ifdef DEST_WITH_OPENCV

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
        {
        }

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
        
        

        
        bool FaceDetector::detectFaces(const cv::Mat &img, std::vector<cv::Rect> &faces) const {
            
            FaceDetector::data &data = *_data;
            
            if (img.channels() == 3 || img.channels() == 4) {
                cv::cvtColor(img, data.gray, CV_BGR2GRAY);
            }
            else {
                img.copyTo(data.gray);
            }
            cv::equalizeHist(data.gray, data.gray);
            
            //-- Detect faces
            data.classifierFace.detectMultiScale(data.gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(10, 10));
            if (data.withEyes) {
                    std::vector<cv::Rect> finalFaces;
                    for (size_t i = 0; i < faces.size(); ++i) {
                        cv::Mat roi = data.gray(faces[i]);
                        std::vector<cv::Rect> eyes;
                        data.classifierEyes.detectMultiScale(roi, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(10, 10));
                        if (eyes.size() > 0)
                            finalFaces.push_back(faces[i]);
                    }
                    finalFaces.swap(faces);
                }
            return faces.size() > 0;
        }
        
        bool FaceDetector::detectFaces(const core::Image &img, std::vector<core::Rect> &faces) const {
            
            
            cv::Mat hdr;
            
            util::toCVHeaderOnly(img, hdr);
            
            std::vector<cv::Rect> cvFaces;
            if (!detectFaces(hdr, cvFaces)) {
                return  false;
            }
            
            faces.clear();
            core::Rect face;
            for (size_t i = 0; i < cvFaces.size(); ++i) {
                cv::Rect r = cvFaces[i];
                util::toDest(r, face);
                faces.push_back(face);
            }
            
            return true;
        }
        


        bool FaceDetector::detectSingleFace(const cv::Mat &img, cv::Rect &face) const
        {
            std::vector<cv::Rect> faces;
            if (!detectFaces(img, faces))
                return false;
            
            std::sort(faces.begin(), faces.end(), [](const cv::Rect &a, const cv::Rect &b) {return a.area() > b.area();});
            
            face = faces.front();
            return true;
            
        }

        bool FaceDetector::detectSingleFace(const core::Image &img, core::Rect &face) const
        {
            cv::Rect r;
            cv::Mat tmp;
            util::toCV(img, tmp);
            
            if (!detectSingleFace(tmp, r))
                return false;
            
            util::toDest(r, face);
            return true;
        }
    }
}

#endif