/**
    This file is part of Deformable Shape Tracking (DEST).

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/

#ifndef DEST_FACE_DETECTOR_H
#define DEST_FACE_DETECTOR_H

#include <dest/core/config.h>
#if !defined(DEST_WITH_OPENCV)
#error OpenCV is required for this part of DEST.
#endif

#include <dest/core/shape.h>
#include <dest/core/image.h>
#include <opencv2/core/core.hpp>
#include <string>
#include <memory>

namespace dest {
    namespace face {

        /**
            OpenCV based face detector.

            This face detector can be used to find a coarse bounding box individual faces in images. It's based on
            the Viola Jones algorithm and requires a set of trained classifers that can eihter be found on the 
            OpenCV homepage or in this project's etc directory.

            Usually DEST is trained on results produced by a face detector. The face detector is used as helper tool
            to estimate a coarse global transformation (translation, scaling and rotation (unsupported by this detector)).
            DEST then reports the optimal landmark positions given the initial transformation.
         
            Note that OpenCV 3.0 / CascadeClassifier seems to have troubles when being reused.
            Current solution is to disable OpenCL
            See https://github.com/Itseez/opencv/issues/5475
        */
        class FaceDetector {
        public:
            FaceDetector();
            ~FaceDetector();

            /**
                Load classifier from trained file.

                \param frontalFaceClassifier Serialized classifier to be used to determine frontal faces.
                \param eyeClassifier Optional serialized classifier to be used to validate results from frontal face classifier.
                \returns True on success, false otherwise.
            */
            bool loadClassifiers(const std::string &frontalFaceClassifier, const std::string &eyeClassifier = "");

            /**
                Detect all faces in the given image.

                \param img Image to detect faces in.
                \param faces List of faces detected.
                \returns True on success, false otherwise.
            */
            bool detectFaces(const core::Image &img, std::vector<core::Rect> &faces) const;

            /**
                Detect all faces in the given image.

                \param img Image to detect faces in.
                \param faces List of faces detected.
                \returns True on success, false otherwise.
            */
            bool detectFaces(const cv::Mat &img, std::vector<cv::Rect> &faces) const;

            /**
                Detect the biggest face in the given image.

                \param img Image to detect faces in.
                \param face Detected face
                \returns True on success, false otherwise.
            */
            bool detectSingleFace(const core::Image &img, core::Rect &face) const;

            /**
                Detect the biggest face in the given image.

                \param img Image to detect faces in.
                \param face Detected face
                \returns True on success, false otherwise.
            */
            bool detectSingleFace(const cv::Mat &img, cv::Rect &face) const;

        private:
            struct data;
            std::unique_ptr<data> _data;
        };     
        
    }
}

#endif