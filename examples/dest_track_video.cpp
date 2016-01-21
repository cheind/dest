/**
This file is part of Deformable Shape Tracking (DEST).

Copyright(C) 2015/2016 Christoph Heindl
All rights reserved.

This software may be modified and distributed under the terms
of the BSD license.See the LICENSE file for details.
*/

#include <dest/dest.h>
#include <opencv2/opencv.hpp>
#include <tclap/CmdLine.h>

#include <dest/face/face_detector.h>
#include <dest/util/draw.h>
#include <dest/util/convert.h>
#include <random>


/**
    Track on video sequence.

    This tool supports two operation modes. 
        - Use face-detector then tracker on every frame (accurate but slow as face detector is the slowest component, 60ms in total per frame).
        - Use face-detector only every n-th frame. 
          In between detector frames, a combination of tracker and mock face-detector (fast 4ms in total per frame) is used.

    This application uses OpenCV capture device to open the input device. As such it supports web cams and video files.
    During execution press any key except 'x' to trigger a new face detection.

*/
int main(int argc, char **argv)
{
    
    struct {
        std::string tracker;
        std::string detector;
        std::string device;
        int detectRate;
        bool drawRect;
    } opts;
    
    try {
        TCLAP::CmdLine cmd("Track on video stream.", ' ', "0.9");
        
        TCLAP::ValueArg<std::string> detectorArg("d", "detector", "Detector to provide initial bounds.", true, "classifier.xml", "XML file", cmd);
        TCLAP::ValueArg<std::string> trackerArg("t", "tracker", "Tracker to align landmarks based initial bounds", true, "dest.bin", "Tracker file", cmd);
        TCLAP::UnlabeledValueArg<std::string> deviceArg("device", "Device to be opened. Either filename of video or camera device id.", true, "0", "string", cmd);
        TCLAP::SwitchArg drawRectArg("", "draw-rect", "Draw face detector rectangle", cmd, false);
        TCLAP::ValueArg<int> detectInNthFrameArg("", "detect-rate", "Use detector in every n-th frame. If false tries to mimick detector for fast tracking.", false, 5, "int", cmd);
        
        cmd.parse(argc, argv);
        
        opts.tracker = trackerArg.getValue();
        opts.detector = detectorArg.getValue();
        opts.device = deviceArg.getValue();
        opts.detectRate = detectInNthFrameArg.getValue();
        opts.drawRect = drawRectArg.getValue();
    }
    catch (TCLAP::ArgException &e) {
        std::cerr << "Error: " << e.error() << " for arg " << e.argId() << std::endl;
        return -1;
    }
    
    dest::core::Tracker t;
    if (!t.load(opts.tracker)) {
        std::cerr << "Failed to load tracker." << std::endl;
        return -1;
    }
      
    dest::face::FaceDetector fd;
    if (!fd.loadClassifiers(opts.detector)) {
        std::cerr << "Failed to load classifiers." << std::endl;
        return -1;
    }
    
    cv::VideoCapture cap;
    
    if (opts.device.size() == 1 && isdigit(opts.device[0])) {
        // Open capture device by index
        cap.open(atoi(opts.device.c_str()));
    } else {
        // Open video video
        cap.open(opts.device.c_str());
    }
    
    if (!cap.isOpened()) {
        std::cerr << "Failed to open capture device." << std::endl;
        return -1;
    }

    // The OpenCV detector rectangles are significantly different from tight bounds.
    // The values below attempt to match tight rects from landmarks and OpenCV rects.
    float scaleToCV = 1.25f; // Scale between rectangles
    float txToCV = -0.01f; // Translation in x normalized by image width
    float tyToCV = -0.05f; // Translation in y normalized by image height
    
    cv::Mat imgCV, grayCV;
    cv::Rect cvRect;
    dest::core::Image img;
    dest::core::Rect r;
    dest::core::Shape s;
    dest::core::ShapeTransform shapeToImage;
    bool done = false;
    bool requestDetect = false;
    bool detectSuccess = false;
    int frameCount = 0;
    while (!done) {
        cap >> imgCV;
        
        if (imgCV.empty())
            break;
        
        cv::cvtColor(imgCV, grayCV, CV_BGR2GRAY);
        dest::util::toDest(grayCV, img);
        
        const bool isDetectFrame = (frameCount % opts.detectRate == 0);

        if (requestDetect || isDetectFrame) {

            if (fd.detectSingleFace(grayCV, cvRect)) {
                dest::util::toDest(cvRect, r);
                shapeToImage = dest::core::estimateSimilarityTransform(dest::core::unitRectangle(), r);
                s = t.predict(img, shapeToImage);

                requestDetect = false;
                detectSuccess = true;
            } else {
                detectSuccess = false;
            }
        }


        if (!isDetectFrame && detectSuccess) {
            // Mimick detector behaviour. Only works for OpenCV face detectors.
            r = dest::core::shapeBounds(s);
            shapeToImage = dest::core::estimateSimilarityTransform(dest::core::unitRectangle(), r);
            Eigen::AffineCompact2f tr;
            tr.setIdentity();
            tr = Eigen::Translation2f(txToCV * img.cols(), tyToCV * img.rows()) *
                Eigen::Translation2f(shapeToImage.translation()) *
                Eigen::Scaling(scaleToCV) *
                Eigen::Translation2f(-shapeToImage.translation());
            r = tr * r.colwise().homogeneous();

            shapeToImage = dest::core::estimateSimilarityTransform(dest::core::unitRectangle(), r);
            s = t.predict(img, shapeToImage);
        }

        dest::util::drawShape(imgCV, s, cv::Scalar(255, 0, 102));
       
        if (opts.drawRect)
            dest::util::drawRect(imgCV, r, cv::Scalar(0, 255, 0));

        cv::imshow("DEST Tracking", imgCV);
        int key = cv::waitKey(1);
        if (key == 'x')
            done = true;
        else if (key != -1)
            requestDetect = true;

        ++frameCount;
        
    }

    return 0;
}
