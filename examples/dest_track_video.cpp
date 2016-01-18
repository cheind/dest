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
        - Use face-detector only in first frame, and a combination of tracker and pseudo face-detector afterwards (fast 4ms in total per frame).

    This application uses OpenCV capture device to open the input device. As such it supports web cams and video files.
    During execution press any key except 'x' to trigger a new face detection.

*/
int main(int argc, char **argv)
{
    
    struct {
        std::string tracker;
        std::string detector;
        std::string device;
        bool detectInEveryFrame;
    } opts;
    
    try {
        TCLAP::CmdLine cmd("Track on video stream.", ' ', "0.9");
        
        TCLAP::ValueArg<std::string> detectorArg("d", "detector", "Detector to provide initial bounds.", true, "classifier.xml", "string", cmd);
        TCLAP::ValueArg<std::string> trackerArg("t", "tracker", "Tracker to align landmarks based initial bounds", true, "dest.bin", "string", cmd);
        TCLAP::UnlabeledValueArg<std::string> device("device", "Device to be opened. Either filename of video or camera device id.", true, "0", "string", cmd);
        TCLAP::SwitchArg detectInEveryFrame("", "always-detect", "Use detector in every frame. If false tries to mimick detector for fast tracking.", cmd, false);
        
        cmd.parse(argc, argv);
        
        opts.tracker = trackerArg.getValue();
        opts.detector = detectorArg.getValue();
        opts.device = device.getValue();
        opts.detectInEveryFrame = detectInEveryFrame.getValue();
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
    // The values below attempt to match tight rects and OpenCV rects.
    float scaleToCV = 1.25f; // Scale between rectangles
    float txToCV = -0.01f; // Translation in x normalized by image width
    float tyToCV = -0.05f; // Translation in y normalized by image height
    
    cv::Mat imgCV, grayCV;
    cv::Rect cvRect;
    dest::core::Image img;
    dest::core::Rect r, r2;
    dest::core::Shape s;
    dest::core::ShapeTransform shapeToImage;
    bool done = false;
    bool detect = false;
    bool detectSuccess = false;
    while (!done) {
        cap >> imgCV;
        
        if (imgCV.empty())
            break;
        
        cv::cvtColor(imgCV, grayCV, CV_BGR2GRAY);
        dest::util::toDest(grayCV, img);
        
        if (detect || opts.detectInEveryFrame) {

            if (fd.detectSingleFace(grayCV, cvRect)) {
                dest::util::toDest(cvRect, r);
                shapeToImage = dest::core::estimateSimilarityTransform(dest::core::unitRectangle(), r);
                s = t.predict(img, shapeToImage);

                detect = false;
                detectSuccess = true;
            } else {
                detectSuccess = false;
            }
        }


        if (!opts.detectInEveryFrame && detectSuccess) {
            for (int i = 0; i < 2; ++i) {
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
        }

    
        dest::util::drawShape(imgCV, s, cv::Scalar(0,0,255));
        dest::util::drawRect(imgCV, r, cv::Scalar(0, 255, 0));
        dest::util::drawRect(imgCV, r2, cv::Scalar(0, 0, 255));
        cv::imshow("DEST Tracking", imgCV);
        int key = cv::waitKey(1);
        if (key == 'x')
            done = true;
        else if (key != -1)
            detect = true;
        
    }

    return 0;
}
