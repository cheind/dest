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

#include <dest/face/database_importers.h>
#include <dest/face/face_detector.h>
#include <dest/util/draw.h>
#include <dest/util/convert.h>
#include <random>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv)
{
    
    dest::core::Tracker t[2];
    if (!t[0].load(argv[1])) {
        std::cout << "Failed to load tier 1tracker." << std::endl;
        return 0;
    }
    
    if (!t[1].load(argv[2])) {
        std::cout << "Failed to load tier 1tracker." << std::endl;
        return 0;
    }
    
    dest::face::FaceDetector fd;
    if (!fd.loadClassifiers("classifier_frontalface.xml")) {
        std::cout << "Failed to load classifiers." << std::endl;
        return 0;
    }
    
    cv::VideoCapture cap;
    
    if (argc > 3) {
        if (isdigit(argv[3][0])) {
            // Open capture device by index
            cap.open(atoi(argv[3]));
        } else {
            // Open video video
            cap.open(argv[3]);
        }
    } else {
        // Open default device;
        cap.open(0);
    }
    
    if (!cap.isOpened()) {
        std::cout << "Failed to open capture device." << std::endl;
        return 0;
    }
    
    cv::Mat imgCV, grayCV;
    dest::core::Image img;
    dest::core::Rect r;
    dest::core::Shape s;
    dest::core::ShapeTransform shapeToImage;
    bool done = false;
    bool detect = false;
    while (!done) {
        cap >> imgCV;
        
        if (imgCV.empty())
            break;
        
        cv::cvtColor(imgCV, grayCV, CV_BGR2GRAY);
        dest::util::toDest(grayCV, img);
        
        if (detect) {
            cv::Rect cvRect;
            if (!fd.detectSingleFace(imgCV, cvRect))
                continue;
            
            dest::util::toDest(cvRect, r);
            shapeToImage = dest::core::estimateSimilarityTransform(dest::core::unitRectangle(), r);
            s = t[0].predict(img, shapeToImage);
            r = dest::core::shapeBounds(s);
            detect = false;
        }
        
        for (int i = 0; i < 5; ++i) {
            shapeToImage = dest::core::estimateSimilarityTransform(dest::core::unitRectangle(), r);
            s = t[1].predict(img, shapeToImage);
            r = dest::core::shapeBounds(s);
        }

        

        //dest::util::drawShape(imgCV, steps[0], cv::Scalar(0,255,0));
        dest::util::drawShape(imgCV, s, cv::Scalar(0,0,255));
        //cv::rectangle(imgCV, cvRect, cv::Scalar(0,255,0));
        
        cv::imshow("track", imgCV);
        int key = cv::waitKey(1);
        if (key == 'x')
            done = true;
        else if (key > 0)
            detect = true;
        
    }

    return 0;
}
