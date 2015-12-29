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
    
    dest::core::Tracker t;
    if (!t.load(argv[1])) {
        std::cout << "Failed to load tracker." << std::endl;
        return 0;
    }
    
    dest::face::FaceDetector fd;
    if (!fd.loadClassifiers("classifier_frontalface.xml")) {
        std::cout << "Failed to load classifiers." << std::endl;
        return 0;
    }
    
    cv::VideoCapture cap;
    
    if (argc > 2) {
        if (isdigit(argv[2][0])) {
            // Open capture device by index
            cap.open(atoi(argv[2]));
        } else {
            // Open video video
            cap.open(argv[2]);
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
    bool done = false;
    while (!done) {
        cap >> imgCV;
        
        if (imgCV.empty())
            break;
        
        //cv::resize(imgCV, imgCV, cv::Size(320, 240));
        cv::cvtColor(imgCV, grayCV, CV_BGR2GRAY);
        
        cv::Rect cvRect;
        if (!fd.detectSingleFace(imgCV, cvRect))
            continue;
        
        dest::core::Rect r;
        dest::util::toDest(grayCV, img);
        dest::util::toDest(cvRect, r);
        
        dest::core::ShapeTransform shapeToImage = dest::core::estimateSimilarityTransform(dest::core::unitRectangle(), r);
        dest::core::Shape s = t.predict(img, shapeToImage);

        //dest::util::drawShape(imgCV, steps[0], cv::Scalar(0,255,0));
        dest::util::drawShape(imgCV, s, cv::Scalar(0,0,255));
        //cv::rectangle(imgCV, cvRect, cv::Scalar(0,255,0));
        
        cv::imshow("track", imgCV);
        int key = cv::waitKey(1);
        if (key == 'x')
            done = true;
        
    }

    return 0;
}
