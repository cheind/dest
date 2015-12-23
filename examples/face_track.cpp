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

    cv::Mat imgCV = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
    dest::core::Image img;
    dest::util::toDest(imgCV, img);

    dest::face::FaceDetector fd;
    if (!fd.loadClassifiers("classifier_frontalface.xml")) {
        std::cout << "Failed to load classifiers." << std::endl;
        return 0;
    }
    
    dest::core::Rect r;
    if (!fd.detectSingleFace(img, r)) {
        std::cout << "Failed to detect face" << std::endl;
        return 0;
    }

    
    dest::core::Tracker t;
    if (!t.load(argv[1])) {
        std::cout << "Failed to load tracker." << std::endl;
        return 0;
    }

    std::vector<dest::core::Shape> steps;
    dest::core::Shape s = t.predict(img, r, &steps);

    bool done = false;
    int id = 0;
    while (!done) {
        
        cv::Mat tmp = dest::util::drawShape(img, steps[id], cv::Scalar(0, 255, 0));
        cv::rectangle(tmp, cv::Rect2f(r(0, 0), r(1, 0), r(0, 3) - r(0, 0), r(1, 3) - r(1, 0)), cv::Scalar(255, 0, 0));
        cv::imshow("prediction", tmp);

        id = (id + 1) % steps.size();

        int key = cv::waitKey();
        if (key == 'x')
            done = true;
    }


    
    return 0;
}
