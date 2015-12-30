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

#include <dest/face/face_detector.h>
#include <dest/util/draw.h>
#include <dest/util/convert.h>
#include <random>
#include <opencv2/opencv.hpp>
#include <tclap/CmdLine.h>

int main(int argc, char **argv)
{
    struct {
        std::string detector;
        std::string regressor;
        std::string image;
    } opts;

    try {
        TCLAP::CmdLine cmd("Generate initial bounding boxes for face detection using Viola-Jones algorithm in OpenCV.", ' ', "0.9");
        TCLAP::ValueArg<std::string> detectorArg("d", "detector", "OpenCV face detector to load", true, "cascade.xml", "string");
        TCLAP::ValueArg<std::string> regressorArg("r", "regressor", "Trained regressor to load", true, "dest.bin", "string");
        TCLAP::UnlabeledValueArg<std::string> imageArg("image", "Image to align", true, "img.png", "string");

        cmd.add(&detectorArg);
        cmd.add(&regressorArg);
        cmd.add(&imageArg);

        cmd.parse(argc, argv);

        opts.detector = detectorArg.getValue();
        opts.regressor = regressorArg.getValue();
        opts.image = imageArg.getValue();
    }
    catch (TCLAP::ArgException &e) {
        std::cerr << "Error: " << e.error() << " for arg " << e.argId() << std::endl;
        return -1;
    }

    cv::Mat imgCV = cv::imread(opts.image, cv::IMREAD_GRAYSCALE);
    dest::core::Image img;
    dest::util::toDest(imgCV, img);

    dest::face::FaceDetector fd;
    if (!fd.loadClassifiers(opts.detector)) {
        std::cout << "Failed to load classifiers." << std::endl;
        return 0;
    }
    
    dest::core::Tracker t;
    if (!t.load(opts.regressor)) {
        std::cout << "Failed to load tracker." << std::endl;
        return 0;
    }

    dest::core::Rect r;
    if (!fd.detectSingleFace(img, r)) {
        std::cout << "Failed to detect face" << std::endl;
        return 0;
    }

    dest::core::ShapeTransform shapeToImage = dest::core::estimateSimilarityTransform(dest::core::unitRectangle(), r);

    std::vector<dest::core::Shape> steps;
    dest::core::Shape s = t.predict(img, shapeToImage, &steps);

    bool done = false;
    int id = 0;
    while (!done) {
        
        cv::Scalar color = (id == 0) ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 255, 255);
        cv::Mat tmp = dest::util::drawShape(img, steps[id], color);
        cv::rectangle(tmp, cv::Rect_<float>(r(0, 0), r(1, 0), r(0, 3) - r(0, 0), r(1, 3) - r(1, 0)), color);
        cv::imshow("prediction", tmp);

        id = (id + 1) % steps.size();

        int key = cv::waitKey();
        if (key == 'x')
            done = true;
    }


    
    return 0;
}
