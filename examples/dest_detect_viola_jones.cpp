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

#include <dest/io/database_io.h>
#include <dest/face/face_detector.h>
#include <dest/io/rect_io.h>
#include <dest/util/draw.h>
#include <dest/util/convert.h>
#include <iostream>
#include <opencv2/opencv.hpp>

#include <tclap/CmdLine.h>

float ratioRectShapeOverlap(const dest::core::Rect &r, const dest::core::Shape &s) {
    Eigen::Vector2f minC = r.col(0);
    Eigen::Vector2f maxC = r.col(3);
    
    int numOverlap = 0;
    for (dest::core::Shape::Index i = 0; i < s.cols(); ++i) {
        Eigen::Vector2f a = s.col(i) - minC;
        Eigen::Vector2f b = s.col(i) - maxC;
        
        if ((a.array() >= 0.f).all() && (b.array() <= 0.f).all())
            numOverlap += 1;
    }
    
    return (float)numOverlap / (float)s.cols();
}

int main(int argc, char **argv)
{
    struct {
        std::vector<std::string> detectors;
        std::string db;
        std::string output;
    } opts;

    try {
        TCLAP::CmdLine cmd("Generate initial bounding boxes for face detection using Viola-Jones algorithm in OpenCV.", ' ', "0.9");
        TCLAP::MultiArg<std::string> detectorsArg("d", "detector", "OpenCV classifier to load", true, "string");
        TCLAP::ValueArg<std::string> outputArg("o", "output", "CSV output file", false, "rectangles.csv", "string");
        TCLAP::UnlabeledValueArg<std::string> databaseArg("database", "Path to database directory to load", true, "./db", "string");

        cmd.add(&detectorsArg);
        cmd.add(&outputArg);
        cmd.add(&databaseArg);

        cmd.parse(argc, argv);

        opts.detectors.insert(opts.detectors.end(), detectorsArg.begin(), detectorsArg.end());
        opts.db = databaseArg.getValue();
        opts.output = outputArg.getValue();
    }
    catch (TCLAP::ArgException &e) {
        std::cerr << "Error: " << e.error() << " for arg " << e.argId() << std::endl;
        return -1;
    }
   
    dest::core::InputData inputs;
    std::vector<dest::core::Rect> rects;
    if (!dest::io::importDatabase(opts.db, "", inputs.images, inputs.shapes, rects)) {
        std::cout << "Failed to load database" << std::endl;
        return -1;
    }

    std::vector<dest::face::FaceDetector> detectors(opts.detectors.size());
    for (size_t i = 0; i < opts.detectors.size(); ++i) {
        if (!detectors[i].loadClassifiers(opts.detectors[i])) {
            std::cout << "Failed to load detector " << opts.detectors[i];
            return -1;
        }
    }

    size_t countDetectionSuccess = 0;
    for (size_t i = 0; i < rects.size(); ++i) {

        std::vector<dest::core::Rect> faces;
        for (size_t j = 0; j < detectors.size(); ++j) {
            std::vector<dest::core::Rect> rects;
            detectors[j].detectFaces(inputs.images[i], rects);
            faces.insert(faces.end(), rects.begin(), rects.end());
        }
               
        // Find the face rect with a meaningful shape overlap
        float bestOverlap = 0.f;
        size_t bestId = std::numeric_limits<size_t>::max();
        
        for (size_t j = 0; j < faces.size(); ++j) {
            float o = ratioRectShapeOverlap(faces[j], inputs.shapes[i]);
            if (o > bestOverlap) {
                bestId = j;
                bestOverlap = o;
            }
        }
        
        if ((bestId == std::numeric_limits<size_t>::max()) || bestOverlap < 0.6f) {
            rects[i] = dest::core::shapeBounds(inputs.shapes[i]);
        } else {
            ++countDetectionSuccess;
            rects[i] = faces[bestId];
        }
        
        if (i % 10 == 0) {
            std::cout << "Processing " << i << "\r" << std::flush;
        }
    }

    std::cout << "Detector successful on " << countDetectionSuccess << "/" << rects.size() << " shapes." << std::endl;
    
    dest::io::exportRectangles(opts.output, rects);

    return 0;
}
