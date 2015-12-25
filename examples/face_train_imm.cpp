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
#include <iostream>
#include <opencv2/opencv.hpp>

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
    dest::core::InputData inputs;
    dest::face::importIBugW300FaceDatabase(argv[1], inputs.images, inputs.shapes);

    dest::face::FaceDetector fd;
    if (!fd.loadClassifiers("classifier_frontalface.xml")) {
        std::cout << "Failed to load classifiers." << std::endl;
        return 0;
    }

    inputs.rects.resize(inputs.shapes.size());
    for (size_t i = 0; i < inputs.rects.size(); ++i) {
        std::vector<dest::core::Rect> faces;
        fd.detectFaces(inputs.images[i], faces);
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
            inputs.rects[i] = dest::core::shapeBounds(inputs.shapes[i]);
        } else {
            inputs.rects[i] = faces[bestId];
        }
        
        /*
        cv::Mat tmp = dest::util::drawShape(inputs.images[i], inputs.shapes[i], cv::Scalar(0,255,0));
        cv::Rect_<float> cr;
        dest::util::toCV(inputs.rects[i], cr);
        cv::rectangle(tmp, cr, cv::Scalar(0,255,0));
        
        cv::imshow("img", tmp);
        cv::waitKey();
        */
    }
    
    dest::core::InputData validation;
    dest::core::InputData::randomPartition(inputs, validation, 0.1f);
    
    dest::core::TrainingData td;
    td.params.numCascades = 10;
    td.params.numTrees = 500;
    td.params.learningRate = 0.1f;
    td.params.maxTreeDepth = 4;
    td.params.exponentialLambda = 0.1f;
    td.input = &inputs;
    dest::core::TrainingData::createTrainingSamplesThroughLinearCombinations(inputs, td.samples, inputs.rnd, 200);

    dest::core::Tracker t;
    t.fit(td);
    t.save("dest_tracker_imm.bin");
    
    dest::core::TrainingData::SampleVector validationSamples;
    dest::core::TrainingData::createTrainingSamplesThroughLinearCombinations(validation, validationSamples, inputs.rnd, 1);
    for (size_t i = 0; i < validationSamples.size(); ++i) {
        dest::core::TrainingData::Sample &s = validationSamples[i];
        
        dest::core::Shape shape = t.predict(validation.images[s.inputIdx], s.targetRectInImageSpace);
        cv::Mat tmp = dest::util::drawShape(validation.images[s.inputIdx], shape, cv::Scalar(0, 255, 0));
        cv::imshow("result", tmp);
        cv::waitKey();
    }


    return 0;
}
