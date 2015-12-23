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

int main(int argc, char **argv)
{
    dest::core::InputData inputs;
    dest::face::importIMMFaceDatabase(argv[1], inputs.images, inputs.shapes);

    dest::face::FaceDetector fd;
    if (!fd.loadClassifiers("classifier_frontalface.xml")) {
        std::cout << "Failed to load classifiers." << std::endl;
        return 0;
    }

    inputs.rects.resize(inputs.shapes.size());
    for (size_t i = 0; i < inputs.rects.size(); ++i) {
        if (!fd.detectSingleFace(inputs.images[i], inputs.rects[i])) {
            inputs.rects[i] = dest::core::shapeBounds(inputs.shapes[i]);
        }
    }
    
    dest::core::InputData validation;
    dest::core::InputData::randomPartition(inputs, validation, 0.1f);
    
    dest::core::TrainingData td;
    td.params.numCascades = 10;
    td.params.numTrees = 500;
    td.params.learningRate = 0.1f;
    td.params.maxTreeDepth = 5;
    td.params.exponentialLambda = 0.08f;
    td.input = &inputs;
    dest::core::TrainingData::createTrainingSamplesThroughLinearCombinations(inputs, td.samples, inputs.rnd, 20);

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
