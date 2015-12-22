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
    dest::core::TrainingData td;
    td.params.numCascades = 10;
    td.params.numTrees = 500;
    td.params.learningRate = 0.1f;
    td.params.maxTreeDepth = 3;

    dest::face::importIMMFaceDatabase(argv[1], td.images, td.shapes);

    dest::face::FaceDetector fd;
    if (!fd.loadClassifiers("classifier_frontalface.xml")) {
        std::cout << "Failed to load classifiers." << std::endl;
        return 0;
    }

    std::vector<size_t> removeIds;
    td.rects.resize(td.shapes.size());
    for (size_t i = 0; i < td.rects.size(); ++i) {
        if (!fd.detectSingleFace(td.images[i], td.rects[i])) {
            removeIds.push_back(i);
        }
    }

    for (size_t i = 0; i < removeIds.size(); ++i) {
        td.rects[removeIds[i]] = td.rects.back(); td.rects.pop_back();
        td.shapes[removeIds[i]] = td.shapes.back(); td.shapes.pop_back();
        td.images[removeIds[i]] = td.images.back(); td.images.pop_back();
    }

    
    dest::core::TrainingData::convertShapesToNormalizedShapeSpace(td.rects, td.shapes);
    dest::core::TrainingData::createTrainingSamplesThroughLinearCombinations(td.shapes, td.trainSamples, td.rnd, 20);

    dest::core::TrainingData::SampleVector validate;
    dest::core::TrainingData::randomPartitionTrainingSamples(td.trainSamples, validate, td.rnd, 0.1f);


    dest::core::Tracker t;
    t.fit(td);
    t.save("dest_tracker_imm.bin");

    for (size_t i = 0; i < validate.size(); ++i) {
        dest::core::Shape s = t.predict(td.images[validate[i].idx], td.rects[validate[i].idx]);
        cv::Mat tmp = dest::util::drawShape(td.images[validate[i].idx], s, cv::Scalar(0, 255, 0));
        cv::imshow("result", tmp);
        cv::waitKey();
    }


    return 0;
}
