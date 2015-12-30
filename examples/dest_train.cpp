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
#include <dest/util/draw.h>
#include <dest/util/convert.h>
#include <iostream>
#include <opencv2/opencv.hpp>

#include <tclap/CmdLine.h>

int main(int argc, char **argv)
{
    struct {
        dest::io::ImportParameters importParams;
        std::string db;
        std::string rects;
        std::string output;
    } opts;

    try {
        TCLAP::CmdLine cmd("Train cascade of regressors using a landmark database and initial rectangles.", ' ', "0.9");
        TCLAP::ValueArg<std::string> rectsArg("r", "rectangles", "Initial detection rectangles to train on.", true, "rectangles.csv", "string");
        TCLAP::ValueArg<std::string> outputArg("o", "output", "Trained cascade of regressors file.", false, "dest.bin", "string");
        TCLAP::ValueArg<int> maxImageSize("", "max-image-size", "Maximum size of images in the database", false, 640, "int");
        TCLAP::UnlabeledValueArg<std::string> databaseArg("database", "Path to database directory to load", true, "./db", "string");

        cmd.add(&rectsArg);
        cmd.add(&outputArg);
        cmd.add(&maxImageSize);
        cmd.add(&databaseArg);

        cmd.parse(argc, argv);

        opts.importParams.maxImageSideLength = maxImageSize.getValue();
        opts.db = databaseArg.getValue();
        opts.rects = rectsArg.getValue();
        opts.output = outputArg.getValue();        
    }
    catch (TCLAP::ArgException &e) {
        std::cerr << "Error: " << e.error() << " for arg " << e.argId() << std::endl;
        return -1;
    }

    dest::core::InputData inputs;
    std::vector<dest::core::Rect> rects;
    if (!dest::io::importDatabase(opts.db, opts.rects, inputs.images, inputs.shapes, rects, opts.importParams)) {
        std::cerr << "Failed to load database." << std::endl;
        return -1;
    }

    dest::core::InputData::normalizeShapes(inputs, rects);    
    dest::core::InputData validation;
    dest::core::InputData::randomPartition(inputs, validation, 0.01f);
    
    dest::core::TrainingData td;
    td.params.numCascades = 10;
    td.params.numTrees = 50;
    td.params.learningRate = 0.1f;
    td.params.maxTreeDepth = 5;
    td.params.exponentialLambda = 0.1f;
    td.input = &inputs;
    dest::core::TrainingData::createTrainingSamplesThroughLinearCombinations(inputs, td.samples, inputs.rnd, 200);

    dest::core::Tracker t;
    t.fit(td);
    t.save(opts.output);
    
    dest::core::TrainingData::SampleVector validationSamples;
    dest::core::TrainingData::createTrainingSamplesThroughLinearCombinations(validation, validationSamples, inputs.rnd, 1);
    for (size_t i = 0; i < validationSamples.size(); ++i) {
        dest::core::TrainingData::Sample &s = validationSamples[i];
        
        dest::core::Shape shape = t.predict(validation.images[s.inputIdx], s.shapeToImage);
        cv::Mat tmp = dest::util::drawShape(validation.images[s.inputIdx], shape, cv::Scalar(0, 255, 0));
        cv::imshow("result", tmp);
        cv::waitKey();
    }


    return 0;
}
