/**
This file is part of Deformable Shape Tracking (DEST).

Copyright(C) 2015/2016 Christoph Heindl
All rights reserved.

This software may be modified and distributed under the terms
of the BSD license.See the LICENSE file for details.
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
        dest::core::TrainingParameters trainingParams;
        dest::core::SampleCreationParameters createParams;
        int loadMaxSize;
        bool mirror;
        std::string db;
        std::string rects;
        std::string output;
        int randomSeed;
        bool showInitialSamples;
    } opts;

    try {
        TCLAP::CmdLine cmd("Train cascade of regressors using a landmark database and initial rectangles.", ' ', "0.9");
        
        TCLAP::ValueArg<int> numCascadesArg("", "train-num-cascades", "Number of cascades to train.", false, 10, "int", cmd);
        TCLAP::ValueArg<int> numTreesArg("", "train-num-trees", "Number of trees per cascade.", false, 500, "int", cmd);
        TCLAP::ValueArg<int> maxTreeDepthArg("", "train-max-depth", "Maximum tree depth.", false, 5, "int", cmd);
        TCLAP::ValueArg<int> numPixelsArg("", "train-num-pixels", "Number of random pixel coordinates", false, 400, "int", cmd);
        TCLAP::ValueArg<int> numSplitTestsArg("", "train-num-splits", "Number of random split tests at each tree node", false, 20, "int", cmd);
        TCLAP::ValueArg<int> randomSeedArg("", "train-rnd-seed", "Seed for the random number generator", false, 10, "int", cmd);
        TCLAP::ValueArg<float> lambdaArg("", "train-lambda", "Prior that favors closer pixel coordinates.", false, 0.1f, "float", cmd);
        TCLAP::ValueArg<float> learnArg("", "train-learn", "Learning rate of each tree.", false, 0.08f, "float", cmd);
        
        TCLAP::ValueArg<int> numShapesPerImageArg("", "create-num-shapes", "Number of shapes per image to create.", false, 20, "int", cmd);
        
        TCLAP::SwitchArg showInitialSamplesArg("", "show-samples", "Show generated samples", cmd, false);
        TCLAP::ValueArg<std::string> rectsArg("", "rectangles", "Initial detection rectangles to train on.", false, "rectangles.csv", "string", cmd);
        TCLAP::ValueArg<std::string> outputArg("o", "output", "Trained regressor output.", false, "dest.bin", "string", cmd);
        TCLAP::ValueArg<int> maxImageSizeArg("", "load-max-size", "Maximum size of images in the database", false, 2048, "int", cmd);
        TCLAP::SwitchArg mirrorImageArg("", "load-mirrored", "Additionally mirror each database image, shape and rects.", cmd, false);
        TCLAP::UnlabeledValueArg<std::string> databaseArg("database", "Path to database directory to load", true, "./db", "string", cmd);


        cmd.parse(argc, argv);
        
        opts.createParams.numShapesPerImage = numShapesPerImageArg.getValue();

        opts.trainingParams.numCascades = numCascadesArg.getValue();
        opts.trainingParams.numTrees = numTreesArg.getValue();
        opts.trainingParams.maxTreeDepth = maxTreeDepthArg.getValue();
        opts.trainingParams.numRandomPixelCoordinates = numPixelsArg.getValue();
        opts.trainingParams.numRandomSplitTestsPerNode = numSplitTestsArg.getValue();
        opts.trainingParams.exponentialLambda = lambdaArg.getValue();
        opts.trainingParams.learningRate = learnArg.getValue();
        opts.randomSeed = randomSeedArg.getValue();
        
        opts.loadMaxSize = maxImageSizeArg.getValue();
        opts.mirror = mirrorImageArg.getValue();
        
        opts.showInitialSamples = showInitialSamplesArg.getValue();
        opts.db = databaseArg.getValue();
        opts.rects = rectsArg.isSet() ? rectsArg.getValue() : "";
        opts.output = outputArg.getValue();        
    }
    catch (TCLAP::ArgException &e) {
        std::cerr << "Error: " << e.error() << " for arg " << e.argId() << std::endl;
        return -1;
    }

    std::vector<dest::core::Rect> rects;
    if (!opts.rects.empty() && !dest::io::importRectangles(opts.rects, rects)) {
        std::cerr << "Failed to load rectangles." << std::endl;
        return -1;
    }

    dest::io::ShapeDatabase sd;
    sd.setMaxImageLoadSize(opts.loadMaxSize);
    sd.enableMirroring(opts.mirror);
    sd.setRectangles(rects);

    dest::core::InputData inputs;
    if (!sd.load(opts.db, inputs.images, inputs.shapes, inputs.rects)) {
        std::cerr << "Failed to load database." << std::endl;
        return -1;
    }

    dest::core::InputData::normalizeShapes(inputs);
    
    dest::core::SampleData td(inputs);
    td.params = opts.trainingParams;
    dest::core::SampleData::createTrainingSamples(td, opts.createParams);
    
    if (opts.showInitialSamples) {
        size_t i = 0;
        bool done = false;
        while (i < td.samples.size() && !done) {
            dest::core::SampleData::Sample &s = td.samples[i];
            
            cv::Mat tmp = dest::util::drawShape(td.input->images[s.inputIdx], s.shapeToImage * s.estimate.colwise().homogeneous(), cv::Scalar(0, 255, 0));
            dest::core::Rect r = s.shapeToImage * dest::core::unitRectangle().colwise().homogeneous();
            dest::core::Shape target = s.shapeToImage * s.target.colwise().homogeneous();
            dest::util::drawShape(tmp, target, cv::Scalar(255,255,255));
            dest::util::drawRect(tmp, r, cv::Scalar(0,255,0));
            
            cv::imshow("Samples - Press ESC to skip", tmp);
            if (cv::waitKey() == 27)
                done = true;
            ++i;
        }
    }


    dest::core::Tracker t;
    t.fit(td);
    
    std::cout << "Saving tracker to " << opts.output << std::endl;
    t.save(opts.output);

    return 0;
}
