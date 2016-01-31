/**
    This file is part of Deformable Shape Tracking (DEST).

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/

#include <dest/dest.h>
#include <tclap/CmdLine.h>
#include <opencv2/opencv.hpp>

/**
    Evaluate a trained tracker based on some test data.

    This methods loads a database of test samples and evaluates the tracker. Deviations 
    from the true shape are normalized by the inter-ocular distance.

*/
int main(int argc, char **argv)
{
    struct {
        std::string tracker;
        std::string database;
        std::string rectangles;
        dest::io::ImportParameters importParams;
    } opts;

    try {
        TCLAP::CmdLine cmd("Evaluate regressor on test database.", ' ', "0.9");
        TCLAP::ValueArg<std::string> trackerArg("t", "tracker", "Trained tracker to load", true, "dest.bin", "file", cmd);
        TCLAP::ValueArg<std::string> rectanglesArg("r", "rectangles", "Initial rectangles to provide to tracker", false, "rectangles.csv", "file", cmd);
        TCLAP::ValueArg<int> maxImageSizeArg("", "load-max-size", "Maximum size of images in the database", false, 2048, "int", cmd);
        TCLAP::UnlabeledValueArg<std::string> databaseArg("database", "Path to database directory to load", true, "./db", "string", cmd);
        

        cmd.parse(argc, argv);

        opts.rectangles = rectanglesArg.isSet() ? rectanglesArg.getValue() : "";
        opts.database = databaseArg.getValue();
        opts.tracker = trackerArg.getValue();
        opts.importParams.maxImageSideLength = maxImageSizeArg.getValue();
    }
    catch (TCLAP::ArgException &e) {
        std::cerr << "Error: " << e.error() << " for arg " << e.argId() << std::endl;
        return -1;
    }
    
    dest::core::Tracker t;
    if (!t.load(opts.tracker)) {
        std::cerr << "Failed to load tracker." << std::endl;
        return -1;
    }
    
    dest::core::InputData inputs;
    dest::io::DatabaseType dbt = dest::io::importDatabase(opts.database, opts.rectangles, inputs.images, inputs.shapes, inputs.rects, opts.importParams);
    if (dbt == dest::io::DATABASE_ERROR) {
        std::cerr << "Failed to load database." << std::endl;
        return -1;
    }
    
    dest::core::InputData::normalizeShapes(inputs);
    dest::core::SampleData td(inputs);
    dest::core::SampleData::createTestingSamples(td);
    
    dest::core::LandmarkDistanceNormalizer ldn;
    switch (dbt) {
        case dest::io::DATABASE_IMM:
            ldn = dest::core::LandmarkDistanceNormalizer::createInterocularNormalizerIMM();
            break;
        case dest::io::DATABASE_IBUG:
            ldn = dest::core::LandmarkDistanceNormalizer::createInterocularNormalizerIBug();
            break;
        default:
            std::cerr << "Unknown database type" << std::endl;
            return -1;
    }
    
    dest::core::TestResult tr = dest::core::testTracker(td, t, ldn);

    std::cout << std::setw(40) << std::left << "Average normalized error: " << tr.meanNormalizedDistance << std::endl;
    std::cout << std::setw(40) << std::left << "Stddev normalized error: " << tr.stddevNormalizedDistance << std::endl;
    std::cout << std::setw(40) << std::left << "Median normalized error: " << tr.medianNormalizedDistance << std::endl;
    std::cout << std::setw(40) << std::left << "Worst normalized error: " << tr.worstNormalizedDistance << std::endl;
    return 0;
}
