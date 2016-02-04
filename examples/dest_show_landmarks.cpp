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
    Show landmarks on faces.
*/
int main(int argc, char **argv)
{
    struct {
        std::string database;
        int loadMaxSize;
        int loadMinSize;
    } opts;

    try {
        TCLAP::CmdLine cmd("Evaluate regressor on test database.", ' ', "0.9");
        TCLAP::ValueArg<int> maxImageSizeArg("", "load-max-size", "Maximum size of images in the database", false, 2048, "int", cmd);
        TCLAP::ValueArg<int> minImageSizeArg("", "load-min-size", "Minimum size of images in the database", false, 640, "int", cmd);
        TCLAP::UnlabeledValueArg<std::string> databaseArg("database", "Path to database directory to load", true, "./db", "string", cmd);
        

        cmd.parse(argc, argv);

        opts.database = databaseArg.getValue();
        opts.loadMaxSize = maxImageSizeArg.getValue();
        opts.loadMinSize = minImageSizeArg.getValue();
    }
    catch (TCLAP::ArgException &e) {
        std::cerr << "Error: " << e.error() << " for arg " << e.argId() << std::endl;
        return -1;
    }
    
    dest::io::ShapeDatabase sd;
    sd.setMaxImageLoadSize(opts.loadMaxSize);
    sd.setMinImageLoadSize(opts.loadMinSize);
    sd.enableMirroring(true);
    sd.setMaxElementsToLoad(10);

    dest::core::InputData inputs;    
    if (!sd.load(opts.database, inputs.images, inputs.shapes, inputs.rects)) {
        std::cerr << "Failed to load database." << std::endl;
        return -1;
    }

    size_t i = 0;
    bool done = false;
    while (i < inputs.images.size() && !done) {

        cv::Mat tmp = dest::util::drawShape(inputs.images[i], inputs.shapes[i], cv::Scalar(255, 255, 255));
        dest::util::drawShapeText(tmp, inputs.shapes[i], cv::Scalar(0, 0, 255));

        cv::imshow("Inputs - Press ESC to skip", tmp);
        if (cv::waitKey() == 27)
            done = true;

        ++i;
    }


    return 0;
}
