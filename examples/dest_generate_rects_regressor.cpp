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
#include <tclap/CmdLine.h>

int main(int argc, char **argv)
{
    struct {
        std::string rectIn;
        std::string rectOut;
        std::string regressor;
        std::string db;
    } opts;

    try {
        TCLAP::CmdLine cmd("Generate refined bounding boxes through running a regressor on initial bounding boxes.", ' ', "0.9");
        
        TCLAP::ValueArg<std::string> rectInArg("", "rect-in", "Path to initial rectangles", true, "rectangles.csv", "string", cmd);
        TCLAP::ValueArg<std::string> rectOutArg("", "rect-out", "Path to output rectangles", false, "new-rectangles.csv", "string", cmd);
        TCLAP::ValueArg<std::string> regressorArg("r", "regressor", "Path to regressor to use for alignment", true, "dest.bin", "strinsg", cmd);
        TCLAP::UnlabeledValueArg<std::string> databaseArg("database", "Path to database directory to load", true, "./db", "string", cmd);

        cmd.parse(argc, argv);

        opts.rectIn = rectInArg.getValue();
        opts.rectOut = rectOutArg.getValue();
        opts.regressor = regressorArg.getValue();
        opts.db = databaseArg.getValue();
    }
    catch (TCLAP::ArgException &e) {
        std::cerr << "Error: " << e.error() << " for arg " << e.argId() << std::endl;
        return -1;
    }
   
    dest::core::InputData inputs;
    if (!dest::io::importDatabase(opts.db, opts.rectIn, inputs.images, inputs.shapes, inputs.rects)) {
        std::cerr << "Failed to load database" << std::endl;
        return -1;
    }
    
    dest::core::Tracker t;
    if (!t.load(opts.regressor)) {
        std::cerr << "Failed to load tracker." << std::endl;
        return -1;
    }
    
    std::vector<dest::core::Rect> rects(inputs.rects.size());
    for (size_t i = 0; i < rects.size(); ++i) {
        dest::core::ShapeTransform trans = dest::core::estimateSimilarityTransform(dest::core::unitRectangle(), inputs.rects[i]);
        dest::core::Shape s = t.predict(inputs.images[i], trans);
        rects[i] = dest::core::shapeBounds(s);
        
        if (i % 100 == 0)
            std::cout << "Processing " << i << "/" << rects.size() << " elements.\r" << std::flush;
    }
    
    std::cout << "Saving new rectangles to " << opts.rectOut << std::endl;
    dest::io::exportRectangles(opts.rectOut, rects);

    return 0;
}
