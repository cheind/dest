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
        std::string regressor;
        std::string database;
        std::string rectangles;
    } opts;

    try {
        TCLAP::CmdLine cmd("Evaluate regressor on test database.", ' ', "0.9");
        TCLAP::ValueArg<std::string> regressorArg("r", "regressor", "Trained regressor to load", true, "dest.bin", "string", cmd);
        TCLAP::ValueArg<std::string> rectanglesArg("", "rect", "Initial rectangles to provide to tracker", true, "rectangles.csv", "string", cmd);
        TCLAP::UnlabeledValueArg<std::string> databaseArg("database", "Path to database directory to load", true, "./db", "string", cmd);

        cmd.parse(argc, argv);

        opts.rectangles = rectanglesArg.getValue();
        opts.database = databaseArg.getValue();
        opts.regressor = regressorArg.getValue();
    }
    catch (TCLAP::ArgException &e) {
        std::cerr << "Error: " << e.error() << " for arg " << e.argId() << std::endl;
        return -1;
    }
    
    dest::core::Tracker t;
    if (!t.load(opts.regressor)) {
        std::cerr << "Failed to load tracker." << std::endl;
        return -1;
    }
    
    dest::core::InputData inputs;
    dest::io::DatabaseType dbt = dest::io::importDatabase(opts.database, opts.rectangles, inputs.images, inputs.shapes, inputs.rects);
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

    std::cout << "Average normalized error: " << tr.meanNormalizedDistance << std::endl;
    return 0;
}
