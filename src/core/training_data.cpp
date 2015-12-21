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

#include <dest/core/training_data.h>

namespace dest {
    namespace core {
       
        AlgorithmParameters::AlgorithmParameters()
        {
            numInitializationsPerImage = 20;
            numCascades = 10;
            numTrees = 500;
            maxTreeDepth = 5;
            numRandomPixelCoordinates = 400;
            numRandomSplitTestsPerNode = 20;
            exponentialLambda = 0.1f;
            learningRate = 0.1f;
        }

        void TrainingData::createTrainingSamplesKazemi(TrainingData &t)
        {
            int numShapes = static_cast<int>(t.shapes.size());
            int numSamples = numShapes * t.params.numInitializationsPerImage;

            std::uniform_int_distribution<int> dist(0, numShapes - 1);
            t.samples.resize(numSamples);
            for (int i = 0; i < numSamples; ++i) {
                int id = dist(t.rnd);

                t.samples[i].idx = i % numShapes;
                t.samples[i].estimate = t.shapes[id];
            }
        }
       
    }
}