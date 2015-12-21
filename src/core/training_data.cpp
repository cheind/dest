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
#include <set>

namespace dest {
    namespace core {
       
        AlgorithmParameters::AlgorithmParameters()
        {
            numCascades = 10;
            numTrees = 500;
            maxTreeDepth = 5;
            numRandomPixelCoordinates = 400;
            numRandomSplitTestsPerNode = 20;
            exponentialLambda = 0.1f;
            learningRate = 0.1f;
        }

        struct ImageShapeId {
            int shapeId;
            int imageId;

            bool operator<(const ImageShapeId &other) const {
                if (shapeId != other.shapeId) {

                }
            }
        };

        void TrainingData::createTrainingSamplesKazemi(TrainingData &t, int numInitializationsPerImage, float validationPercent)
        {
            const int numShapes = static_cast<int>(t.shapes.size());
            const int numSamples = numShapes * numInitializationsPerImage;
            const int numValidationSamples = static_cast<int>(numShapes * validationPercent);

            std::uniform_int_distribution<int> dist(0, numShapes - 1);
            std::set< std::pair<int, int> > validationSet;

            t.validationSamples.resize(numValidationSamples);
            for (int i = 0; i < numValidationSamples; ++i) {
                int imageid;
                int shapeid;
                std::pair<int, int> isPair;
                do {
                    isPair.first = dist(t.rnd);
                    isPair.second = dist(t.rnd);
                } while (validationSet.find(isPair) != validationSet.end());
                validationSet.insert(isPair);

                t.validationSamples[i].idx = isPair.first;
                t.validationSamples[i].estimate = t.shapes[isPair.second];
            }

            t.trainSamples.resize(numSamples);
            for (int i = 0; i < numSamples; ++i) {
                std::pair<int, int> isPair;
                do {
                    isPair.first = i % numShapes;
                    isPair.second = dist(t.rnd);
                } while (validationSet.find(isPair) != validationSet.end());

                t.trainSamples[i].idx = isPair.first;
                t.trainSamples[i].estimate = t.shapes[isPair.second];
            }
        }
       
    }
}