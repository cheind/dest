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
            numCascades = 10;
            numTrees = 500;
            maxTreeDepth = 5;
            numRandomPixelCoordinates = 400;
            numRandomSplitTestsPerNode = 20;
            exponentialLambda = 0.1f;
            learningRate = 0.1f;
        }

        void TrainingData::createTrainingSamplesKazemi(const ShapeVector &shapes, SampleVector &samples, std::mt19937 &rnd, int numInitializationsPerImage)
        {
            const int numShapes = static_cast<int>(shapes.size());
            const int numSamples = numShapes * numInitializationsPerImage;

            std::uniform_int_distribution<int> dist(0, numShapes - 1);
            samples.resize(numSamples);
            for (int i = 0; i < numSamples; ++i) {
                samples[i].idx = i % numShapes;
                samples[i].estimate = shapes[dist(rnd)];
            }
        }

        void TrainingData::createTrainingSamplesThroughLinearCombinations(const ShapeVector &shapes, SampleVector &samples, std::mt19937 &rnd, int numInitializationsPerImage)
        {
            const int numShapes = static_cast<int>(shapes.size());
            const int numSamples = numShapes * numInitializationsPerImage;

            std::uniform_int_distribution<int> dist(0, numShapes - 1);
            std::uniform_real_distribution<float> zeroOne(0, 1);

            samples.resize(numSamples);
            for (int i = 0; i < numSamples; ++i) {
                float a = zeroOne(rnd);
                float b = zeroOne(rnd);
                float c = zeroOne(rnd);
                float sum = a + b + c;

                a /= sum;
                b /= sum;
                c /= sum;

                samples[i].idx = i % numShapes;
                samples[i].estimate = shapes[dist(rnd)] * a + shapes[dist(rnd)] * b + shapes[dist(rnd)] * c;
            }
        }


        void TrainingData::convertShapesToNormalizedShapeSpace(const RectVector &rects, ShapeVector &shapes)
        {
            const int numShapes = static_cast<int>(shapes.size());
            for (int i = 0; i < numShapes; ++i) {
                Eigen::AffineCompact2f t = estimateSimilarityTransform(rects[i], unitRectangle());
                shapes[i] = (t * shapes[i].colwise().homogeneous()).eval();
            }
        }

        void TrainingData::createTrainingRectsFromShapeBounds(const ShapeVector &shapes, RectVector &rects)
        {
            const int numShapes = static_cast<int>(shapes.size());
            rects.resize(numShapes);

            for (int i = 0; i < numShapes; ++i)
            {
                rects[i] = shapeBounds(shapes[i]);
            }
        }

        struct Generator {
            Generator() : m_value(0) { }
            int operator()() { return m_value++; }
            int m_value;
        };

        void TrainingData::randomPartitionTrainingSamples(SampleVector &train, SampleVector &validate, std::mt19937 &rnd, float validatePercent)
        {
            int numValidate = static_cast<int>((float)train.size() * validatePercent);

            std::vector<int> ids(train.size());
            std::generate(ids.begin(), ids.end(), Generator());
            std::shuffle(ids.begin(), ids.end(), rnd);
            
            validate.clear();
            
            for (size_t i = 0; i < numValidate; ++i) {
                validate.push_back(train[ids[i]]);                
            }

            SampleVector train2;
            for (size_t i = numValidate; i < ids.size(); ++i)
            {                
                train2.push_back(train[ids[i]]);
            }
            
            std::swap(train2, train);
        }
       
    }
}