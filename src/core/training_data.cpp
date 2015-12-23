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
            learningRate = 0.05f;
        }
        
        struct Generator {
            Generator() : m_value(0) { }
            int operator()() { return m_value++; }
            int m_value;
        };
        
        void InputData::randomPartition(InputData &train, InputData &validate, float validatePercent)
        {
            int numValidate = static_cast<int>((float)train.shapes.size() * validatePercent);
            
            std::vector<int> ids(train.shapes.size());
            std::generate(ids.begin(), ids.end(), Generator());
            std::shuffle(ids.begin(), ids.end(), train.rnd);
            
            validate.shapes.clear();
            validate.rects.clear();
            validate.images.clear();
            
            for (size_t i = 0; i < numValidate; ++i) {
                validate.shapes.push_back(train.shapes[ids[i]]);
                validate.rects.push_back(train.rects[ids[i]]);
                validate.images.push_back(train.images[ids[i]]);
            }
            
            InputData train2;
            for (size_t i = numValidate; i < ids.size(); ++i)
            {
                train2.shapes.push_back(train.shapes[ids[i]]);
                train2.rects.push_back(train.rects[ids[i]]);
                train2.images.push_back(train.images[ids[i]]);
            }
            
            std::swap(train2, train);
        }

        
        void TrainingData::createTrainingSamplesKazemi(const InputData &input, SampleVector &samples, std::mt19937 &rnd, int numInitializationsPerImage) {
            const int numShapes = static_cast<int>(input.shapes.size());
            const int numSamples = numShapes * numInitializationsPerImage;
            
            InputData::ShapeVector nshapes(numShapes);
            for (size_t i = 0; i < nshapes.size(); ++i) {
                Eigen::AffineCompact2f t = estimateSimilarityTransform(input.rects[i], unitRectangle());
                nshapes[i] = t * input.shapes[i].colwise().homogeneous();
            }
            
            std::uniform_int_distribution<int> dist(0, numShapes - 1);
            samples.resize(numSamples);
            for (int i = 0; i < numSamples; ++i) {
                samples[i].inputIdx = i % numShapes;
                samples[i].estimateInNormalizedSpace = nshapes[dist(rnd)];
                samples[i].targetInNormalizedSpace = nshapes[samples[i].inputIdx];
                samples[i].targetRectInImageSpace = input.rects[samples[i].inputIdx];
            }

        }
        
        
        void TrainingData::createTrainingSamplesThroughLinearCombinations(const InputData &input, SampleVector &samples, std::mt19937 &rnd, int numInitializationsPerImage) {
            const int numShapes = static_cast<int>(input.shapes.size());
            const int numSamples = numShapes * numInitializationsPerImage;
            
            std::uniform_int_distribution<int> dist(0, numShapes - 1);
            std::uniform_real_distribution<float> zeroOne(0, 1);
            
            InputData::ShapeVector nshapes(numShapes);
            for (size_t i = 0; i < nshapes.size(); ++i) {
                Eigen::AffineCompact2f t = estimateSimilarityTransform(input.rects[i], unitRectangle());
                nshapes[i] = t * input.shapes[i].colwise().homogeneous();
            }
            
            samples.resize(numSamples);
            for (int i = 0; i < numSamples; ++i) {
                float a = zeroOne(rnd);
                float b = zeroOne(rnd);
                float c = zeroOne(rnd);
                float sum = a + b + c;
                
                a /= sum;
                b /= sum;
                c /= sum;
                
                samples[i].inputIdx = i % numShapes;
                samples[i].estimateInNormalizedSpace = nshapes[dist(rnd)] * a + nshapes[dist(rnd)] * b + nshapes[dist(rnd)] * c;
                samples[i].targetInNormalizedSpace = nshapes[samples[i].inputIdx];
                samples[i].targetRectInImageSpace = input.rects[samples[i].inputIdx];
            }
        }
    }
}