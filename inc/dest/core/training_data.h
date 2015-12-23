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

#ifndef DEST_TRAINING_DATA_H
#define DEST_TRAINING_DATA_H

#include <dest/core/shape.h>
#include <dest/core/image.h>
#include <vector>
#include <random>

namespace dest {
    namespace core {
        
        struct AlgorithmParameters {
            int numCascades;
            int numTrees;
            int maxTreeDepth;
            int numRandomPixelCoordinates;
            int numRandomSplitTestsPerNode;
            float exponentialLambda;
            float learningRate;
            
            AlgorithmParameters();
        };
        
        struct InputData {
            typedef std::vector<Rect> RectVector;
            typedef std::vector<Shape> ShapeVector;
            typedef std::vector<Image> ImageVector;
            
            RectVector rects;
            ShapeVector shapes;
            ImageVector images;
            std::mt19937 rnd;
            
            static void randomPartition(InputData &train, InputData &validate, float validatePercent = 0.1f);
        };
        
        struct TrainingData {
            
            struct Sample {
                int inputIdx;
                Shape estimateInNormalizedSpace;
                Shape targetInNormalizedSpace;
                Rect targetRectInImageSpace;
            };
            typedef std::vector<Sample> SampleVector;

            InputData *input;
            SampleVector samples;
            AlgorithmParameters params;

            static void createTrainingSamplesKazemi(const InputData &input, SampleVector &samples, std::mt19937 &rnd, int numInitializationsPerImage = 20);
            static void createTrainingSamplesThroughLinearCombinations(const InputData &input, SampleVector &samples, std::mt19937 &rnd, int numInitializationsPerImage = 20);
            static void convertShapesToNormalizedShapeSpace(SampleVector &samples);
        };
        
        struct RegressorTraining {
            InputData *input;
            TrainingData *training;
            Shape meanShape;            
            int numLandmarks;
        };
        
        struct TreeTraining {
            struct Sample {
                ShapeResidual residual;
                PixelIntensities intensities;
                
                friend inline void swap(Sample& a, Sample& b)
                {
                    using std::swap;
                    swap(a.residual, b.residual);
                    swap(a.intensities, b.intensities);
                }
            };
            typedef std::vector<Sample> SampleVector;
            
            InputData *input;
            TrainingData *training;
            SampleVector samples;
            PixelCoordinates pixelCoordinates;
            int numLandmarks;
        };
    }
}

#endif