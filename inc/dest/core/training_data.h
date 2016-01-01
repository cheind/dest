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
#include <iosfwd>

namespace dest {
    namespace core {
        
        struct TrainingParameters {
            int numCascades;
            int numTrees;
            int maxTreeDepth;
            int numRandomPixelCoordinates;
            int numRandomSplitTestsPerNode;
            float exponentialLambda;
            float learningRate;
            
            TrainingParameters();
        };
        
        std::ostream& operator<<(std::ostream &stream, const TrainingParameters &obj);
        
        struct InputData {
            typedef std::vector<Rect> RectVector;
            typedef std::vector<ShapeTransform> ShapeTransformVector;
            typedef std::vector<Shape> ShapeVector;
            typedef std::vector<Image> ImageVector;
            
            RectVector rects;
            ShapeVector shapes;
            ImageVector images;
            ShapeTransformVector shapeToImage;
            std::mt19937 rnd;
            
            static void randomPartition(InputData &input, InputData &validate, float validatePercent = 0.1f);
            static void normalizeShapes(InputData &input);
        };
        
        struct SampleCreationParameters {
            int numShapesPerImage;
            int numTransformPertubationsPerShape;
            bool useLinearCombinationsOfShapes;
            std::pair<float, float> transformScaleRange;
            std::pair<float, float> transformTranslateRangeX;
            std::pair<float, float> transformTranslateRangeY;
            std::pair<float, float> transformRotateRange;
            
            SampleCreationParameters();
        };
        std::ostream& operator<<(std::ostream &stream, const SampleCreationParameters &obj);
        
        struct TrainingData {
            
            struct Sample {
                int inputIdx;
                Shape estimate;
                Shape target;
                ShapeTransform shapeToImage;
            };
            typedef std::vector<Sample> SampleVector;
            
            TrainingData(InputData &input);

            InputData *input;
            SampleVector samples;
            TrainingParameters params;

            static void createTrainingSamples(TrainingData &td, const SampleCreationParameters &params);
            static void createTestingSamples(TrainingData &td);
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