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
        
        /** 
            Training parameters. 

            For more info consult the original paper.

            [1] Kazemi, Vahdat, and Josephine Sullivan.
                "One millisecond face alignment with an ensemble of regression trees."
                Computer Vision and Pattern Recognition (CVPR), 2014 IEEE Conference on. IEEE, 2014.
        */
        struct TrainingParameters {
            /** Number of incremental cascades. Defaults to 10. */
            int numCascades;

            /** Number of trees per cascade. Defaults to 500. */
            int numTrees;

            /** Maximum depth of each tree. Defaults to 5 (including root level) */
            int maxTreeDepth;

            /** Number pixel coordinates to randomly generate per cascade. Defaults to 400.*/
            int numRandomPixelCoordinates;

            /** Number of split tests to evaluate to find the best possible split. Defaults to 20. */
            int numRandomSplitTestsPerNode;

            /** 
                Exponential lambda. Determines to which extend closer pixel coordinates are preferred. 
                Measured in normalized shape space units. Larger values tend to allow bigger distances.
                Defaults to 0.1 
            */
            float exponentialLambda;

            /** Shrinks the contribution of each tree by a factor of learningRate. Defaults to 0.1.*/
            float learningRate;

            /** Offset to allow sampling of pixel coordinates outside of mean shape.
                Measured in normalized shape space units.
                Defaults to 0.05
            */
            float expansionRandomPixelCoordinates;
            
            TrainingParameters();
        };
        
        /**
            Inspect training parameters.
        */
        std::ostream& operator<<(std::ostream &stream, const TrainingParameters &obj);
        
        /**
            Necessary input data derive generate training samples from.

            Use dest::importDatabase to load from various existing shape databases.
        */
        struct InputData {
            typedef std::vector<Rect> RectVector;
            typedef std::vector<ShapeTransform> ShapeTransformVector;
            typedef std::vector<Shape> ShapeVector;
            typedef std::vector<Image> ImageVector;
            
            /** 
                Initial rectangles for each shape. 
                Used for computing shape normalizing transforms. 
            */
            RectVector rects;

            /**
                A list of training shapes.
            */
            ShapeVector shapes;

            /**
                A list of training images. 
            */
            ImageVector images;

            /**
                A list of inverse shape normalizing transforms. 
                Use normalizeShapes to fill with defaults based on rectangles and unit rectangles.
            */
            ShapeTransformVector shapeToImage;

            /**
                Random number generator used during training.
            */
            std::mt19937 rnd;
            
            /**
                Automatically partition input into a training and validation set.

                \param input Input data to be split. Result will contain the training set.
                \param validate Validation set
                \param validatePercent Percentile input data to be used for validation.
            */
            static void randomPartition(InputData &input, InputData &validate, float validatePercent = 0.05f);

            /**
                Normalize shapes.

                Used the corresponding rectangle and dest::unitRectangle() to find a shape normalizing transform.
                Transforms shape and stores inverse transformation in shapeToImage.
            */
            static void normalizeShapes(InputData &input);
        };
        
        /**
            Parameters to control training sample creation from input data.
        */
        struct SampleCreationParameters {

            /** Number of shapes to generate per image. Defaults to 20. */
            int numShapesPerImage;

            /** 
                Number of rectangle perturbations to generate per shape. 
                Defaults to 1. 
                This is a rather experimental feature and at this point not enough experiments
                have been carried out to validate its usefulness.
            */            
            int numTransformPertubationsPerShape;

            /** Whether or not to extend shape space by linearly combining training shapes. Defaults to true. */
            bool useLinearCombinationsOfShapes;

            /** Scale perturbation range. Only applicable if numTransformPertubationsPerShape > 1. */
            std::pair<float, float> transformScaleRange;

            /** 
                Translate in x perturbation range. 
                Only applicable if numTransformPertubationsPerShape > 1. 
                Measured in image dimensions. Set to empty range to prevent random perturbations of this category.
            */
            std::pair<float, float> transformTranslateRangeX;

            /** 
                Translate in y perturbation range. 
                Only applicable if numTransformPertubationsPerShape > 1. 
                Measured in image dimensions. Set to empty range to prevent random perturbations of this category.
            */
            std::pair<float, float> transformTranslateRangeY;

            /** 
                Rotation range range. 
                Only applicable if numTransformPertubationsPerShape > 1. 
                Measured in angles of radians.
            */
            std::pair<float, float> transformRotateRange;
            
            SampleCreationParameters();
        };

        /**
            Inspect sample creation parameters.
        */
        std::ostream& operator<<(std::ostream &stream, const SampleCreationParameters &obj);
        
        /**
            Generated training samples.
        */
        struct SampleData {
            
            /** 
                A training sample.
            */
            struct Sample {
                int inputIdx;
                Shape estimate;
                Shape target;
                ShapeTransform shapeToImage;
            };
            typedef std::vector<Sample> SampleVector;
            
            SampleData(InputData &input);

            InputData *input;
            SampleVector samples;
            TrainingParameters params;

            /**
                Create training samples.
            */
            static void createTrainingSamples(SampleData &td, const SampleCreationParameters &params);

            /**
                Create standard test samples from validation set. 

                Treats each input data a single training sample.
            */
            static void createTestingSamples(SampleData &td);
        };
        
        /**
            Input data for regressor training.
        */
        struct RegressorTraining {
            
            InputData *input;
            SampleData *training;
            Shape meanShape;            
            int numLandmarks;
        };
        
        /**
            Input data for tree training.
        */
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
            SampleData *training;
            SampleVector samples;
            PixelCoordinates pixelCoordinates;
            int numLandmarks;
        };
    }
}

#endif