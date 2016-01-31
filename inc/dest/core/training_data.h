/**
    This file is part of Deformable Shape Tracking (DEST).

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
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

            [1] Kazemi, Vahid, and Josephine Sullivan.
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
                Include mean shape as separate sample.
                When set to true, the mean shape will be added once per image
                as sample. Defaults to true. */
            bool includeMeanShape;

            /** 
                To increase the shape space, linear combination of two shapes
                are added to the sample space. The linearWeightRage specifies the
                interval to draw a random weight from assigned to the first shape.
                The second shape selected is weighted by 1.f - weight. Defaults to [0.65, 0.9].
            */
            std::pair<float,float> linearWeightRange;


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
            Shape meanShape;

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
