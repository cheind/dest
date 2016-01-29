/**
    This file is part of Deformable Shape Tracking (DEST).

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/

#include <dest/core/training_data.h>
#include <iomanip>
#include <dest/util/log.h>

namespace dest {
    namespace core {
       
        TrainingParameters::TrainingParameters()
        {
            numCascades = 10;
            numTrees = 500;
            maxTreeDepth = 5;
            numRandomPixelCoordinates = 400;
            numRandomSplitTestsPerNode = 20;
            exponentialLambda = 0.1f;
            learningRate = 0.05f;
            expansionRandomPixelCoordinates = 0.05f;
        }
        
        std::ostream& operator<<(std::ostream &stream, const TrainingParameters &obj) {
            stream << std::setw(30) << std::left << "Number of cascades" << std::setw(10) << obj.numCascades << std::endl
                   << std::setw(30) << std::left << "Number of trees" << std::setw(10) << obj.numTrees << std::endl
                   << std::setw(30) << std::left << "Maximum tree depth" << std::setw(10) << obj.maxTreeDepth << std::endl
                   << std::setw(30) << std::left << "Random pixel locations" << std::setw(10) << obj.numRandomPixelCoordinates << std::endl
                   << std::setw(30) << std::left << "Random split tests" << std::setw(10) << obj.numRandomSplitTestsPerNode << std::endl
                   << std::setw(30) << std::left << "Random pixel expansion" << std::setw(10) << obj.expansionRandomPixelCoordinates << std::endl
                   << std::setw(30) << std::left << "Exponential lambda" << std::setw(10) << obj.exponentialLambda << std::endl
                   << std::setw(30) << std::left << "Learning rate" << std::setw(10) << obj.learningRate;
            return stream;
        }
        
        SampleCreationParameters::SampleCreationParameters()
        {
            numShapesPerImage = 20;
            linearWeight = 0.8f;
            includeMeanShape = true;
        }
        
        std::ostream& operator<<(std::ostream &stream, const std::pair<float,float> &obj) {
            stream << std::setw(1) << "[" << obj.first << "," << obj.second << "]";
            return stream;
        }
        
        std::ostream& operator<<(std::ostream &stream, const SampleCreationParameters &obj) {
            stream  << std::setw(30) << std::left << "Number shapes per image" << std::setw(10) << obj.numShapesPerImage << std::endl
                    << std::setw(30) << std::left << "Linear weight" << std::setw(10) << obj.linearWeight << std::endl
                    << std::setw(40) << std::left << "Include mean shape" << std::setw(10) << (obj.includeMeanShape ? "true" : "false");
            
            return stream;
        }
        
        struct Generator {
            Generator() : m_value(0) { }
            int operator()() { return m_value++; }
            int m_value;
        };

        void InputData::normalizeShapes(InputData & input)
        {
            const int numShapes = static_cast<int>(input.shapes.size());

            input.shapeToImage.resize(numShapes);
            for (size_t i = 0; i < numShapes; ++i) {
                ShapeTransform t = estimateSimilarityTransform(input.rects[i], unitRectangle());
                input.shapes[i] = t * input.shapes[i].colwise().homogeneous();
                input.shapeToImage[i] = t.inverse();
            }
        }
        
        void InputData::computeMeanShape(InputData &input)
        {
            const int numShapes = static_cast<int>(input.shapes.size());
            const int numLandmarks = static_cast<int>(input.shapes.front().cols());
            
            input.meanShape = Shape::Zero(2, numLandmarks);
            for (int i = 0; i < numShapes; ++i) {
                input.meanShape += input.shapes[i];
            }
            input.meanShape /= static_cast<float>(numShapes);
        }
        
        void InputData::randomPartition(InputData &train, InputData &validate, float validatePercent)
        {
            int numValidate = static_cast<int>((float)train.shapes.size() * validatePercent);
            
            std::vector<int> ids(train.shapes.size());
            std::generate(ids.begin(), ids.end(), Generator());
            std::shuffle(ids.begin(), ids.end(), train.rnd);
            
            validate.shapes.clear();
            validate.shapeToImage.clear();
            validate.images.clear();
            validate.rects.clear();
            
            for (size_t i = 0; i < numValidate; ++i) {
                validate.shapes.push_back(train.shapes[ids[i]]);
                validate.shapeToImage.push_back(train.shapeToImage[ids[i]]);
                validate.images.push_back(train.images[ids[i]]);
                validate.rects.push_back(train.rects[ids[i]]);
            }
            
            InputData train2;
            for (size_t i = numValidate; i < ids.size(); ++i)
            {
                train2.shapes.push_back(train.shapes[ids[i]]);
                train2.shapeToImage.push_back(train.shapeToImage[ids[i]]);
                train2.images.push_back(train.images[ids[i]]);
                train2.rects.push_back(train.rects[ids[i]]);
            }
            
            std::swap(train2, train);
        }

        SampleData::SampleData(InputData &input_)
        : input(&input_)
        {}
        
        void SampleData::createTestingSamples(SampleData &td) {
            const int numSamples = static_cast<int>(td.input->shapes.size());
            td.samples.resize(numSamples);
            
            for (int i = 0; i < numSamples; ++i) {
                td.samples[i].inputIdx = i;
                td.samples[i].target = td.input->shapes[i];
                td.samples[i].shapeToImage = td.input->shapeToImage[i];
                // Note, estimate is not set by this method as it is not used during testing.
            }
        }
        
        void SampleData::createTrainingSamples(SampleData &td, const SampleCreationParameters &params) {
            
            SampleCreationParameters validatedParams = params;
            validatedParams.numShapesPerImage = std::max<int>(validatedParams.numShapesPerImage, 1);
            validatedParams.linearWeight = std::max<float>(0.f, std::min<float>(1.f, params.linearWeight));
            
            DEST_LOG("Creating training samples. " << std::endl);
            DEST_LOG(validatedParams << std::endl);
            
            const int numShapes = static_cast<int>(td.input->shapes.size());
            int numSamples = numShapes * validatedParams.numShapesPerImage;
            
            std::uniform_int_distribution<int> dist(0, numShapes - 1);
            
            td.samples.resize(numSamples);
            for (int i = 0; i < numSamples; ++i) {
                
                if (i < numShapes && validatedParams.includeMeanShape) {
                    td.samples[i].inputIdx = i;
                    td.samples[i].target = td.input->shapes[i];
                    td.samples[i].shapeToImage = td.input->shapeToImage[i];
                    td.samples[i].estimate = td.input->meanShape;
                } else {
                    int idx = i % numShapes;
                    td.samples[i].inputIdx = idx;
                    td.samples[i].target = td.input->shapes[idx];
                    td.samples[i].shapeToImage = td.input->shapeToImage[idx];
                    
                    td.samples[i].estimate = td.input->shapes[dist(td.input->rnd)] * validatedParams.linearWeight +
                                             td.input->shapes[dist(td.input->rnd)] * (1.f - validatedParams.linearWeight);

                }
            }
        }
    }
}