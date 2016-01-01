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
            numTransformPertubationsPerShape = 1;
            useLinearCombinationsOfShapes = true;
            transformRotateRange = std::pair<float, float>(0.f,0.f);
            transformScaleRange = std::pair<float, float>(0.85f, 1.15f);
            transformTranslateRangeX = std::pair<float, float>(-10.f, -10.f);
            transformTranslateRangeY = std::pair<float, float>(-10.f, 10.f);
        }
        
        std::ostream& operator<<(std::ostream &stream, const std::pair<float,float> &obj) {
            stream << std::setw(1) << "[" << obj.first << "," << obj.second << "]";
            return stream;
        }
        
        std::ostream& operator<<(std::ostream &stream, const SampleCreationParameters &obj) {
            stream  << std::setw(40) << std::left << "Number shapes per image" << std::setw(10) << obj.numShapesPerImage << std::endl
                    << std::setw(40) << std::left << "Number of transforms per shape" << std::setw(10) << obj.numTransformPertubationsPerShape << std::endl
                    << std::setw(40) << std::left << "Random rotate angle range" << std::setw(10) << obj.transformRotateRange << std::endl
                    << std::setw(40) << std::left << "Random scale factor range" << std::setw(10) << obj.transformScaleRange << std::endl
                    << std::setw(40) << std::left << "Random translate x range" << std::setw(10) << obj.transformTranslateRangeX << std::endl
                    << std::setw(40) << std::left << "Random translate y range" << std::setw(10) << obj.transformTranslateRangeY << std::endl
                    << std::setw(40) << std::left << "Use linear combination" << std::setw(10) << (obj.useLinearCombinationsOfShapes ? "true" : "false");
            
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
            validatedParams.numTransformPertubationsPerShape = std::max<int>(validatedParams.numTransformPertubationsPerShape, 1);
            
            DEST_LOG("Creating training samples. " << std::endl);
            DEST_LOG(validatedParams << std::endl);
            
            const int numShapes = static_cast<int>(td.input->shapes.size());
            const int numSamples = numShapes * validatedParams.numShapesPerImage * validatedParams.numTransformPertubationsPerShape;
            const int numShapesTimesShapesPerImage = numShapes * validatedParams.numShapesPerImage;
            
            std::uniform_int_distribution<int> dist(0, numShapes - 1);
            std::uniform_real_distribution<float> zeroOne(0, 1);
            std::uniform_real_distribution<float> rotate(validatedParams.transformRotateRange.first, validatedParams.transformRotateRange.second);
            std::uniform_real_distribution<float> scale(validatedParams.transformScaleRange.first, validatedParams.transformScaleRange.second);
            std::uniform_real_distribution<float> tx(validatedParams.transformTranslateRangeX.first, validatedParams.transformTranslateRangeX.second);
            std::uniform_real_distribution<float> ty(validatedParams.transformTranslateRangeY.first, validatedParams.transformTranslateRangeY.second);
            
            td.samples.resize(numSamples);
            for (int i = 0; i < numSamples; ++i) {
                
                td.samples[i].inputIdx = i % numShapes;
                td.samples[i].target = td.input->shapes[td.samples[i].inputIdx];
                
                if (params.useLinearCombinationsOfShapes) {
                    float a = zeroOne(td.input->rnd);
                    float b = zeroOne(td.input->rnd);
                    float c = zeroOne(td.input->rnd);
                    float sum = a + b + c;
                    
                    a /= sum;
                    b /= sum;
                    c /= sum;
                    
                    
                    td.samples[i].estimate = td.input->shapes[dist(td.input->rnd)] * a +
                                             td.input->shapes[dist(td.input->rnd)] * b +
                                             td.input->shapes[dist(td.input->rnd)] * c;
                } else {
                    td.samples[i].estimate = td.input->shapes[dist(td.input->rnd)];
                }
                
                if (i < numShapesTimesShapesPerImage) {
                    td.samples[i].shapeToImage = td.input->shapeToImage[td.samples[i].inputIdx];
                } else {
                    // Note, the following code works well when the shapeToImage transform was generated
                    // with respect to centered unit rectangle. See normalizeShapes and unitRectangle()
                    ShapeTransform trans = td.input->shapeToImage[td.samples[i].inputIdx];
                    
                    ShapeTransform t;
                    t = Eigen::Translation2f(tx(td.input->rnd), ty(td.input->rnd)) *
                        Eigen::Translation2f(trans.translation()) *
                        Eigen::Rotation2Df(rotate(td.input->rnd)) *
                        Eigen::Scaling(scale(td.input->rnd)) *
                        Eigen::Translation2f(-trans.translation()) *
                        trans;
                    
                    td.samples[i].target = t.inverse() * trans * td.samples[i].target.colwise().homogeneous();
                    td.samples[i].shapeToImage = t;
                }
            }
        }
    }
}