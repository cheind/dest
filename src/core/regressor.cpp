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

#include <dest/core/regressor.h>
#include <dest/core/tree.h>
#include <dest/util/log.h>

namespace dest {
    namespace core {
        
        struct Regressor::data {
            
            PixelCoordinates shapeRelativePixelCoordinates;
            Eigen::VectorXi closestShapeLandmark;
            
            ShapeResidual meanResidual;
            Shape meanShape;
            std::vector<Tree> trees;
            float learningRate;
            
            data()
            {
            }
        };
        
        Regressor::Regressor()
        : _data(new data())
        {
        }
        
        Regressor::Regressor(const Regressor &other)
        :_data(new data(*other._data))
        {}
        
        Regressor::~Regressor()
        {}
        
        bool Regressor::fit(RegressorTraining &t)
        {
            Regressor::data &data = *_data;
            data.learningRate = t.trainingData->params.learningRate;
            data.trees.resize(t.trainingData->params.numTrees);
            data.meanShape = t.meanShape;
            
            TreeTraining tt;
            tt.numLandmarks = t.numLandmarks;
            tt.trainingData = t.trainingData;
            tt.samples.resize(t.samples.size());
            
            // Draw random samples
            tt.pixelCoordinates = sampleCoordinates(t);
            
            // Encode them with respect to the mean shape
            shapeRelativePixelCoordinates(t.meanShape, tt.pixelCoordinates, data.shapeRelativePixelCoordinates, data.closestShapeLandmark);
            
            // Compute the mean residual, to be used as base learner
            data.meanResidual = ShapeResidual::Zero(2, t.numLandmarks);
            for (size_t i = 0; i < t.samples.size(); ++i) {

                tt.samples[i].residual = t.trainingData->shapes[t.samples[i].idx] - t.samples[i].estimate;
                data.meanResidual += tt.samples[i].residual;
                
                Eigen::AffineCompact2f trans = estimateSimilarityTransform(t.meanShape, t.samples[i].estimate);
                readPixelIntensities(trans, t.samples[i].estimate, t.trainingData->images[t.samples[i].idx], tt.samples[i].intensities);
                
            }
            data.meanResidual /= static_cast<float>(t.samples.size());
            
            for (int k = 0; k < t.trainingData->params.numTrees; ++k) {
                DEST_LOG("Building tree " << std::setw(3) << k << "\r");
                for (size_t i = 0; i < t.samples.size(); ++i) {
                    
                    if (k == 0) {
                        tt.samples[i].residual -= data.meanResidual;
                    } else {
                        tt.samples[i].residual -= data.learningRate * data.trees[k - 1].predict(tt.samples[i].intensities);
                    }
                }
                data.trees[k].fit(tt);
            }
            
            
            
            return false;
        }
        
        PixelCoordinates Regressor::sampleCoordinates(RegressorTraining &t) const {
            
            Eigen::Vector2f minC = t.meanShape.rowwise().minCoeff();
            Eigen::Vector2f maxC = t.meanShape.rowwise().maxCoeff();

            const int numCoords = t.trainingData->params.numRandomPixelCoordinates;
            PixelCoordinates result(2, numCoords);
            
            std::uniform_real_distribution<float> dx(0.f, maxC.x() - minC.x());
            std::uniform_real_distribution<float> dy(0.f, maxC.y() - minC.y());
            
            for (int i = 0; i < numCoords; ++i) {
                result(0, i) = minC.x() + dx(t.trainingData->rnd);
                result(1, i) = maxC.y() + dy(t.trainingData->rnd);
            }
            
            return result;
        }
        
        
        void Regressor::readPixelIntensities(const Eigen::AffineCompact2f &t, const Shape &s, const Image &img, PixelIntensities &intensities) const
        {
            Regressor::data &data = *_data;
            
            PixelCoordinates coords = t.matrix().block<2,2>(0,0) * data.shapeRelativePixelCoordinates;
            
            const Shape::Index numCoords = data.shapeRelativePixelCoordinates.cols();
            for(Shape::Index i = 0; i < numCoords; ++i) {
                coords.col(i) += s.col(data.closestShapeLandmark(i));
            }
            
            readImage(img, coords, intensities);
        }
        
        ShapeResidual Regressor::predict(const Image &img, const Shape &shape) const
        {
            Regressor::data &data = *_data;
            
            PixelIntensities intensities;
            Eigen::AffineCompact2f trans = estimateSimilarityTransform(data.meanShape, shape);
            readPixelIntensities(trans, shape, img, intensities);
            
            const size_t numTrees = data.trees.size();
            
            ShapeResidual sr = data.meanResidual;
            for(size_t i = 0; i < numTrees; ++i) {
                sr += data.trees[i].predict(intensities) * data.learningRate;
            }
            
            return sr;
        }
    }
}