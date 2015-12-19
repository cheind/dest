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
        
        Regressor::~Regressor()
        {}
        
        bool Regressor::fit(RegressorTraining &t)
        {
            Regressor::data &data = *_data;
            data.learningRate = t.learningRate;
            data.trees.resize(t.numTrees);
            data.meanShape = t.meanShape;
            
            TreeTraining tt;
            tt.rnd = t.rnd;
            tt.numLandmarks = t.numLandmarks;
            tt.numSplitPositions = t.numRandomSplitPositions;
            tt.maxDepth = t.maxTreeDepth;
            tt.samples.resize(t.samples.size());
            
            // Draw random samples
            tt.pixelCoordinates = sampleCoordinates(t);
            
            // Encode them with respect to the mean shape
            relativePixelCoordinates(t, tt.pixelCoordinates, data.shapeRelativePixelCoordinates, data.closestShapeLandmark);
            
            // Compute the mean residual, to be used as base learner
            data.meanResidual = ShapeResidual::Zero(2, t.numLandmarks);
            for (size_t i = 0; i < t.samples.size(); ++i) {
                tt.samples[i].residual = t.shapes[t.samples[i].idx] - t.samples[i].estimate;
                data.meanResidual += tt.samples[i].residual;
                
                Eigen::Matrix3f trans = estimateSimilarityTransform(t.meanShape, t.samples[i].estimate);
                readPixelIntensities(trans, t.samples[i].estimate, t.images[t.samples[i].idx], tt.samples[i].intensities);
                
            }
            data.meanResidual /= t.samples.size();

            for (size_t i = 0; i < t.samples.size(); ++i) {
                tt.samples[i].residual -= data.meanResidual;
            }
            
            for (int k = 0; k < t.numTrees; ++k) {
                for (size_t i = 0; i < t.samples.size(); ++i) {
                    
                    if (k > 0) {
                        tt.samples[i].residual -= data.learningRate *  data.trees[k - 1].predict(tt.samples[i].intensities);
                    }
                    
                    data.trees[k].fit(tt);
                }
            }
            
            
            
            return false;
        }
        
        PixelCoordinates Regressor::sampleCoordinates(RegressorTraining &t) const {
            
            Eigen::Vector2f minC = t.meanShape.rowwise().minCoeff();
            Eigen::Vector2f maxC = t.meanShape.rowwise().maxCoeff();

            PixelCoordinates result(2, t.numPixelSamplePositions);
            
            std::uniform_real_distribution<float> dx(maxC.x() - minC.x());
            std::uniform_real_distribution<float> dy(maxC.y() - minC.y());
            
            for (int i = 0; i < t.numPixelSamplePositions; ++i) {
                result(i, 0) = minC.x() + dx(t.rnd);
                result(i, 1) = maxC.y() + dy(t.rnd);
            }
            
            return result;
        }
        
        void Regressor::relativePixelCoordinates(RegressorTraining &t, const PixelCoordinates &pcoords, PixelCoordinates &relcoords, Eigen::VectorXi &closestLandmarks) const
        {
            
            relcoords.resize(pcoords.rows(), pcoords.cols());
            closestLandmarks.resize(pcoords.cols());
            
            for (int i  = 0; i < t.numLandmarks; ++i) {
                int closestLandmark = closestLandmarkIndex(t.meanShape, pcoords.col(i));
                relcoords.col(i) = pcoords.col(i) - t.meanShape.col(closestLandmark);
                closestLandmarks(i) = closestLandmark;
            }
            
        }
        
        int Regressor::closestLandmarkIndex(const Shape &s, const Eigen::Ref<const Eigen::Vector2f> &x) const
        {
            const int numLandmarks = s.cols();
            
            int bestLandmark = -1;
            float bestD2 = std::numeric_limits<float>::max();
            
            for (int i = 0; i < numLandmarks; ++i) {
                float d2 = (s.col(i) - x).squaredNorm();
                if (d2 < bestD2) {
                    bestD2 = d2;
                    bestLandmark = i;
                }
            }
            
            return bestLandmark;
        }
        
        void Regressor::readPixelIntensities(const Eigen::Matrix3f &t, const Shape &s, const Image &img, PixelIntensities &intensities) const
        {
            Regressor::data &data = *_data;
            
            int numCoords = data.shapeRelativePixelCoordinates.cols();
            PixelCoordinates coords = t.block<2,2>(0,0) * data.shapeRelativePixelCoordinates;
            
            for(int i = 0; i < numCoords; ++i) {
                coords.col(i) += s.col(data.closestShapeLandmark[i]);
            }
            
            readImage(img, coords, intensities);
        }
        
        ShapeResidual Regressor::predict(const Image &img, const Shape &shape) const
        {
            Regressor::data &data = *_data;
            
            PixelIntensities intensities;
            Eigen::Matrix3f trans = estimateSimilarityTransform(data.meanShape, shape);
            readPixelIntensities(trans, shape, img, intensities);
            
            const float lr = _data->learningRate;
            const std::vector<Tree> &trees = _data->trees;
            const size_t numTrees = trees.size();
            
            ShapeResidual sr = _data->meanResidual;
            for(size_t i = 0; i < numTrees; ++i) {
                sr += trees[i].predict(intensities) * lr;
            }
            
            return sr;
        }
    }
}