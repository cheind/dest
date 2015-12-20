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

#ifndef DEST_REGRESSOR_H
#define DEST_REGRESSOR_H

#include <dest/core/triplet.h>
#include <dest/core/image.h>
#include <dest/core/shape.h>
#include <memory>
#include <random>
#include <vector>

namespace dest {
    namespace core {
        
        struct RegressorTraining {
            typedef std::vector<Shape> ShapeVector;
            typedef std::vector<Image> ImageVector;
            
            std::mt19937 rnd;
            
            struct Sample {
                int idx;
                Shape estimate;
            };
            typedef std::vector<Sample> SampleVector;
            
            ShapeVector shapes;
            ImageVector images;
            Shape meanShape;
            SampleVector samples;
            
            int numLandmarks;
            int maxTreeDepth;
            int numTrees;
            int numRandomSplitPositions;
            int numPixelSamplePositions;
            float exponentialLambda;
            float learningRate;
        };
    
        class Regressor {
        public:
            Regressor();
            Regressor(const Regressor &other);
            ~Regressor();
            
            bool fit(RegressorTraining &t);
            
            ShapeResidual predict(const Image &img, const Shape &shape) const;
            
        private:
            
            PixelCoordinates sampleCoordinates(RegressorTraining &t) const;
            void relativePixelCoordinates(RegressorTraining &t, const PixelCoordinates &pcoords, PixelCoordinates &relcoords, Eigen::VectorXi &closestLandmarks) const;
            
            int closestLandmarkIndex(const Shape &s, const Eigen::Ref<const Eigen::Vector2f> &x) const;
            
            void readPixelIntensities(const Eigen::AffineCompact2f &t, const Shape &s, const Image &i, PixelIntensities &intensities) const;
            
            struct data;
            std::unique_ptr<data> _data;
        };
        
    }
}

#endif