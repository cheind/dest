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

#ifndef DEST_SHAPE_H
#define DEST_SHAPE_H

#include <Eigen/Geometry>
#include <dest/core/image.h>

namespace dest {
    namespace core {
    
        typedef Eigen::Matrix<float, 2, 4, Eigen::DontAlign> Rect;
        typedef Eigen::Matrix<float, 2, Eigen::Dynamic> Shape;
        typedef Shape ShapeResidual;
        
        Eigen::AffineCompact2f estimateSimilarityTransform(const Eigen::Ref<const Shape> &from, const Eigen::Ref<const Shape> &to);
        
        void shapeRelativePixelCoordinates(const Shape &s, const PixelCoordinates &abscoords, PixelCoordinates &relcoords, Eigen::VectorXi &closestLandmarks);
        
        const Rect &unitRectangle();
        
        Rect createRectangle(const Eigen::Vector2f &minCorner, const Eigen::Vector2f &maxCorner);

        Rect shapeBounds(const Eigen::Ref<const Shape> &s);
    }
}

#endif