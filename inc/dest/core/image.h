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

#ifndef DEST_IMAGE_H
#define DEST_IMAGE_H

#include <Eigen/Core>
#include <random>

namespace dest {
    namespace core {
    
        /** Type of single channel intensity image. */
        typedef Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Image;

        /** Type of list of image coordinates in columns. */
        typedef Eigen::Matrix<float, 2, Eigen::Dynamic> PixelCoordinates;

        /** Type of list of sampled image intensities. */        
        typedef Eigen::Matrix<float, 1, Eigen::Dynamic> PixelIntensities;
        
        /**
            Read image intensities at given locations.

            Performs bilinear interpolation at coordinates given. When coordinates are out of image
            bounds a clamp to edge will be performed.

            \param img Image to sample from
            \param coords Sub-pixel coordinates to sample at.
            \param intentsities Bilinear interpolated intensities for all coordintes.
         */
        void readImage(const Image &img, const PixelCoordinates &coords, PixelIntensities &intensities);
        
    }
}

#endif