/**
    This file is part of Deformable Shape Tracking (DEST).

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/

#ifndef DEST_IMAGE_H
#define DEST_IMAGE_H

#include <Eigen/Core>
#include <random>

namespace dest {
    namespace core {
    
        /** 
            Type of single channel intensity image. 
            Memory order is row-major to provide non-copy mapping with OpenCV matrices.
        */
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