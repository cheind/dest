/**
    This file is part of Deformable Shape Tracking (DEST).

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/

#include <dest/core/image.h>

namespace dest {
    namespace core {
        
        inline int clampToEdge(int v, Image::Index len) {
            return std::min<int>(static_cast<int>(len) - 1, std::max<int>(0, v));
        }
        
        inline float bilinearSample(const Image &img, float x, float y) {
            
            const int ix = static_cast<int>(std::floor(x));
            const int iy = static_cast<int>(std::floor(y));
            
            int x0 = clampToEdge(ix, img.cols());
            int x1 = clampToEdge(ix + 1, img.cols());
            int y0 = clampToEdge(iy, img.rows());
            int y1 = clampToEdge(iy + 1, img.rows());

            float a = x - (float)ix;
            float b = y - (float)iy;
            
            const unsigned char *ptrY0 = img.row(y0).data();
            const unsigned char *ptrY1 = img.row(y1).data();
            
            const float f0 = static_cast<float>(ptrY0[x0]);
            const float f1 = static_cast<float>(ptrY0[x1]);
            const float f2 = static_cast<float>(ptrY1[x0]);
            const float f3 = static_cast<float>(ptrY1[x1]);
            
            return (f0 * (float(1) - a) + f1 * a) * (float(1) - b) +
                   (f2 * (float(1) - a) + f3 * a) * b;
        }
        
        void readImage(const Image &img, const PixelCoordinates &coords, PixelIntensities &intensities) {
            const int numCoords = static_cast<int>(coords.cols());
            
            intensities.resize(coords.cols());
            
            for (int i = 0; i < numCoords; ++i) {
                intensities(i) = bilinearSample(img, coords(0, i), coords(1, i));
            }
        }
        
    }
}