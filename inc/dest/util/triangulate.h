/**
    This file is part of Deformable Shape Tracking (DEST).

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/

#ifndef DEST_TRIANGULATE_H
#define DEST_TRIANGULATE_H

#include <dest/core/config.h>
#if !defined(DEST_WITH_OPENCV)
#error OpenCV is required for this part of DEST.
#endif

#include <opencv2/core/core.hpp>

namespace dest {
    namespace util {
        
        /**
            Triangulate shape landmarks.

            Performs Delaunay triangulation of landmarks and returns indices of
            triangles mapping to columns of shape. Each triplet of indices corresponds
            to one triangle.

            \param s Shape to triangulate.
            \returns Indices of triangle vertices.
        */
        std::vector<core::Shape::Index> triangulateShape(const core::Shape &s);

    }
}

#endif