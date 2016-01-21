/**
    This file is part of Deformable Shape Tracking (DEST).

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/

#ifndef DEST_SHAPE_H
#define DEST_SHAPE_H

#include <Eigen/Geometry>
#include <dest/core/image.h>

namespace dest {
    namespace core {

        /** Type of orientable rectangle. */
        typedef Eigen::Matrix<float, 2, 4, Eigen::DontAlign> Rect;

        /** Type of shape consisting of two-dimensional landmarks in columns */
        typedef Eigen::Matrix<float, 2, Eigen::Dynamic> Shape;

        /** Type of incremental shape estimate / residual. */
        typedef Shape ShapeResidual;

        /** Type of global similarity transform to convert between normalized shape space and image space */
        typedef Eigen::AffineCompact2f ShapeTransform;

        /**
            Estimate a best-fit similarity transform (rotation, translation, uniform scale) between shapes.

            Note, since a rectangle is also a shape you can use this method also for estimating best-fit
            transformations between bounding rectangles.

            \param from Source shape.
            \param to Target shape.
            \returns Estimated transform.
        */
        Eigen::AffineCompact2f estimateSimilarityTransform(const Eigen::Ref<const Shape> &from, const Eigen::Ref<const Shape> &to);

        /**
            Encode pixel coordinates relative to shape.

            In [1] relative encoding is used for approximations of landmark positions of evolving shapes. Pixel locations / intensities
            used as input for decision trees are chosen with respect to the mean shape of training samples. To warp these locations
            to the current shape estimate a crude approximation is performed by a single similarity transform in addition to local
            translations.

            For each absolute pixel coordinate this method looks up the nearest landmark and computes its relative offset.

            \param s Shape landmarks.
            \param abscoords Absolute pixel coordinates.
            \param relcoords Relative pixel coordinates to nearest landmark neighbor.
            \param closestLandmarks Indices of closest landmarks.

            [1] Kazemi, Vahid, and Josephine Sullivan.
                "One millisecond face alignment with an ensemble of regression trees."
                Computer Vision and Pattern Recognition (CVPR), 2014 IEEE Conference on. IEEE, 2014.

        */
        void shapeRelativePixelCoordinates(
            const Shape &s,
            const PixelCoordinates &abscoords,
            PixelCoordinates &relcoords,
            Eigen::VectorXi &closestLandmarks);

        /**
            The unit square centered around the origin.

            The unit rectangle is usually used to convert a shape from image space to
            normalized shape space. Given an initial bounding rectangle of the face (as determined
            by a face detector) and the unit rectangle, use estimateSimilarityTransform to
            retrieve a normalized shape transform (which you effectively is a PROCRUSTES analysis
            between two shapes).

            Note we have choses then origin as center for unit square so things are simplified
            when arbitrarily rotating initial shape bounds.

            \returns the unit square centered around the origin.
        */
        const Rect &unitRectangle();

        /**
            Create an axis aligned rectangles from two corners.
        */
        Rect createRectangle(const Eigen::Vector2f &minCorner, const Eigen::Vector2f &maxCorner);

        /**
            Determine the axis aligned bounding rectangles of the given shape.
        */
        Rect shapeBounds(const Eigen::Ref<const Shape> &s);
    }
}

#endif
