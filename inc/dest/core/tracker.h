/**
    This file is part of Deformable Shape Tracking (DEST).

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/

#ifndef DEST_TRACKER_H
#define DEST_TRACKER_H

#include <dest/core/image.h>
#include <dest/core/shape.h>
#include <dest/core/training_data.h>
#include <dest/io/dest_io_generated.h>
#include <memory>
#include <string>

namespace dest {
    namespace core {

        /**
            Provides alignment of shape landmarks.

            Uses a cascade of learnt gradient boosted tree classifiers to estimate the best fit
            shape landmark positions from a given input image and a normalization transform.

            While the normalization transform is is used to model global shape properties (rough
            translation, rotation and uniform scaling), the cascade is used to incrementally refine
            the landmark positions.

            Based on the work of
            [1] Kazemi, Vahid, and Josephine Sullivan.
                "One millisecond face alignment with an ensemble of regression trees."
                Computer Vision and Pattern Recognition (CVPR), 2014 IEEE Conference on. IEEE, 2014.
        */
        class Tracker {
        public:
            Tracker();
            ~Tracker();
            Tracker(const Tracker &other);

            /**
                Fit to training data.
            */
            bool fit(SampleData &t);

            /**
                Predict shape landmarks from image and a global transform.

                Given a rough shape normalization transform (oder more accurately its inverse) and
                and an input image this method computes the position of the shape landmarks in the image,
                based on previously learnt samples.

                \param img Single channel intensity input image.
                \param shapeToImage Inverse of shape normalization transform. Note, you have to apply
                                    the same normalization routine as used during training. If for example,
                                    initial face locations have been found via the OpenCV face detector the
                                    resulting face rectangle was used for normalization, you should use the
                                    same routine here. Deviating from this strategy will lead to errononeous
                                    results.
                \param stepResults If not null, contains the results from each regression cascade.
                \returns the computed landmark positions in image space.
            */
            Shape predict(const Image &img, const ShapeTransform &shapeToImage, std::vector<Shape> *stepResults = 0) const;

            /**
                Save trained tracker to flatbuffers.
            */
            flatbuffers::Offset<io::Tracker> save(flatbuffers::FlatBufferBuilder &fbb) const;

            /**
                Load trained regressor from flatbuffers.
            */
            void load(const io::Tracker &fbs);

            /**
                Save trained tracker to file.
            */
            bool save(const std::string &path) const;

            /**
                Load trained tracker from file.
            */
            bool load(const std::string &path);

        private:

            struct data;
            std::unique_ptr<data> _data;
        };

    }
}

#endif
