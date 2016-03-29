/**
    This file is part of Deformable Shape Tracking (DEST).

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/

#ifndef DEST_REGRESSOR_H
#define DEST_REGRESSOR_H

#include <dest/core/image.h>
#include <dest/core/shape.h>
#include <dest/core/training_data.h>
#include <dest/io/dest_io_generated.h>
#include <memory>

namespace dest {
    namespace core {
        
        /**
            Multi-dimensional regressor based on GBDT (Gradient boosted decision trees).
        */
        class Regressor {
        public:
            Regressor();
            Regressor(const Regressor &other);
            ~Regressor();
			Regressor &operator=(const Regressor &other);
            
            /**
                Fit to training data.
            */
            bool fit(RegressorTraining &t);
            
            /** 
                Predict incremental shape from current shape estimate.

                \param img Image to sample from
                \param shape Current shape estimate
                \param shapeToImage Global similarity transform from normalized shape space to image.
            */
            ShapeResidual predict(const Eigen::Ref<const Image> &img, const Shape &shape, const ShapeTransform &shapeToImage) const;

            /**
                Save trained regressor to flatbuffers.
            */
            flatbuffers::Offset<io::Regressor> save(flatbuffers::FlatBufferBuilder &fbb) const;

            /**
                Load trained regressor from flatbuffers.
            */
            void load(const io::Regressor &fbs);
            
        private:
            
            PixelCoordinates sampleCoordinates(RegressorTraining &t) const;
            void readPixelIntensities(const Eigen::AffineCompact2f &shapeToShape, const Eigen::AffineCompact2f &shapeToImage, const Shape &s, const Image &i, PixelIntensities &intensities) const;
            
            struct data;
            std::unique_ptr<data> _data;
        };
        
    }
}

#endif