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

#include <dest/core/image.h>
#include <dest/core/shape.h>
#include <dest/core/training_data.h>
#include <dest/io/dest_io_generated.h>
#include <memory>

namespace dest {
    namespace core {
        
            
        class Regressor {
        public:
            Regressor();
            Regressor(const Regressor &other);
            ~Regressor();
            
            bool fit(RegressorTraining &t);
            
            ShapeResidual predict(const Image &img, const Shape &shape, const Rect &rect) const;

            flatbuffers::Offset<io::Regressor> save(flatbuffers::FlatBufferBuilder &fbb) const;
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