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

namespace dest {
    namespace core {
    
        class Regressor {
        public:
            Regressor();
            ~Regressor();
            
            bool fit(const std::vector<Image> &images, const std::vector<Triplet> &triplets);
            
            Shape predict(const Image &img, const Shape &shape) const;
            
        private:
            struct data;
            std::unique_ptr<data> _data;
        };
        
    }
}

#endif