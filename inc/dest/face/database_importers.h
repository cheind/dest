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

#ifndef DEST_DATABASE_IMPORTERS_H
#define DEST_DATABASE_IMPORTERS_H

#include <dest/core/shape.h>
#include <opencv2/core/core.hpp>
#include <string>
#include <vector>

namespace dest {
    namespace face {
        
        /**
            Load the IMM face database.
         
            References:
            Nordstrøm, Michael M., et al. 
            The IMM face database-an annotated dataset of 240 face images. Technical
            University of Denmark, DTU Informatics, Building 321, 2004.
        */
        bool importIMMFaceDatabase(const std::string &directory, std::vector<cv::Mat> &images, core::ShapeVector &shapes);
        
    }
}

#endif