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
#include <dest/core/image.h>
#include <string>
#include <vector>

namespace dest {
    namespace face {
        

        /**
            Options when importing databases
         */
        struct ImportParameters {
            int maxImageSideLength;
            bool generateVerticallyMirrored;
            
            ImportParameters();
        };

        /**
            Import a face database.

            Automatically tries to derive type of database from files in the directory. 
            Currently supported databases are
                - IMM face database, see importIMMFaceDatabase
                - ibug annotated database, see importIBugAnnotatedFaceDatabase
        */
        bool importFaceDatabase(const std::string &directory,
                                std::vector<core::Image> &images,
                                std::vector<core::Shape> &shapes,
                                const ImportParameters &opts = ImportParameters());
    
        /**
            Load the IMM face database.
         
            References:
            Nordstrøm, Michael M., et al. 
            The IMM face database-an annotated dataset of 240 face images. Technical
            University of Denmark, DTU Informatics, Building 321, 2004.
            http://www.imm.dtu.dk/~aam/datasets/datasets.html
        */
        bool importIMMFaceDatabase(const std::string &directory,
                                   std::vector<core::Image> &images,
                                   std::vector<core::Shape> &shapes,
                                   const ImportParameters &opts = ImportParameters());
        
        
        /** 
            Load the face databases annotated by ibug.
         
            References:
            C. Sagonas, G. Tzimiropoulos, S. Zafeiriou, M. Pantic. 
            A semi-automatic methodology for facial landmark annotation. Proceedings of IEEE Int’l Conf. 
            Computer Vision and Pattern Recognition (CVPR-W’13), 5th Workshop on Analysis and Modeling of Faces and Gestures (AMFG '13). Oregon, USA, June 2013.
            http://ibug.doc.ic.ac.uk/resources/facial-point-annotations/
         
            C. Sagonas, G. Tzimiropoulos, S. Zafeiriou, M. Pantic. 
            300 Faces in-the-Wild Challenge: The first facial landmark localization Challenge. 
            Proceedings of IEEE Int’l Conf. on Computer Vision (ICCV-W 2013), 300 Faces in-the-Wild Challenge (300-W). Sydney, Australia, December 2013
            http://ibug.doc.ic.ac.uk/resources/facial-point-annotations/
         */
        bool importIBugAnnotatedFaceDatabase(const std::string &directory,
                                             std::vector<core::Image> &images,
                                             std::vector<core::Shape> &shapes,
                                             const ImportParameters &opts = ImportParameters());
    }
}

#endif