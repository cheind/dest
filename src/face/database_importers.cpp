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

#include <dest/face/database_importers.h>
#include <dest/util/log.h>
#include <dest/util/draw.h>
#include <dest/util/convert.h>
#include <dest/util/glob.h>
#include <opencv2/opencv.hpp>
#include <iomanip>
#include <fstream>

namespace dest {
    namespace face {
        
        bool parseAsfFile(const std::string& fileName, core::Shape &s) {
            
            int landmarkCount = 0;
            
            std::ifstream file(fileName);
            
            std::string line;
            while (std::getline(file, line)) {
                if (line.size() > 0 && line[0] != '#') {
                    if (line.find(".jpg") != std::string::npos) {
                        // ignored: file name of jpg image
                    }
                    else if (line.size() < 10) {
                        int nbPoints = atol(line.c_str());
                        
                        s.resize(2, nbPoints);
                        s.fill(0);
                    }
                    else {
                        std::stringstream stream(line);
                        
                        std::string path;
                        std::string type;
                        float x, y;
                        
                        stream >> path;
                        stream >> type;
                        stream >> x;
                        stream >> y;
                        
                        s(0, landmarkCount) = x;
                        s(1, landmarkCount) = y;
                        
                        ++landmarkCount;
                    }
                }
            }
            
            return s.rows() > 0 && s.cols() > 0;
        }
        
        bool importIMMFaceDatabase(const std::string &directory, std::vector<core::Image> &images, std::vector<core::Shape> &shapes) {
            
            images.clear();
            shapes.clear();
            
            std::vector<std::string> paths = util::findFilesInDir(directory, "asf", true);
            DEST_LOG("Loading IMM database. Found " << paths.size() << " canditate entries." << std::endl);
            
            for (size_t i = 0; i < paths.size(); ++i) {
                const std::string fileNameImg = paths[i] + ".jpg";
                const std::string fileNamePts = paths[i] + ".asf";
                
                core::Shape s;
                bool asfOk = parseAsfFile(fileNamePts, s);
                cv::Mat cvImg = cv::imread(fileNameImg, cv::IMREAD_GRAYSCALE);
                
                if(asfOk && !cvImg.empty()) {
                    
                    // Scale to image dimensions
                    s.row(0) *= static_cast<float>(cvImg.cols);
                    s.row(1) *= static_cast<float>(cvImg.rows);
                    
                    core::Image img;
                    util::toDest(cvImg, img);
                    
                    images.push_back(img);
                    shapes.push_back(s);
                }
            }
            
            DEST_LOG("Successfully loaded " << images.size() << " entries from database." << std::endl);
            return shapes.size() > 0;
        }
        
    }
}