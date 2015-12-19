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
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <iomanip>
#include <fstream>

namespace dest {
    namespace face {
        
        bool parseAsfFile(const std::string& fileName, core::Shape &s) {
            
            int landmarkCount = 0;
            
            std::ifstream file(fileName);
            
            std::string line;
            while (std::getline(file, line)) {
                if (line.size() > 0) {
                    if (line[0] != '#') {
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
                            stream >> x;
                            stream >> y;
                            
                            s(0, landmarkCount) = x;
                            s(1, landmarkCount) = y;
                            
                            ++landmarkCount;
                        }
                    }
                }
            }
            
            return s.rows() > 0 && s.cols() > 0;
        }
        
        bool importIMMFaceDatabase(const std::string &directory, std::vector<core::Image> &images, std::vector<core::Shape> &shapes) {
            
            images.clear();
            shapes.clear();
            
            bool ok = true;
            int i = 1;
            do {
                bool subIdOK = true;
                int j = 1;
                do {
                    std::stringstream prefix;
                    prefix  << directory << "/"
                            << std::setfill('0') << std::setw(2) << i
                            << "-"
                            << j << "m";
                    
                    const std::string fileNameImg = prefix.str() + ".jpg";
                    const std::string fileNamePts = prefix.str() + ".asf";
                    
                    core::Shape s;
                    bool asfOk = parseAsfFile(fileNamePts, s);
                    cv::Mat cvImg = cv::imread(fileNameImg, cv::IMREAD_GRAYSCALE);
                    
                    if(asfOk && !cvImg.empty()) {
                        cv::Mat cvImgF;
                        cvImg.convertTo(cvImgF, CV_32F);
                        
                        core::Image img;
                        cv::cv2eigen(cvImgF, img);
                        
                        images.push_back(img);
                        shapes.push_back(s);
                    } else {
                        subIdOK = false;
                    }
                    
                } while (subIdOK);
                
                if (j == 1) {
                    ok = false;
                }
                
                i++;
            } while (ok);
            
            return shapes.size() > 0;
        }
        
    }
}