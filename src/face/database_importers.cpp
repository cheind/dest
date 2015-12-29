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
        
        ImportParameters::ImportParameters() {
            maxImageSideLength = std::numeric_limits<int>::max();
            generateVerticallyMirrored = false;
        }

        bool importRectangles(const std::string &pathToCSV, std::vector<core::Rect> &rects) {
            std::ifstream file(pathToCSV);
            if (!file.is_open())
                return false;
            
            std::string line;
            while (std::getline(file, line)) {
                
                if (line.empty())
                    break;
                
                core::Rect r(2, 4);
                std::istringstream str(line);
                str >> r(0, 0) >> r(0, 1) >> r(0, 2) >> r(0, 3) >> r(1, 0) >> r(1, 1) >> r(1, 2) >> r(1, 3);
                rects.push_back(r);
            }
            
            return true;
        }

        bool exportRectangles(const std::string &pathToCSV, const std::vector<core::Rect> &rects) {
            std::ofstream ofs(pathToCSV);
            if (!ofs.is_open())
                return false;
            
            Eigen::IOFormat csvFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", " ");
            for(size_t i = 0; i < rects.size(); ++i) {
                ofs << rects[i].format(csvFormat) << std::endl;
            }
            
            ofs.close();
            return true;
        }
        
        bool imageNeedsScaling(cv::Size s, const ImportParameters &p, float &factor) {
            int maxLen = std::max<int>(s.width, s.height);
            if (maxLen > p.maxImageSideLength) {
                factor = static_cast<float>(p.maxImageSideLength) / static_cast<float>(maxLen);
                return true;
            } else {
                return false;
            }
        }
        
        void scaleImageAndShape(cv::Mat &img, core::Shape &s, float factor) {
            cv::resize(img, img, cv::Size(0,0), factor, factor, CV_INTER_CUBIC);
            s *= factor;
        }
        
        void mirrorImageAndShapeVertically(const cv::Mat &img, const core::Shape &s, cv::Mat &dstImage, core::Shape &dstShape) {
            cv::flip(img, dstImage, 1);
            dstShape.resize(2, s.cols());
            for (core::Shape::Index i = 0; i < s.cols(); ++i) {
                dstShape(0, i) = static_cast<float>(img.cols - 1) - s(0, i);
                dstShape(1, i) = s(1, i);
            }
        }
        
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
        
        bool importIMMFaceDatabase(const std::string &directory, std::vector<core::Image> &images, std::vector<core::Shape> &shapes, const ImportParameters &opts) {
            
            
            std::vector<std::string> paths = util::findFilesInDir(directory, "asf", true);
            DEST_LOG("Loading IMM database. Found " << paths.size() << " canditate entries." << std::endl);
            
            size_t initialSize = images.size();
            
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
                    
                    float f;
                    if (imageNeedsScaling(cvImg.size(), opts, f)) {
                        scaleImageAndShape(cvImg, s, f);
                    }
                    
                    core::Image img;
                    util::toDest(cvImg, img);
                    
                    images.push_back(img);
                    shapes.push_back(s);
                    
                    if (opts.generateVerticallyMirrored) {
                        cv::Mat cvFlipped;
                        core::Shape shapeFlipped;
                        mirrorImageAndShapeVertically(cvImg, s, cvFlipped, shapeFlipped);
                        
                        core::Image imgFlipped;
                        util::toDest(cvFlipped, imgFlipped);
                        
                        images.push_back(imgFlipped);
                        shapes.push_back(shapeFlipped);
                    }
                }
            }
            
            DEST_LOG("Successfully loaded " << (shapes.size() - initialSize) << " entries from database." << std::endl);
            return shapes.size() > 0;
        }
        
        
        bool parsePtsFile(const std::string& fileName, core::Shape &s) {
            
            std::ifstream file(fileName);
            
            std::string line;
            std::getline(file, line); // Version
            std::getline(file, line); // NPoints
            
            int numPoints;
            std::stringstream str(line);
            str >> line >> numPoints;
            
            std::getline(file, line); // {
            
            s.resize(2, numPoints);
            s.fill(0);
            
            for (int i = 0; i < numPoints; ++i) {
                if (!std::getline(file, line)) {
                    DEST_LOG("Failed to read points." << std::endl);
                    return false;
                }
                str = std::stringstream(line);
                
                float x, y;
                str >> x >> y;
                
                s(0, i) = x - 1.f; // Matlab to C++ offset
                s(1, i) = y - 1.f;
                
            }
            
            return true;
            
        }
        
        bool importIBugAnnotatedFaceDatabase(const std::string &directory, std::vector<core::Image> &images, std::vector<core::Shape> &shapes, const ImportParameters &opts) {
            
            std::vector<std::string> paths = util::findFilesInDir(directory, "pts", true);
            DEST_LOG("Loading ibug database. Found " << paths.size() << " canditate entries." << std::endl);
            
            size_t initialSize = images.size();
            
            for (size_t i = 0; i < paths.size(); ++i) {
                const std::string fileNameImg = paths[i] + ".jpg";
                const std::string fileNamePts = paths[i] + ".pts";
                
                core::Shape s;
                bool ptsOk = parsePtsFile(fileNamePts, s);
                cv::Mat cvImg = cv::imread(fileNameImg, cv::IMREAD_GRAYSCALE);
                
                if(ptsOk && !cvImg.empty()) {
                    
                    float f;
                    if (imageNeedsScaling(cvImg.size(), opts, f)) {
                        scaleImageAndShape(cvImg, s, f);
                    }
                    
                    core::Image img;
                    util::toDest(cvImg, img);
                    
                    images.push_back(img);
                    shapes.push_back(s);
                    
                    
                    if (opts.generateVerticallyMirrored) {
                        cv::Mat cvFlipped;
                        core::Shape shapeFlipped;
                        mirrorImageAndShapeVertically(cvImg, s, cvFlipped, shapeFlipped);
                        
                        core::Image imgFlipped;
                        util::toDest(cvFlipped, imgFlipped);
                        
                        images.push_back(imgFlipped);
                        shapes.push_back(shapeFlipped);
                    }
                }
            }
            
            DEST_LOG("Successfully loaded " << (shapes.size() - initialSize) << " entries from database." << std::endl);
            return (shapes.size() - initialSize) > 0;

        }
        
    }
}