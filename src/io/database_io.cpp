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

#include <dest/io/database_io.h>
#include <dest/util/log.h>
#include <dest/util/draw.h>
#include <dest/util/convert.h>
#include <dest/util/glob.h>
#include <dest/io/rect_io.h>
#include <opencv2/opencv.hpp>
#include <iomanip>
#include <fstream>

namespace dest {
    namespace io {
        
        ImportParameters::ImportParameters() {
            maxImageSideLength = std::numeric_limits<int>::max();
            generateVerticallyMirrored = false;
        }

        DatabaseType importDatabase(const std::string & directory,
                                    const std::string &rectangleFile,
                                    std::vector<core::Image>& images,
                                    std::vector<core::Shape>& shapes,
                                    std::vector<core::Rect>& rects,
                                    const ImportParameters & opts,
                                    std::vector<float> *scaleFactors)
        {
            const bool isIMM = util::findFilesInDir(directory, "asf", true).size() > 0;
            const bool isIBUG = util::findFilesInDir(directory, "pts", true).size() > 0;

            if (isIMM) {
                bool ok = importIMMFaceDatabase(directory, rectangleFile, images, shapes, rects, opts, scaleFactors);
                return ok ? DATABASE_IMM : DATABASE_ERROR;
            } else if (isIBUG) {
                bool ok = importIBugAnnotatedFaceDatabase(directory, rectangleFile, images, shapes, rects, opts, scaleFactors);
                return ok ? DATABASE_IBUG : DATABASE_ERROR;
            } else {
                DEST_LOG("Unknown database format." << std::endl);
                return DATABASE_ERROR;
            }
        }
        
        bool imageNeedsScaling(cv::Size s, const ImportParameters &p, float &factor) {
            int maxLen = std::max<int>(s.width, s.height);
            if (maxLen > p.maxImageSideLength) {
                factor = static_cast<float>(p.maxImageSideLength) / static_cast<float>(maxLen);
                return true;
            } else {
                factor = 1.f;
                return false;
            }
        }
        
        void scaleImageShapeAndRect(cv::Mat &img, core::Shape &s, core::Rect &r, float factor) {
            cv::resize(img, img, cv::Size(0,0), factor, factor, CV_INTER_CUBIC);
            s *= factor;
            r *= factor;
        }
        
        void mirrorImageShapeAndRectVertically(cv::Mat &img,
                                               core::Shape &s,
                                               core::Rect &r,
                                               const Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> &permLandmarks,
                                               const Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> &permRectangle)
        {
            cv::flip(img, img, 1);
            for (core::Shape::Index i = 0; i < s.cols(); ++i) {
                s(0, i) = static_cast<float>(img.cols - 1) - s(0, i);
            }
            s = (s * permLandmarks).eval();
            
            
            for (core::Rect::Index i = 0; i < r.cols(); ++i) {
                r(0, i) = static_cast<float>(img.cols - 1) - r(0, i);
            }
            
            r = (r * permRectangle).eval();
        }
        
        Eigen::PermutationMatrix<Eigen::Dynamic> createPermutationMatrixForMirroredRectangle() {
            Eigen::PermutationMatrix<Eigen::Dynamic> perm(4);
            perm.setIdentity();
            Eigen::PermutationMatrix<Eigen::Dynamic>::IndicesType &ids = perm.indices();
            
            std::swap(ids(0), ids(1));
            std::swap(ids(2), ids(3));
            
            return perm;
        }
        
        const Eigen::PermutationMatrix<Eigen::Dynamic> &permutationMatrixForMirroredRectangle() {
            const static Eigen::PermutationMatrix<Eigen::Dynamic> _instance = createPermutationMatrixForMirroredRectangle();
            return _instance;
        }
        
        Eigen::PermutationMatrix<Eigen::Dynamic> createPermutationMatrixForMirroredIMM() {
            Eigen::PermutationMatrix<Eigen::Dynamic> perm(58);
            perm.setIdentity();
            Eigen::PermutationMatrix<Eigen::Dynamic>::IndicesType &ids = perm.indices();
            
            // http://ibug.doc.ic.ac.uk/resources/facial-point-annotations/
            
            // Contour
            std::swap(ids(0), ids(12));
            std::swap(ids(1), ids(11));
            std::swap(ids(2), ids(10));
            std::swap(ids(3), ids(9));
            std::swap(ids(4), ids(8));
            std::swap(ids(5), ids(7));
            std::swap(ids(6), ids(6));
            
            // Eye
            std::swap(ids(13), ids(21));
            std::swap(ids(14), ids(22));
            std::swap(ids(15), ids(23));
            std::swap(ids(16), ids(24));
            std::swap(ids(17), ids(25));
            std::swap(ids(18), ids(26));
            std::swap(ids(19), ids(27));
            std::swap(ids(20), ids(28));
            
            // Eyebrow
            std::swap(ids(29), ids(34));
            std::swap(ids(30), ids(35));
            std::swap(ids(31), ids(36));
            std::swap(ids(32), ids(37));
            std::swap(ids(33), ids(38));
            
            // Mouth
            std::swap(ids(39), ids(43));
            std::swap(ids(46), ids(44));
            std::swap(ids(41), ids(41));
            std::swap(ids(40), ids(42));
            std::swap(ids(45), ids(45));
            
            // Nose
            std::swap(ids(47), ids(57));
            std::swap(ids(48), ids(56));
            std::swap(ids(49), ids(55));
            std::swap(ids(50), ids(54));
            std::swap(ids(51), ids(53));
            std::swap(ids(52), ids(52));
            
            
            return perm;
        }
        
        const Eigen::PermutationMatrix<Eigen::Dynamic> &permutationMatrixForMirroredIMM() {
            const static Eigen::PermutationMatrix<Eigen::Dynamic> _instance = createPermutationMatrixForMirroredIMM();
            return _instance;
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
        
        bool importIMMFaceDatabase(const std::string &directory,
                                   const std::string &rectangleFile,
                                   std::vector<core::Image> &images,
                                   std::vector<core::Shape> &shapes,
                                   std::vector<core::Rect>& rects,
                                   const ImportParameters &opts,
                                   std::vector<float> *scaleFactors)
        {
                                    
            std::vector<std::string> paths = util::findFilesInDir(directory, "asf", true);
            DEST_LOG("Loading IMM database. Found " << paths.size() << " candidate entries." << std::endl);

            std::vector<core::Rect> loadedRects;
            io::importRectangles(rectangleFile, loadedRects);

            if (loadedRects.empty()) {
                DEST_LOG("No rectangles found, using tight axis aligned bounds." << std::endl);
            } else {
                if (paths.size() != loadedRects.size()) {
                    DEST_LOG("Mismatch between number of shapes in database and rectangles found." << std::endl);
                    return false;
                }
            }
            
            size_t initialSize = images.size();
            
            for (size_t i = 0; i < paths.size(); ++i) {
                const std::string fileNameImg = paths[i] + ".jpg";
                const std::string fileNamePts = paths[i] + ".asf";
                
                core::Shape s;
                core::Rect r;
                bool asfOk = parseAsfFile(fileNamePts, s);
                cv::Mat cvImg = cv::imread(fileNameImg, cv::IMREAD_GRAYSCALE);
                
                if(asfOk && !cvImg.empty()) {
                    
                    // Scale to image dimensions
                    s.row(0) *= static_cast<float>(cvImg.cols);
                    s.row(1) *= static_cast<float>(cvImg.rows);

                    if (loadedRects.empty()) {
                        r = core::shapeBounds(s);
                    } else {
                        r = loadedRects[i];
                    }
                    
                    float f;
                    if (imageNeedsScaling(cvImg.size(), opts, f)) {
                        scaleImageShapeAndRect(cvImg, s, r, f);
                    }
                    
                    if (scaleFactors) {
                        scaleFactors->push_back(f);
                    }
                        
                    
                    core::Image img;
                    util::toDest(cvImg, img);
                    
                    images.push_back(img);
                    shapes.push_back(s);
                    rects.push_back(r);
                    
                    if (opts.generateVerticallyMirrored) {
                        cv::Mat cvFlipped = cvImg.clone();
                        mirrorImageShapeAndRectVertically(cvFlipped, s, r, permutationMatrixForMirroredIMM(), permutationMatrixForMirroredRectangle());
                        
                        core::Image imgFlipped;
                        util::toDest(cvFlipped, imgFlipped);
                         
                        images.push_back(imgFlipped);
                        shapes.push_back(s);
                        rects.push_back(r);
                        
                        if (scaleFactors) {
                            scaleFactors->push_back(f);
                        }
                    }
                }
            }
            
            DEST_LOG("Successfully loaded " << (shapes.size() - initialSize) << " entries from database." << std::endl);
            return shapes.size() > 0;
        }
        
        Eigen::PermutationMatrix<Eigen::Dynamic> createPermutationMatrixForMirroredIBug() {
            Eigen::PermutationMatrix<Eigen::Dynamic> perm(68);
            perm.setIdentity();
            //return perm;
            Eigen::PermutationMatrix<Eigen::Dynamic>::IndicesType &ids = perm.indices();
            
            // http://ibug.doc.ic.ac.uk/resources/facial-point-annotations/
            
            // Contour
            std::swap(ids(0), ids(16));
            std::swap(ids(1), ids(15));
            std::swap(ids(2), ids(14));
            std::swap(ids(3), ids(13));
            std::swap(ids(4), ids(12));
            std::swap(ids(5), ids(11));
            std::swap(ids(6), ids(10));
            std::swap(ids(7), ids(9));
            std::swap(ids(8), ids(8));
            
            // Eyebrow
            std::swap(ids(17), ids(26));
            std::swap(ids(18), ids(25));
            std::swap(ids(19), ids(24));
            std::swap(ids(20), ids(23));
            std::swap(ids(21), ids(22));
            
            // Nose
            std::swap(ids(27), ids(27));
            std::swap(ids(28), ids(28));
            std::swap(ids(29), ids(29));
            std::swap(ids(30), ids(30));
            
            std::swap(ids(31), ids(35));
            std::swap(ids(32), ids(34));
            std::swap(ids(33), ids(33));
            
            // Eye
            std::swap(ids(39), ids(42));
            std::swap(ids(38), ids(43));
            std::swap(ids(37), ids(44));
            std::swap(ids(36), ids(45));
            std::swap(ids(40), ids(47));
            std::swap(ids(41), ids(46));
            
            // Mouth
            std::swap(ids(48), ids(54));
            std::swap(ids(49), ids(53));
            std::swap(ids(50), ids(52));
            std::swap(ids(51), ids(51));
            
            std::swap(ids(59), ids(55));
            std::swap(ids(58), ids(56));
            std::swap(ids(57), ids(57));
            
            std::swap(ids(60), ids(64));
            std::swap(ids(61), ids(63));
            std::swap(ids(62), ids(62));
            
            std::swap(ids(67), ids(65));
            std::swap(ids(66), ids(66));
            
            return perm;
        }
        
        const Eigen::PermutationMatrix<Eigen::Dynamic> &permutationMatrixForMirroredIBug() {
            const static Eigen::PermutationMatrix<Eigen::Dynamic> _instance = createPermutationMatrixForMirroredIBug();
            return _instance;
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
        
        bool importIBugAnnotatedFaceDatabase(const std::string &directory,
                                             const std::string &rectangleFile,
                                             std::vector<core::Image> &images,
                                             std::vector<core::Shape> &shapes,
                                             std::vector<core::Rect> &rects,
                                             const ImportParameters &opts,
                                             std::vector<float> *scaleFactors)
        {
            
            std::vector<std::string> paths = util::findFilesInDir(directory, "pts", true);
            DEST_LOG("Loading ibug database. Found " << paths.size() << " candidate entries." << std::endl);

            std::vector<core::Rect> loadedRects;
            io::importRectangles(rectangleFile, loadedRects);

            if (loadedRects.empty()) {
                DEST_LOG("No rectangles found, using tight axis aligned bounds." << std::endl);
            }
            else {
                if (paths.size() != loadedRects.size()) {
                    DEST_LOG("Mismatch between number of shapes in database and rectangles found." << std::endl);
                    return false;
                }
            }
            
            size_t initialSize = images.size();
            
            for (size_t i = 0; i < paths.size(); ++i) {
                const std::string fileNameImg = paths[i] + ".jpg";
                const std::string fileNamePts = paths[i] + ".pts";
                
                core::Shape s;
                core::Rect r;
                bool ptsOk = parsePtsFile(fileNamePts, s);
                cv::Mat cvImg = cv::imread(fileNameImg, cv::IMREAD_GRAYSCALE);
                
                if(ptsOk && !cvImg.empty()) {

                    if (loadedRects.empty()) {
                        r = core::shapeBounds(s);
                    }
                    else {
                        r = loadedRects[i];
                    }
                    
                    float f;
                    if (imageNeedsScaling(cvImg.size(), opts, f)) {
                        scaleImageShapeAndRect(cvImg, s, r, f);
                    }
                    
                    core::Image img;
                    util::toDest(cvImg, img);
                    
                    images.push_back(img);
                    shapes.push_back(s);
                    rects.push_back(r);
                    
                    if (scaleFactors) {
                        scaleFactors->push_back(f);
                    }
                    
                    
                    if (opts.generateVerticallyMirrored) {
                        cv::Mat cvFlipped = cvImg.clone();
                        mirrorImageShapeAndRectVertically(cvFlipped, s, r, permutationMatrixForMirroredIBug(), permutationMatrixForMirroredRectangle());
                        
                        core::Image imgFlipped;
                        util::toDest(cvFlipped, imgFlipped);
                        
                        images.push_back(imgFlipped);
                        shapes.push_back(s);
                        rects.push_back(r);
                        
                        if (scaleFactors) {
                            scaleFactors->push_back(f);
                        }
                    }
                }
            }
            
            DEST_LOG("Successfully loaded " << (shapes.size() - initialSize) << " entries from database." << std::endl);
            return (shapes.size() - initialSize) > 0;

        }
        
    }
}