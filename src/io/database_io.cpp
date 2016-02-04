/**
    This file is part of Deformable Shape Tracking (DEST).

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/


#include <dest/core/config.h>
#ifdef DEST_WITH_OPENCV

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
        
        Eigen::PermutationMatrix<Eigen::Dynamic> createPermutationMatrixForMirroredRectangle() {
            Eigen::PermutationMatrix<Eigen::Dynamic> perm(4);
            perm.setIdentity();
            Eigen::PermutationMatrix<Eigen::Dynamic>::IndicesType &ids = perm.indices();
            
            std::swap(ids(0), ids(1));
            std::swap(ids(2), ids(3));
            
            return perm;
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

        Eigen::PermutationMatrix<Eigen::Dynamic> createPermutationMatrixForMirroredIBug() {
            Eigen::PermutationMatrix<Eigen::Dynamic> perm(68);
            perm.setIdentity();

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
        

        cv::Mat DatabaseLoader::loadImageFromFilePrefix(const std::string & prefix) const
        {
            const std::string extensions[] = { ".png", ".jpg", ".jpeg", ".bmp", "" };

            cv::Mat img;
            const std::string *ext = extensions;

            do {
                img = cv::imread(prefix + *ext, cv::IMREAD_GRAYSCALE);
                ++ext;
            } while (*ext != "" && img.empty());

            return img;
        }

        struct DatabaseLoaderIMM::data {
            std::vector<std::string> paths;
        };
        
        DatabaseLoaderIMM::DatabaseLoaderIMM()
            :_data(new data())
        {
        }

        DatabaseLoaderIMM::~DatabaseLoaderIMM()
        {
        }

        std::string DatabaseLoaderIMM::identifier() const
        {
            return std::string("imm");
        }

        size_t DatabaseLoaderIMM::glob(const std::string & directory)
        {
            _data->paths = util::findFilesInDir(directory, "asf", true, true);
            return _data->paths.size();
        }

        bool DatabaseLoaderIMM::loadImage(size_t index, cv::Mat & dst)
        {
            dst = this->loadImageFromFilePrefix(_data->paths[index]);
            return !dst.empty();
        }

        bool DatabaseLoaderIMM::loadShape(size_t index, core::Shape & dst)
        {

            dst = core::Shape();
            int landmarkCount = 0;

            const std::string fileName = _data->paths[index] + ".asf";
            std::ifstream file(fileName);

            std::string line;
            while (std::getline(file, line)) {
                if (line.size() > 0 && line[0] != '#') {
                    if (line.find(".jpg") != std::string::npos) {
                        // ignored: file name of jpg image
                    }
                    else if (line.size() < 10) {
                        int nbPoints = atol(line.c_str());

                        dst.resize(2, nbPoints);
                        dst.fill(0);
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

                        dst(0, landmarkCount) = x;
                        dst(1, landmarkCount) = y;

                        ++landmarkCount;
                    }
                }                
            }
            return dst.rows() > 0 && dst.cols() > 0;

        }

        Eigen::PermutationMatrix<Eigen::Dynamic> DatabaseLoaderIMM::shapeMirrorMatrix()
        {
            return createPermutationMatrixForMirroredIMM();
        }

        struct DatabaseLoaderIBug::data {
            std::vector<std::string> paths;
        };

        DatabaseLoaderIBug::DatabaseLoaderIBug()
            :_data(new data())
        {
        }
        
        DatabaseLoaderIBug::~DatabaseLoaderIBug()
        {
        }
        
        std::string DatabaseLoaderIBug::identifier() const
        {
            return std::string("ibug");
        }
        
        size_t DatabaseLoaderIBug::glob(const std::string & directory)
        {
            _data->paths = util::findFilesInDir(directory, "pts", true, true);
            return _data->paths.size();
        }

        bool DatabaseLoaderIBug::loadImage(size_t index, cv::Mat & dst)
        {
            dst = this->loadImageFromFilePrefix(_data->paths[index]);
            return !dst.empty();
        }

        bool DatabaseLoaderIBug::loadShape(size_t index, core::Shape & dst)
        {
            const std::string fileName = _data->paths[index] + ".pts";
            std::ifstream file(fileName);

            std::string line;
            std::getline(file, line); // Version
            std::getline(file, line); // NPoints

            int numPoints;
            std::stringstream str(line);
            str >> line >> numPoints;

            std::getline(file, line); // {

            dst.resize(2, numPoints);
            dst.fill(0);

            for (int i = 0; i < numPoints; ++i) {
                if (!std::getline(file, line)) {
                    DEST_LOG("Failed to read points." << std::endl);
                    return false;
                }
                str = std::stringstream(line);

                float x, y;
                str >> x >> y;

                dst(0, i) = x - 1.f; // Matlab to C++ offset
                dst(1, i) = y - 1.f;

            }

            return numPoints > 0;
        }

        Eigen::PermutationMatrix<Eigen::Dynamic> DatabaseLoaderIBug::shapeMirrorMatrix()
        {
            return createPermutationMatrixForMirroredIBug();
        }

        struct DatabaseLoaderLAND::data {
            std::vector<std::string> paths;
        };

        DatabaseLoaderLAND::DatabaseLoaderLAND()
            :_data(new data())
        {
        }

        DatabaseLoaderLAND::~DatabaseLoaderLAND()
        {
        }

        std::string DatabaseLoaderLAND::identifier() const
        {
            return std::string("land");
        }

        size_t DatabaseLoaderLAND::glob(const std::string & directory)
        {
            _data->paths = util::findFilesInDir(directory, "land", true, true);
            return _data->paths.size();
        }

        bool DatabaseLoaderLAND::loadImage(size_t index, cv::Mat & dst)
        {
            dst = this->loadImageFromFilePrefix(_data->paths[index]);
            return !dst.empty();
        }

        bool DatabaseLoaderLAND::loadShape(size_t index, core::Shape & dst)
        {
            const std::string fileName = _data->paths[index] + ".land";
            std::ifstream file(fileName);

            if (!file.is_open())
                return false;

            std::string line;
            std::getline(file, line); // Number of landmarks

            int numPoints;
            std::stringstream str(line);
            str >> numPoints;

            dst.resize(2, numPoints);
            dst.fill(0);

            for (int i = 0; i < numPoints; ++i) {
                if (!std::getline(file, line)) {
                    DEST_LOG("Failed to read points." << std::endl);
                    return false;
                }
                str = std::stringstream(line);

                float x, y;
                str >> x >> y;

                dst(0, i) = x;
                dst(1, i) = y;

            }

            return numPoints > 0;
        }

        Eigen::PermutationMatrix<Eigen::Dynamic> DatabaseLoaderLAND::shapeMirrorMatrix()
        {
            return Eigen::PermutationMatrix<Eigen::Dynamic>();
        }

        struct ShapeDatabase::data 
        {
            std::vector< std::shared_ptr<DatabaseLoader> > loaders;
            std::vector<core::Rect> rects;
            bool mirror;
            int maxLoadSize;
            std::string type, lastType;
        };

        ShapeDatabase::ShapeDatabase()
            :_data(new data())
        {
            _data->loaders.push_back(std::make_shared<DatabaseLoaderIMM>());
            _data->loaders.push_back(std::make_shared<DatabaseLoaderIBug>());
            _data->loaders.push_back(std::make_shared<DatabaseLoaderLAND>());

            _data->mirror = false;
            _data->maxLoadSize = std::numeric_limits<int>::max();
            _data->type = std::string("auto");
        }

        ShapeDatabase::~ShapeDatabase()
        {
        }

        void ShapeDatabase::enableMirroring(bool enable)
        {
            _data->mirror = enable;
        }

        void ShapeDatabase::setMaxImageLoadSize(int size)
        {
            _data->maxLoadSize = size;
        }

        void ShapeDatabase::setLoaderType(const std::string & type)
        {
            _data->type = type;
        }

        void ShapeDatabase::setRectangles(const std::vector<core::Rect>& rects)
        {
            _data->rects = rects;
        }

        void ShapeDatabase::addLoader(std::shared_ptr<DatabaseLoader> l)
        {
            _data->loaders.insert(_data->loaders.begin(), l);
        }

        std::string ShapeDatabase::lastLoaderType() const
        {
            return _data->lastType;
        }

        bool ShapeDatabase::load(const std::string & directory, std::vector<core::Image>& images, std::vector<core::Shape>& shapes, std::vector<core::Rect>& rects, std::vector<float>* scaleFactors)
        {
            std::shared_ptr<DatabaseLoader> loader;
            size_t candidates = 0;
            if (_data->type == std::string("auto")) {
                for (size_t i = 0; i < _data->loaders.size(); ++i) {
                    loader = _data->loaders[i];
                    candidates = loader->glob(directory);
                    if (candidates > 0)
                        break;
                }                
            } else {
                std::string type = _data->type;
                auto iter = std::find_if(_data->loaders.begin(), _data->loaders.end(), [type](std::shared_ptr<DatabaseLoader> &l) {
                    return l->identifier() == type;
                });
                if (iter != _data->loaders.end()) {
                    loader = *iter;
                    candidates = loader->glob(directory);
                }
            }

            if (candidates == 0) {
                DEST_LOG("Could not find any loadable items.");
                return false;
            }
            _data->lastType = loader->identifier();

            DEST_LOG("Loading " << loader->identifier() << " database. Found " << candidates << " candidate entries." << std::endl);

            std::vector<core::Rect> &loadedRects = _data->rects;
            if (loadedRects.empty()) {
                DEST_LOG("No rectangles found, using tight shape bounds." << std::endl);
            } else if (candidates != loadedRects.size()) {
                DEST_LOG("Mismatch between number of shapes in database and rectangles found." << std::endl);
                return false;
            }

            size_t initialSize = images.size();
            cv::Mat img;
            Eigen::PermutationMatrix<Eigen::Dynamic> permutShape = loader->shapeMirrorMatrix();
            Eigen::PermutationMatrix<Eigen::Dynamic> permutRect = createPermutationMatrixForMirroredRectangle();

            if (permutShape.size() == 0 && _data->mirror) {
                DEST_LOG("Mirroring will be skipped. Requested but database loader does not support it." << std::endl);
            }

            for (size_t i = 0; i < candidates; ++i) {
                core::Shape s;
                core::Rect r;
                
                bool shapeOk = loader->loadShape(i, s);
                bool imageOk = loader->loadImage(i, img);
                bool rectOk = loadedRects.empty() || !loadedRects[i].isZero();

                if (!shapeOk || !imageOk || !rectOk)
                    continue;

                r = loadedRects.empty() ? core::shapeBounds(s) : loadedRects[i];

                float f;
                if (imageNeedsScaling(img.size(), _data->maxLoadSize, f)) {
                    scaleImageShapeAndRect(img, s, r, f);
                }

                core::Image destImg;
                util::toDest(img, destImg);

                images.push_back(destImg);
                shapes.push_back(s);
                rects.push_back(r);

                if (scaleFactors) {
                    scaleFactors->push_back(f);
                }

                if (_data->mirror && permutShape.size() > 0) {
                    cv::Mat cvFlipped = img.clone();
                    mirrorImageShapeAndRectVertically(cvFlipped, s, r, permutShape, permutRect);

                    core::Image destImgFlipped;
                    util::toDest(cvFlipped, destImgFlipped);

                    images.push_back(destImgFlipped);
                    shapes.push_back(s);
                    rects.push_back(r);

                    if (scaleFactors) {
                        scaleFactors->push_back(f);
                    }
                }

            }

            DEST_LOG("Successfully loaded " << (shapes.size() - initialSize) << " entries from database." << std::endl);
            return (shapes.size() - initialSize) > 0;
        }

        bool ShapeDatabase::imageNeedsScaling(cv::Size s, int maxImageSize, float & factor) const
        {
            int maxLen = std::max<int>(s.width, s.height);
            if (maxLen > maxImageSize) {
                factor = static_cast<float>(maxImageSize) / static_cast<float>(maxLen);
                return true;
            }
            else {
                factor = 1.f;
                return false;
            }
        }

        void ShapeDatabase::scaleImageShapeAndRect(cv::Mat &img, core::Shape & s, core::Rect & r, float factor) const
        {
            cv::resize(img, img, cv::Size(0, 0), factor, factor, CV_INTER_CUBIC);
            s *= factor;
            r *= factor;
        }

        void ShapeDatabase::mirrorImageShapeAndRectVertically(cv::Mat & img, core::Shape & s, core::Rect & r, const Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>& permLandmarks, const Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>& permRectangle) const
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
       
}
}

#endif