/**
    This file is part of Deformable Shape Tracking (DEST).

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/

#ifndef DEST_DATABASE_IO_H
#define DEST_DATABASE_IO_H

#include <dest/core/config.h>
#if !defined(DEST_WITH_OPENCV)
    #error OpenCV is required for this part of DEST.
#endif

#include <opencv2/core/core.hpp>

#include <dest/core/shape.h>
#include <dest/core/image.h>
#include <string>
#include <vector>
#include <memory>

namespace dest {
    namespace io {
        
        /**
            A database loader for a specific database.

            Use ShapeDatabase for generic access.
        */
        class DatabaseLoader {
        public:
            /**
                Return identifier of loader.
            */
            virtual std::string identifier() const = 0;

            /*
                Find all items in directory. 
            */
            virtual size_t glob(const std::string &directory) = 0;

            /**
                Load image of n-th item.
            */
            virtual bool loadImage(size_t index, cv::Mat &dst) = 0;

            /**
                Load shape of n-th item.
            */
            virtual bool loadShape(size_t index, core::Shape &dst) = 0;

            /**
                Access the shape permutation matrix for vertical mirroring.
                
                \returns a permutation matrix for shape landmarks when mirroring the image. 
                         When unsupported, return empty permutation matrix.
            */
            virtual Eigen::PermutationMatrix<Eigen::Dynamic> shapeMirrorMatrix() = 0;
        protected:
            cv::Mat loadImageFromFilePrefix(const std::string &prefix) const;
        };

        /** 
            Load the IMM face database.

            References:
                Nordstrøm, Michael M., et al.
                The IMM face database-an annotated dataset of 240 face images. Technical
                University of Denmark, DTU Informatics, Building 321, 2004.
                http://www.imm.dtu.dk/~aam/datasets/datasets.html
        */
        class DatabaseLoaderIMM : public DatabaseLoader {
        public:
            DatabaseLoaderIMM();
            ~DatabaseLoaderIMM();

            std::string identifier() const;
            virtual size_t glob(const std::string &directory);
            virtual bool loadImage(size_t index, cv::Mat &dst);
            virtual bool loadShape(size_t index, core::Shape &dst);
            virtual Eigen::PermutationMatrix<Eigen::Dynamic> shapeMirrorMatrix();

        private:
            struct data;
            std::unique_ptr<data> _data;
        };

        /**
            Load the ibug annotated databases.

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
        class DatabaseLoaderIBug : public DatabaseLoader {
        public:
            DatabaseLoaderIBug();
            ~DatabaseLoaderIBug();

            std::string identifier() const;
            virtual size_t glob(const std::string &directory);
            virtual bool loadImage(size_t index, cv::Mat &dst);
            virtual bool loadShape(size_t index, core::Shape &dst);
            virtual Eigen::PermutationMatrix<Eigen::Dynamic> shapeMirrorMatrix();

        private:
            struct data;
            std::unique_ptr<data> _data;
        };

        /**
            Generic base class for loading shapes and images from existing databases.
        */
        class ShapeDatabase {
        public:
            ShapeDatabase();
            ~ShapeDatabase();

            void enableMirroring(bool enable);
            void setMaxImageLoadSize(int size);
            void setLoaderType(const std::string &type);
            void setRectangles(const std::vector<core::Rect> &rects);
            void addLoader(std::shared_ptr<DatabaseLoader> l);
            std::string lastLoaderType() const;

            /**
                Load shapes / images from directory.

                \param directory directory containing training files
                \param images Loaded images
                \param shapes Loaded shapes
                \param rects Loaded rectangles
                \param scaleFactors If not null contains applied scale factors to images, rectangles and shapes.
            */
            bool load(const std::string &directory,
                      std::vector<core::Image> &images,
                      std::vector<core::Shape> &shapes,
                      std::vector<core::Rect> &rects,
                      std::vector<float> *scaleFactors = 0);

        private:

            bool imageNeedsScaling(cv::Size s, int maxImageSize, float &factor) const;
            void scaleImageShapeAndRect(cv::Mat &img, core::Shape &s, core::Rect &r, float factor) const;
            void mirrorImageShapeAndRectVertically(cv::Mat &img,
                core::Shape &s,
                core::Rect &r,
                const Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> &permLandmarks,
                const Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> &permRectangle) const;

            struct data;
            std::unique_ptr<data> _data;
        };
    }
}

#endif