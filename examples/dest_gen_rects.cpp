/**
    This file is part of Deformable Shape Tracking (DEST).

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/

#include <dest/dest.h>

#include <dest/io/database_io.h>
#include <dest/face/face_detector.h>
#include <dest/io/rect_io.h>
#include <dest/util/draw.h>
#include <dest/util/convert.h>
#include <dest/util/log.h>
#include <iostream>
#include <opencv2/opencv.hpp>

#include <tclap/CmdLine.h>
#include <opencv2/core/ocl.hpp>

float ratioRectShapeOverlap(const dest::core::Rect &r, const dest::core::Shape &s) {
    Eigen::Vector2f minC = r.col(0);
    Eigen::Vector2f maxC = r.col(3);
    
    int numOverlap = 0;
    for (dest::core::Shape::Index i = 0; i < s.cols(); ++i) {
        Eigen::Vector2f a = s.col(i) - minC;
        Eigen::Vector2f b = s.col(i) - maxC;
        
        if ((a.array() >= 0.f).all() && (b.array() <= 0.f).all())
            numOverlap += 1;
    }
    
    return (float)numOverlap / (float)s.cols();
}

/**
    Generate face rectangles for tracker training using OpenCV face detector.

    dest::Tracker training requires initial bounding rectangles to learn from. Use this
    tool to generate rectangles for an exisiting face/shape database.
*/
int main(int argc, char **argv)
{
    enum FallbackMode {
        Fallback_SimulateOpenCV,
        Fallback_TightBounds,
        Fallback_Skip
    };
    
    struct {
        std::vector<std::string> detectors;
        std::string db;
        std::string output;
        FallbackMode fbm;
        dest::io::ImportParameters importParams;
    } opts;

    try {
        TCLAP::CmdLine cmd("Generate initial bounding boxes for face detection using Viola-Jones algorithm in OpenCV.", ' ', "0.9");
        TCLAP::MultiArg<std::string> detectorsArg("d", "detector", "OpenCV classifier to load", true, "string");
        TCLAP::ValueArg<std::string> outputArg("o", "output", "CSV output file", false, "rectangles.csv", "string");
        TCLAP::ValueArg<std::string> fallbackArg("", "fallback", "What to do when OpenCV detector fails. Default is skip", false, "skip", "simulatecv, tightbounds, skip");
        
        TCLAP::ValueArg<int> maxImageSizeArg("", "load-max-size", "Maximum size of images in the database", false, 2048, "int");
        TCLAP::UnlabeledValueArg<std::string> databaseArg("database", "Path to database directory to load", true, "./db", "string");

        cmd.add(&detectorsArg);
        cmd.add(&outputArg);
        cmd.add(&maxImageSizeArg);
        cmd.add(&fallbackArg);
        cmd.add(&databaseArg);
        

        cmd.parse(argc, argv);

        opts.detectors.insert(opts.detectors.end(), detectorsArg.begin(), detectorsArg.end());
        opts.db = databaseArg.getValue();
        opts.output = outputArg.getValue();
        opts.importParams.maxImageSideLength = maxImageSizeArg.getValue();
        
        if (fallbackArg.getValue() == "simulatecv") {
            opts.fbm = Fallback_SimulateOpenCV;
        } else if (fallbackArg.getValue() == "tightbounds") {
            opts.fbm = Fallback_TightBounds;
        } else if (fallbackArg.getValue() == "skip") {
            opts.fbm = Fallback_Skip;
        } else {
            throw TCLAP::ArgException("Unknwon fallback method", fallbackArg.getName());
        }
        
    }
    catch (TCLAP::ArgException &e) {
        std::cerr << "Error: " << e.error() << " for arg " << e.argId() << std::endl;
        return -1;
    }
    
    // Note that OpenCV 3.0 / CascadeClassifier seems to have troubles when being reused.
    // Current solution is to disable OpenCL
    // See https://github.com/Itseez/opencv/issues/5475
    cv::ocl::setUseOpenCL(false);
   
    dest::core::InputData inputs;
    std::vector<dest::core::Rect> rects;
    std::vector<float> scalings;
    if (!dest::io::importDatabase(opts.db, "", inputs.images, inputs.shapes, rects, opts.importParams, &scalings)) {
        std::cout << "Failed to load database" << std::endl;
        return -1;
    }

    std::vector<dest::face::FaceDetector> detectors(opts.detectors.size());
    for (size_t i = 0; i < opts.detectors.size(); ++i) {
        if (!detectors[i].loadClassifiers(opts.detectors[i])) {
            std::cout << "Failed to load detector " << opts.detectors[i];
            return -1;
        }
    }

    size_t countDetectionSuccess = 0;

    // The OpenCV detector rectangles are significantly different from tight bounds.
    // The values below attempt to match simulate face rects from tight bounds 
    // (used when the detector fails).
    float scaleToCV = 1.25f; // Scale between rectangles
    float txToCV = -0.01f;   // Translation in x normalized by image width
    float tyToCV = -0.05f;   // Translation in y normalized by image height

    for (size_t i = 0; i < rects.size(); ++i) {
        

        std::vector<dest::core::Rect> faces;
        for (size_t j = 0; j < detectors.size(); ++j) {
            std::vector<dest::core::Rect> myrects;
            detectors[j].detectFaces(inputs.images[i], myrects);
            faces.insert(faces.end(), myrects.begin(), myrects.end());
        }
        
        // Find the face rect with a meaningful shape overlap
        float bestOverlap = 0.f;
        size_t bestId = std::numeric_limits<size_t>::max();
        
        for (size_t j = 0; j < faces.size(); ++j) {
            float o = ratioRectShapeOverlap(faces[j], inputs.shapes[i]);
            if (o > bestOverlap) {
                bestId = j;
                bestOverlap = o;
            }
        }
        
        const bool detectionSuccess = (bestId != std::numeric_limits<size_t>::max() && bestOverlap >= 0.5f);


        if (!detectionSuccess) {
            
            switch(opts.fbm) {
                case Fallback_SimulateOpenCV: {
                    dest::core::Rect r = dest::core::shapeBounds(inputs.shapes[i]);
                    // Match CV detector
                    
                    Eigen::AffineCompact2f t;
                    t.setIdentity();
                    t = Eigen::Translation2f(txToCV * inputs.images[i].cols(), tyToCV * inputs.images[i].rows()) * Eigen::Scaling(scaleToCV);
                    r = t * r.colwise().homogeneous();
                    // Match original image size.
                    rects[i] =  r / scalings[i];
                    break;
                }
                case Fallback_TightBounds: {
                    dest::core::Rect r = dest::core::shapeBounds(inputs.shapes[i]);
                    rects[i] = r / scalings[i];
                    break;
                }
                case Fallback_Skip: {
                    rects[i] = dest::core::Rect::Zero(2, 4);
                    break;
                }
            }
            
        } else {
            ++countDetectionSuccess;
            rects[i] = faces[bestId] / scalings[i];
        }
        
        if (i % 10 == 0) {
            std::cout << "Processing " << i << "\r" << std::flush;
        }
    }
    std::cout << "Detector successful on " << countDetectionSuccess << "/" << rects.size() << " shapes." << std::endl;
    
    dest::io::exportRectangles(opts.output, rects);

    return 0;
}
