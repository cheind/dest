/**
This file is part of Deformable Shape Tracking (DEST).

Copyright(C) 2015/2016 Christoph Heindl
All rights reserved.

This software may be modified and distributed under the terms
of the BSD license.See the LICENSE file for details.
*/

#include <dest/dest.h>
#include <opencv2/opencv.hpp>
#include <tclap/CmdLine.h>

#include <dest/face/face_detector.h>
#include <dest/util/draw.h>
#include <dest/util/convert.h>

/**
    Transfer expressions from one face to another.
*/
int main(int argc, char **argv)
{
    
    struct {
        std::string tracker;
        std::string detector;
        std::string device;
        float imageScale;
    } opts;
    
    try {
        TCLAP::CmdLine cmd("Track on video stream.", ' ', "0.9");
        
        TCLAP::ValueArg<std::string> detectorArg("d", "detector", "Detector to provide initial bounds.", true, "classifier.xml", "XML file", cmd);
        TCLAP::ValueArg<std::string> trackerArg("t", "tracker", "Tracker to align landmarks based initial bounds", true, "dest.bin", "Tracker file", cmd);
        TCLAP::ValueArg<float> imageScaleArg("", "image-scale", "Scale factor to be applied to input image.", false, 1.f, "float", cmd);
        TCLAP::UnlabeledValueArg<std::string> deviceArg("device", "Device to be opened. Either filename of video or camera device id.", true, "0", "string", cmd);
        
        cmd.parse(argc, argv);
        
        opts.tracker = trackerArg.getValue();
        opts.detector = detectorArg.getValue();
        opts.device = deviceArg.getValue();
        opts.imageScale = imageScaleArg.getValue();
    }
    catch (TCLAP::ArgException &e) {
        std::cerr << "Error: " << e.error() << " for arg " << e.argId() << std::endl;
        return -1;
    }
    
    dest::core::Tracker t;
    if (!t.load(opts.tracker)) {
        std::cerr << "Failed to load tracker." << std::endl;
        return -1;
    }
      
    dest::face::FaceDetector fd;
    if (!fd.loadClassifiers(opts.detector)) {
        std::cerr << "Failed to load classifiers." << std::endl;
        return -1;
    }
    
    cv::VideoCapture cap;
    
    if (opts.device.size() == 1 && isdigit(opts.device[0])) {
        // Open capture device by index
        cap.open(atoi(opts.device.c_str()));
        cap.set(CV_CAP_PROP_SETTINGS, 1);
    } else {
        // Open video video
        cap.open(opts.device.c_str());
    }
    
    if (!cap.isOpened()) {
        std::cerr << "Failed to open capture device." << std::endl;
        return -1;
    }
    
    bool done = false;
    
    // Capture the target face, the one expressions get applied to.
    cv::Mat tmp;
    cv::Mat targetRef, targetRefGray, targetRefCopy;
    dest::core::Shape targetShapeRef;
    
    while (!done) {
        cap >> tmp;
        
        if (tmp.empty())
            break;
        
        cv::resize(tmp, targetRef, cv::Size(), opts.imageScale, opts.imageScale);
        cv::cvtColor(targetRef, targetRefGray, CV_BGR2GRAY);
        
        dest::core::MappedImage img = dest::util::toDestHeaderOnly(targetRefGray);
        
        dest::core::Rect r;
        if (!fd.detectSingleFace(img, r)) {
            continue;
        }

        dest::core::ShapeTransform shapeToImage = dest::core::estimateSimilarityTransform(dest::core::unitRectangle(), r);
        targetShapeRef = t.predict(img, shapeToImage);

        targetRef.copyTo(targetRefCopy);
        dest::util::drawShape(targetRefCopy, targetShapeRef, cv::Scalar(0,255,0));
        
        cv::imshow("Input", targetRefCopy);
        int key = cv::waitKey(1);
        if (key != -1)
            done = true;
    }
    
    const int landmarkIdsNormalize[] = {27, 31}; // eyes
    const float unnormalizeTarget = (targetShapeRef.col(landmarkIdsNormalize[0]) - targetShapeRef.col(landmarkIdsNormalize[1])).norm();
    
    std::vector<dest::core::Shape::Index> tris = dest::util::triangulateShape(targetShapeRef);
    
    cv::Mat target = targetRef.clone();
    cv::Mat source, sourceGray, sourceCopy;
    dest::core::Shape sourceShape, sourceShapeRef;
    
    done = false;
    bool hasSourceRef = false;
    float normalizeSource = 1.f;
    while (!done) {
        cap >> tmp;
        
        if (tmp.empty())
            break;
        
        cv::resize(tmp, source, cv::Size(), opts.imageScale, opts.imageScale);
        cv::cvtColor(source, sourceGray, CV_BGR2GRAY);

        dest::core::MappedImage img = dest::util::toDestHeaderOnly(sourceGray);
        
        dest::core::Rect r;
        if (!fd.detectSingleFace(img, r)) {
            continue;
        }
        
        dest::core::ShapeTransform shapeToImage = dest::core::estimateSimilarityTransform(dest::core::unitRectangle(), r);
        sourceShape = t.predict(img, shapeToImage);
        
        if (hasSourceRef) {
            dest::core::Shape r = targetShapeRef + (sourceShape - sourceShapeRef) * normalizeSource * unnormalizeTarget;
            target.setTo(0);
            dest::util::pawShapeTexture(targetRef, target, targetShapeRef, r, tris);
            cv::imshow("Target", target);
        }
        
        source.copyTo(sourceCopy);
        dest::util::drawShape(sourceCopy, sourceShape, cv::Scalar(0,255,0));
        cv::imshow("Input", sourceCopy);
        
        int key = cv::waitKey(1);
        if (key == 'x') {
            done = true;
        } else if (key != -1) {
            sourceShapeRef = sourceShape;
            normalizeSource = 1.f / (sourceShapeRef.col(landmarkIdsNormalize[0]) - sourceShapeRef.col(landmarkIdsNormalize[1])).norm();
            hasSourceRef = true;
        }
    }

    return 0;
}
