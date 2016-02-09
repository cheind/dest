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

/**
    Swap faces in images.
*/
int main(int argc, char **argv)
{
    struct {
        std::string tracker;
        std::string detector;
        std::string image;
    } opts;
    
    try {
        TCLAP::CmdLine cmd("Track on video stream.", ' ', "0.9");
        
        TCLAP::ValueArg<std::string> detectorArg("d", "detector", "Detector to provide initial bounds.", true, "classifier.xml", "XML file", cmd);
        TCLAP::ValueArg<std::string> trackerArg("t", "tracker", "Tracker to align landmarks based initial bounds", true, "dest.bin", "Tracker file", cmd);
        TCLAP::UnlabeledValueArg<std::string> imageArg("image", "Image to be loaded containing multiple faces.", true, "image.png", "image file", cmd);
        
        cmd.parse(argc, argv);
        
        opts.tracker = trackerArg.getValue();
        opts.detector = detectorArg.getValue();
        opts.image = imageArg.getValue();
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
    
    cv::Mat imageRef = cv::imread(opts.image);
    if (imageRef.empty()) {
        std::cerr << "Failed to load image." << std::endl;
        return -1;
    }
    
    // Detect all faces
    std::vector<cv::Rect> faceRects;
    std::vector<dest::core::Shape> faces;
    
    if (!fd.detectFaces(imageRef, faceRects) || faceRects.size() < 2) {
        std::cerr << "Failed to find multiple faces." << std::endl;
        return -1;
    }
    
    // Run predictor on all face rectangles
    cv::Mat imageRefGray;
    cv::cvtColor(imageRef, imageRefGray, CV_BGR2GRAY);
    dest::core::MappedImage mappedGray = dest::util::toDestHeaderOnly(imageRefGray);
    
    std::vector<size_t> permutation;
    std::vector<dest::core::Shape> boundaryFaces;
    for (size_t i = 0; i < faceRects.size(); ++i) {
        dest::core::Rect r;
        dest::util::toDest(faceRects[i], r);
        dest::core::ShapeTransform shapeToImage = dest::core::estimateSimilarityTransform(dest::core::unitRectangle(), r);
        faces.push_back(t.predict(mappedGray, shapeToImage));
        
        std::vector<dest::core::Shape::Index> tris = dest::util::triangulateShape(faces.back());
        dest::core::Shape boundaryFace;
        dest::util::boundaryShapeVertices(faces.back(), tris, &boundaryFace);
        boundaryFaces.push_back(boundaryFace);
        
        permutation.push_back(i);
    }
                                          
    std::vector<dest::core::Shape::Index> boundaryTriangulation = dest::util::triangulateShape(boundaryFaces.front());
    
    cv::Mat final = imageRef.clone();
    
    bool done = false;
    while (!done) {
        // New permutation of faces
        std::rotate(permutation.begin(), permutation.begin() + 1, permutation.end());
        
        for (size_t i = 0; i < faces.size(); ++i) {
            size_t tid = permutation[i];
            dest::util::pawShapeTexture(imageRef, final, boundaryFaces[i], boundaryFaces[tid], boundaryTriangulation);
        }
        
        cv::imshow("Input", imageRef);
        cv::imshow("Face swap", final);
        int key = cv::waitKey();
        if (key == 'x')
            done = true;
    }
    
    
    
    
    


    
    
    return 0;
}