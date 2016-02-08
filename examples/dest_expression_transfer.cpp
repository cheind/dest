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
#include <dest/core/tester.h>
#include <random>

void drawTriangulation(cv::Mat img, const dest::core::Shape &s, const std::vector<int> &tris, cv::Scalar color);
void piecewiseWarp(const cv::Mat &src, cv::Mat &dst, const dest::core::Shape &srcShape, const dest::core::Shape &dstShape, const  std::vector<dest::core::Shape::Index> &tris);

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
            piecewiseWarp(targetRef, target, targetShapeRef, r, tris);
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

void drawTriangulation(cv::Mat img, const dest::core::Shape &s, const std::vector<int> &tris, cv::Scalar color)
{
    for (size_t i = 0; i < tris.size() / 3; ++i) {
        
        cv::line(img,
                 cv::Point2f(s(0, tris[i*3+0]), s(1, tris[i*3+0])),
                 cv::Point2f(s(0, tris[i*3+1]), s(1, tris[i*3+1])),
                 color, 1, CV_AA);
        
        cv::line(img,
                 cv::Point2f(s(0, tris[i*3+1]), s(1, tris[i*3+1])),
                 cv::Point2f(s(0, tris[i*3+2]), s(1, tris[i*3+2])),
                 color, 1, CV_AA);
        
        cv::line(img,
                 cv::Point2f(s(0, tris[i*3+2]), s(1, tris[i*3+2])),
                 cv::Point2f(s(0, tris[i*3+0]), s(1, tris[i*3+0])),
                 color, 1, CV_AA);
    }
}

void piecewiseWarp(const cv::Mat &src, cv::Mat &dst, const dest::core::Shape &srcShape, const dest::core::Shape &dstShape, const std::vector<dest::core::Shape::Index> &tris)
{
    cv::Mat warp(2, 3, CV_32FC1);
    cv::Mat warpImg = cv::Mat::zeros(dst.rows, dst.cols, dst.type());
    cv::Mat warpMask = cv::Mat::zeros(dst.rows, dst.cols, CV_8UC1);
    
    std::vector<cv::Point2f> sp(3), dp(3);
    std::vector<cv::Point> dpi(3);
    const int shift = 4;
    const float multiplier = (float) (1 << shift);
    for (size_t i = 0; i < tris.size() / 3; ++i) {
        sp[0].x = srcShape(0, tris[i*3+0]);
        sp[0].y = srcShape(1, tris[i*3+0]);
        sp[1].x = srcShape(0, tris[i*3+1]);
        sp[1].y = srcShape(1, tris[i*3+1]);
        sp[2].x = srcShape(0, tris[i*3+2]);
        sp[2].y = srcShape(1, tris[i*3+2]);

        dp[0].x = dstShape(0, tris[i*3+0]);
        dp[0].y = dstShape(1, tris[i*3+0]);
        dp[1].x = dstShape(0, tris[i*3+1]);
        dp[1].y = dstShape(1, tris[i*3+1]);
        dp[2].x = dstShape(0, tris[i*3+2]);
        dp[2].y = dstShape(1, tris[i*3+2]);
        
        cv::Rect roiSrc = cv::boundingRect(sp);
        cv::Rect roiDst = cv::boundingRect(dp);
        
        roiSrc.x -= 1; roiSrc.y -= 1;
        roiSrc.width += 2; roiSrc.height += 2;
        
        roiDst.x -= 1; roiDst.y -= 1;
        roiDst.width += 2; roiDst.height += 2;
        
        // Correct offsets
        sp[0].x -= roiSrc.x; sp[0].y -= roiSrc.y;
        sp[1].x -= roiSrc.x; sp[1].y -= roiSrc.y;
        sp[2].x -= roiSrc.x; sp[2].y -= roiSrc.y;
        
        dp[0].x -= roiDst.x; dp[0].y -= roiDst.y;
        dp[1].x -= roiDst.x; dp[1].y -= roiDst.y;
        dp[2].x -= roiDst.x; dp[2].y -= roiDst.y;
        
        warp = cv::getAffineTransform(sp, dp);
        
        cv::warpAffine(src(roiSrc), warpImg(roiDst), warp, roiDst.size());
        
        warpMask.setTo(0);
        cv::Mat warpMaskRoi = warpMask(roiDst);
        
        dpi[0] = dp[0] * multiplier;
        dpi[1] = dp[1] * multiplier;
        dpi[2] = dp[2] * multiplier;
        cv::fillConvexPoly(warpMaskRoi, &dpi[0], 3, cv::Scalar(255), -1, shift);
        
        warpImg(roiDst).copyTo(dst(roiDst), warpMaskRoi);
    }

}