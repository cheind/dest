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

#include <dest/core/tracker.h>
#include <dest/core/regressor.h>
#include <dest/core/log.h>
#include <dest/util/draw.h>
#include <opencv2/opencv.hpp>

namespace dest {
    namespace core {
        
        struct Tracker::data {
            typedef std::vector<Regressor> RegressorVector;
            
            RegressorVector cascade;
        };
        
        Tracker::Tracker()
        : _data(new data())
        {
        }
        
        Tracker::Tracker(const Tracker &other)
        : _data(new data(*other._data))
        {
        }
        
        Tracker::~Tracker()
        {}
        
        bool Tracker::fit(TrainingData &t) {
            
            
            const int numShapes = static_cast<int>(t.shapes.size());
            const int numSamples = numShapes * t.params.numInitializationsPerImage;
            
            RegressorTraining rt;
            rt.trainingData = &t;
            rt.samples.resize(numSamples);
            rt.numLandmarks = static_cast<int>(t.shapes.front().cols());
            
            rt.meanShape = Shape::Zero(2, rt.numLandmarks);
            for (int i = 0; i < numShapes; ++i) {
                rt.meanShape += t.shapes[i];
            }
            rt.meanShape /= numShapes;
            
            // Generate training triplets
            std::uniform_int_distribution<int> dist(0, numShapes - 1);
            for (int i = 0; i < numSamples; ++i) {
                int id = dist(t.rnd);
                
                rt.samples[i].idx = i % numShapes;
                rt.samples[i].estimate = t.shapes[id];
            }
            
            // Build cascade
            Tracker::data &data = *_data;
            data.cascade.resize(t.params.numCascades);
            
            for (int i = 0; i < t.params.numCascades; ++i) {
                DEST_LOG("Building cascade " << i << std::endl);
                data.cascade[i].fit(rt);
                
                // Update shape estimate
                for (int s = 0; s < numSamples; ++s) {
                    if (s < 10) {
                        cv::Mat tmp = util::drawShape(t.images[rt.samples[s].idx], rt.samples[s].estimate, cv::Scalar(0,255,0));
                        cv::imshow("x", tmp);
                        cv::waitKey();
                        
                        DEST_LOG( i << " " <<  (t.shapes[rt.samples[s].idx] - rt.samples[s].estimate).norm() << std::endl);
                    }
                    
                    rt.samples[s].estimate += data.cascade[i].predict(t.images[rt.samples[s].idx], rt.samples[s].estimate);
                    
                }
            }
            
            return true;

        }
        
        Shape Tracker::predict(const Image &img, const Shape &shape) const
        {
            return shape;
        }

        
    }
}