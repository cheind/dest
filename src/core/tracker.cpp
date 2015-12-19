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
        
        bool Tracker::fit(TrackerTraining &t) {
            
            
            const int numShapes = static_cast<int>(t.shapes.size());
            const int numSamples = numShapes * t.numInitializationsPerImage;
            
            RegressorTraining rt;
            rt.exponentialLambda = t.exponentialLambda;
            rt.images = t.images;
            rt.shapes = t.shapes;
            rt.learningRate = t.learningRate;
            rt.maxTreeDepth = t.maxTreeDepth;
            rt.numLandmarks = t.shapes.front().cols();
            rt.numPixelSamplePositions = t.numPixelSamplePositions;
            rt.numRandomSplitPositions = t.numRandomSplitPositions;
            rt.numTrees = t.numTrees;
            rt.samples.resize(numSamples);
            rt.rnd.seed(10);
            
            rt.meanShape = Shape::Zero(2, rt.numLandmarks);
            for (int i = 0; i < numShapes; ++i) {
                rt.meanShape += t.shapes[i];
            }
            rt.meanShape /= numShapes;
            
            // Generate training triplets
            std::uniform_int_distribution<int> dist(0, numShapes - 1);
            for (int i = 0; i < numSamples; ++i) {
                int id = dist(rt.rnd);
                
                rt.samples[i].idx = i % numShapes;
                rt.samples[i].estimate = t.shapes[id];
            }
            
            // Build cascade
            Tracker::data &data = *_data;
            data.cascade.resize(t.numCascades);
            
            for (int i = 0; i < t.numCascades; ++i) {
                data.cascade[i].fit(rt);
                
                // Update shape estimate
                for (int s = 0; s < numSamples; ++s) {
                    rt.samples[i].estimate += data.cascade[i].predict(t.images[rt.samples[i].idx], rt.samples[i].estimate);
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