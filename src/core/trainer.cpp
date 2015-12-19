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

#include <dest/core/trainer.h>
#include <dest/core/triplet.h>
#include <dest/core/regressor.h>
#include <vector>
#include <random>

namespace dest {
    namespace core {
        
        struct Trainer::data {
            int numInitializationsPerShape;
            int numRegressors;
            std::mt19937::result_type randomSeed;
            
            data()
            {
                numInitializationsPerShape = 20;
                numRegressors = 10;
                randomSeed = std::mt19937::default_seed;
            }
        };
        
        Trainer::Trainer()
        : _data(new data())
        {
        }
        
        Trainer::~Trainer()
        {}
        
        Tracker Trainer::train(const std::vector<Image> &images, const std::vector<Shape> &shapes) const {
            
            // Generate training triplets
            const int numShapes = static_cast<int>(shapes.size());
            const int numTriplets = numShapes * _data->numInitializationsPerShape;
            std::vector<Triplet> triplets(numTriplets);
            
            std::mt19937 gen;
            gen.seed(_data->randomSeed);
            std::uniform_int_distribution<int> dist(0, numShapes - 1);
            
            for (size_t i = 0; i < numTriplets; ++i) {
                int id = dist(gen);
                
                triplets[i].shapeId = i % numShapes;
                triplets[i].shapeEstimate = shapes[id];
                triplets[i].shapeDelta = shapes[triplets[i].shapeId] - triplets[i].shapeEstimate;
            }
            
            std::vector<Regressor> cascade;
            
            
            return Tracker();
        }
        
    }
}