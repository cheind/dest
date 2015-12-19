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

#ifndef DEST_TREE_H
#define DEST_TREE_H

#include <dest/core/triplet.h>
#include <dest/core/image.h>
#include <dest/core/shape.h>
#include <memory>
#include <vector>
#include <random>

namespace dest {
    namespace core {
        
        struct TreeTraining {
            PixelCoordinates pixelCoordinates;
            
            struct Sample {
                ShapeResidual residual;
                PixelIntensities intensities;
                
                friend void swap(Sample& a, Sample& b)
                {
                    using std::swap;
                    swap(a.residual, b.residual);
                    swap(a.intensities, b.intensities);
                }
            };
            typedef std::vector<Sample> SampleVector;
            SampleVector samples;
            
            std::mt19937 rnd;
            
            int maxDepth;
            int numSplitPositions;
            float lambda;
        };
    
        class Tree {
        public:
            
            Tree();
            ~Tree();
            
            bool fit(TreeTraining &t);
            
            ShapeResidual predict(const PixelIntensities &intensities) const;
            
        private:
            struct TreeNode;
            struct PartitionPredicate;
            struct NodeInfo;
            struct SplitInfo;
            
            bool splitNode(TreeTraining &t, const NodeInfo &parent, NodeInfo &left, NodeInfo &right);
            void makeLeaf(TreeTraining &t, NodeInfo &n);
            void sampleSplitPositions(TreeTraining &t, std::vector<SplitInfo> &splits) const;
            float splitEnergy(TreeTraining &t, const NodeInfo &parent, const ShapeResidual &parentMeanResidual, const SplitInfo &split) const;
            
            struct data;
            std::unique_ptr<data> _data;
        };
        
    }
}

#endif