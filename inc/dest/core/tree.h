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

#include <dest/core/image.h>
#include <dest/core/shape.h>
#include <dest/core/training_data.h>
#include <dest/io/dest_io_generated.h>
#include <memory>

namespace dest {
    namespace core {
    
        class Tree {
        public:
            
            Tree();
            Tree(const Tree &other);
            ~Tree();
            
            bool fit(TreeTraining &t);
            
            ShapeResidual predict(const PixelIntensities &intensities) const;
            
            flatbuffers::Offset<io::Tree> save(flatbuffers::FlatBufferBuilder &fbb) const;
            void load(const io::Tree &fbs);
            
        private:
            
            struct TreeNode;
            struct PartitionPredicate;
            struct NodeInfo;
            struct SplitInfo;
            
            bool splitNode(TreeTraining &t, const NodeInfo &parent, NodeInfo &left, NodeInfo &right);
            void makeLeaf(TreeTraining &t, const NodeInfo &n);
            void sampleSplitPositions(TreeTraining &t, std::vector<SplitInfo> &splits) const;
            float splitEnergy(TreeTraining &t, const NodeInfo &parent, const ShapeResidual &parentMeanResidual, const SplitInfo &split) const;
            
            struct data;
            std::unique_ptr<data> _data;
        };
        
    }
}

#endif