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

#include <dest/core/tree.h>
#include <random>
#include <queue>

namespace dest {
    namespace core {
        
        struct Tree::SplitInfo {
            int idx1;
            int idx2;
            float threshold;
        };
        
        struct Tree::TreeNode {
            // For intermediate nodes
            Tree::SplitInfo split;
            // For leaf nodes
            ShapeResidual mean;
        };
        
        typedef std::pair<TreeTraining::SampleVector::iterator, TreeTraining::SampleVector::iterator> SampleRange;
        
        
        struct Tree::NodeInfo {
            int node;
            int depth;
            SampleRange range;
            
            NodeInfo() {}
            
            NodeInfo(int n, int d, const SampleRange &r)
            :node(n), depth(d), range(r)
            {}
        };
        
        inline ShapeResidual meanResidual(const SampleRange &r) {
            if (r.first != r.second) {
                ShapeResidual mean = ShapeResidual::Zero(2, r.first->residual.cols());
                for (TreeTraining::SampleVector::iterator i = r.first; i != r.second; ++i) {
                    mean += i->residual;
                }
                mean /= static_cast<int>(std::distance(r.first, r.second));
                return mean;
            }
            return ShapeResidual();
        }
        
        struct Tree::data {
            
            std::vector<Tree::TreeNode> nodes;
            int depth;
            
            data()
            : depth(0)
            {}
        };
        
        
        
        Tree::Tree()
        : _data(new data())
        {}
        
        Tree::~Tree()
        {}
        
        bool Tree::fit(TreeTraining &t)
        {
            std::vector<Tree::TreeNode> &nodes = _data->nodes;
            int &depth = _data->depth;
            
            depth = std::max<int>(t.maxDepth, 1);
            const int numNodes = (int)std::pow(2.0, depth) - 1;
            
            
            nodes.resize(numNodes);

            // For each intermediate node in bfs
            std::queue<NodeInfo> queue;
            queue.push(NodeInfo(0, 1, std::make_pair(t.samples.begin(), t.samples.end())));
            
            // At this point only leaf nodes are in the queue.
            while (!queue.empty()) {
                NodeInfo &nr = queue.front(); queue.pop();
                
                if (nr.depth < depth) {
                    // Generate a split
                    NodeInfo left, right;
                    if (splitNode(t, nr, left, right)) {
                        queue.push(left);
                        queue.push(right);
                    } else {
                        makeLeaf(t, nr);
                    }
                    
                } else {
                    makeLeaf(t, nr);
                }
            }
            
            
            
            return true;
        }
        
        struct Tree::PartitionPredicate {
            SplitInfo split;
            
            bool operator()(const TreeTraining::Sample &s) const {
                return (s.intensities(split.idx1) - s.intensities(split.idx2) > split.threshold);
            }
            
        };
        
        bool Tree::splitNode(TreeTraining &t, const NodeInfo &parent, NodeInfo &left, NodeInfo &right) {
            
            const bool emptyRange = parent.range.second == parent.range.first;
            if (emptyRange) {
                // Premature leaf
                return false;
            }
            
            // Generate random split positions
            std::vector<SplitInfo> splits;
            sampleSplitPositions(t, splits);
            
            const ShapeResidual meanResidualParent = meanResidual(parent.range);
            
            // Choose best split according to minimization of residual energy
            float maxEnergy = -std::numeric_limits<float>::max();
            size_t bestSplit = std::numeric_limits<size_t>::max();
            
            for (size_t i = 0; i <  splits.size(); ++i) {
                float e = splitEnergy(t, parent, meanResidualParent, splits[i]);
                if (e > maxEnergy) {
                    maxEnergy = e;
                    bestSplit = i;
                }
            }
            
            if (maxEnergy == -std::numeric_limits<float>::max()) {
                return false;
            }
            
            TreeNode &parentNode = _data->nodes[parent.node];
            parentNode.split = splits[bestSplit];
            
            PartitionPredicate pred;
            pred.split = splits[bestSplit];
            TreeTraining::SampleVector::iterator middle = std::partition(parent.range.first, parent.range.second, pred);
            
            if (middle == parent.range.first || middle == parent.range.second) {
                return false;
            }
            
            left.node = parent.node * 2 + 1;
            right.node = parent.node * 2 + 2;
            left.depth = right.depth = parent.depth + 1;
            left.range = SampleRange(parent.range.first, middle);
            right.range = SampleRange(middle, parent.range.second);
            
            
            return true;
        }
        
        void Tree::makeLeaf(TreeTraining &t, NodeInfo &ni) {
            const int numLandmarks = t.samples.front().residual.cols();
            
            Tree::TreeNode &leaf = _data->nodes[ni.node];
            leaf.split.idx1 = -1;
            leaf.split.idx2 = -1;
            leaf.mean = ShapeResidual::Zero(2, numLandmarks);
            for (TreeTraining::SampleVector::iterator iter = ni.range.first; iter < ni.range.second; ++iter) {
                leaf.mean += iter->residual;
            }
            int numInLeaf = static_cast<int>(std::distance(ni.range.first, ni.range.second));
            if (numInLeaf > 0)
                leaf.mean /= numInLeaf;
        }
        
        void Tree::sampleSplitPositions(TreeTraining &t, std::vector<SplitInfo> &splits) const
        {
            splits.clear();
            
            const int maxAttempts = 100;
            std::uniform_int_distribution<> di(0, t.pixelCoordinates.cols() - 1);
            std::uniform_real_distribution<float> dr(0.f, 1.f);
            
            for (int i = 0; i < t.numSplitPositions; ++i) {
            
                SplitInfo split;
                int iter = 0;
                float e;
                do {
                    split.idx1 = di(t.rnd);
                    split.idx2 = di(t.rnd);
                    float d = (t.pixelCoordinates.col(split.idx1) - t.pixelCoordinates.col(split.idx2)).norm();
                    e = std::exp(-t.lambda * d);
                    ++iter;
                
                } while (iter <= maxAttempts && split.idx1 == split.idx2 && (dr(t.rnd) < e));
                
                if (iter == maxAttempts)
                    break;
                
                split.threshold = dr(t.rnd) * 256.f;
                splits.push_back(split);
            
            }
        }
        
        float Tree::splitEnergy(TreeTraining &t, const NodeInfo &parent, const ShapeResidual &parentMeanResidual, const SplitInfo &split) const {
            const int numLandmarks = t.samples.front().residual.cols();
            
            int numLeft = 0;
            ShapeResidual rLeft = ShapeResidual::Zero(2, numLandmarks);
            
            PartitionPredicate pred;
            pred.split = split;
            
            
            for (TreeTraining::SampleVector::iterator iter = parent.range.first; iter != parent.range.second; ++iter) {
                if (pred(*iter)) {
                    rLeft += iter->residual;
                    ++numLeft;
                }
            }
            if (numLeft > 0) {
                rLeft /= numLeft;
            }
            
            const int numParent = static_cast<int>(std::distance(parent.range.first, parent.range.second));
            const int numRight = numParent - numLeft;
            
            ShapeResidual rRight = (numParent * parentMeanResidual - numLeft * rLeft) / numRight;
            
            return numLeft * rLeft.squaredNorm() + numRight * rRight.squaredNorm();
        }

        
        ShapeResidual Tree::predict(const std::vector<float> &intensities) const
        {
            const TreeNode *nodes = &_data->nodes[0];
            
            const int tests = _data->depth - 1;
            
            int n = 0;
            for (int i = 0; i < tests; ++i) {
                const TreeNode &node = nodes[n];
                
                if (node.split.idx1 < 0)
                    break; // premature leaf
                
                float val = intensities[node.split.idx1] - intensities[node.split.idx2];
                
                n = val > 0.f ? 2 * n + 1 : 2 * n + 2;
            }
            
            return nodes[n].mean;
        }

        
        
    }
}