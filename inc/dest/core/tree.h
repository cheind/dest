/**
    This file is part of Deformable Shape Tracking (DEST).

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
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

        /**
            Binary decision tree.

            During training at each non-leaf node a set of split candidates is randomly generated.
            Each split candidate thresholds the difference of two randomly chosen pixel positions
            (with preference to closer locations). The threshold is also chosen uniform randomly in
            the range [-64, 64]. The best split is found by finding the minimum of a split energy
            function that measures the distance between each shape residual (i.e what's left to
            converge to true shape) in the left child and and the mean of shape residuals in the left
            node plus the same thing for right child.

            This tree is stored implicitely as linear array as in GBDT we usually deal with
            shallow trees without many empty branches.

            Provides parallelization of split position testing when OpenMP is enabled.

            Based on the work of
            [1] Kazemi, Vahid, and Josephine Sullivan.
                "One millisecond face alignment with an ensemble of regression trees."
                Computer Vision and Pattern Recognition (CVPR), 2014 IEEE Conference on. IEEE, 2014.
        */
        class Tree {
        public:

            Tree();
            Tree(const Tree &other);
            ~Tree();

            /**
                Fit tree to training data.
            */
            bool fit(TreeTraining &t);

            /**
                Predict incremental shape update from image intensities.

                \param intensities Image intensities
                \return Incremental shape update.
            */
            ShapeResidual predict(const PixelIntensities &intensities) const;

            /**
                Save tree to flatbuffers.
            */
            flatbuffers::Offset<io::Tree> save(flatbuffers::FlatBufferBuilder &fbb) const;

            /**
                Load tree from flatbuffers.
            */
            void load(const io::Tree &fbs);

        private:

            struct TreeNode;
            struct PartitionPredicate;
            struct NodeInfo;
            struct SplitInfo;

            /**
                Split the given node if applicable.
            */
            bool splitNode(TreeTraining &t, const NodeInfo &parent, NodeInfo &left, NodeInfo &right);

            /**
                Convert node into leaf.
            */
            void makeLeaf(TreeTraining &t, const NodeInfo &n);

            /**
                Randomly generate split candidates.
            */
            void sampleSplitPositions(TreeTraining &t, std::vector<SplitInfo> &splits) const;

            /**
                Compute the split energy for a single candidate.
            */
            float splitEnergy(TreeTraining &t, const NodeInfo &parent, const ShapeResidual &parentMeanResidual, const SplitInfo &split) const;

            struct data;
            std::unique_ptr<data> _data;
        };

    }
}

#endif
