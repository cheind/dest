/**
    This file is part of Deformable Shape Tracking (DEST).

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/

#ifndef DEST_TESTER_H
#define DEST_TESTER_H

#include <dest/core/training_data.h>
#include <dest/core/tracker.h>
#include <vector>

namespace dest {
    namespace core {
        
        /**
            Base class for objects providing distance normalization used during tracker evaluation.
        */
        class DistanceNormalizer {
        public:
            virtual float operator()(const SampleData::Sample &s) const = 0;
        };
        
        /**
            Constant normalization
        */
        class ConstantDistanceNormalizer : public DistanceNormalizer {
        public:
            ConstantDistanceNormalizer(float c);
            virtual float operator()(const SampleData::Sample &s) const;
        private:
            float _c;
        };
        
        /**
            Normalize by inter landmark distance.
        */
        class LandmarkDistanceNormalizer : public DistanceNormalizer {
        public:
            LandmarkDistanceNormalizer();
            LandmarkDistanceNormalizer(int landmarkId0, int landmarkId1);
            virtual float operator()(const SampleData::Sample &s) const;
            
            static LandmarkDistanceNormalizer createInterocularNormalizerIMM();
            static LandmarkDistanceNormalizer createInterocularNormalizerIBug();
        private:
            int _l0, _l1;
        };
        
        struct TestResult {
            float meanNormalizedDistance;
            float stddevNormalizedDistance;
            float medianNormalizedDistance;
            float worstNormalizedDistance;
            std::vector<float> histNormalizedDistance;
        };
        
        /** 
            Test tracker performance.
         
            Computes the average Euclidean distance between target and predicted landmark positions.
            Distances per sample are normlized by the given functor.
         
            \param td SampleData to run tests on. Fills sample estimate with normalized tracker prediction.
            \param t Tracker to evaluate
            \param norm Functor providing a distance normalization factor per sample.
        */ 
        TestResult testTracker(SampleData &td, const Tracker &t, const DistanceNormalizer &norm);
        
    }
}

#endif