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

#ifndef DEST_TESTER_H
#define DEST_TESTER_H

#include <dest/core/training_data.h>
#include <dest/core/tracker.h>

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