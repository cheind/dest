/**
    This file is part of Deformable Shape Tracking (DEST).

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/

#include <dest/core/tester.h>
#include <dest/util/log.h>

namespace dest {
    namespace core {
        
        ConstantDistanceNormalizer::ConstantDistanceNormalizer(float c)
        :_c(c)
        {}
        
        float ConstantDistanceNormalizer::operator()(const SampleData::Sample &s) const {
            return _c;
        }
        
        LandmarkDistanceNormalizer::LandmarkDistanceNormalizer(int landmarkId0, int landmarkId1)
        :_l0(landmarkId0), _l1(landmarkId1)
        {}
        
        LandmarkDistanceNormalizer::LandmarkDistanceNormalizer()
        :_l0(0), _l1(0)
        {}
        
        float LandmarkDistanceNormalizer::operator()(const SampleData::Sample &s) const {
            return 1.f / (s.target.col(_l0) - s.target.col(_l1)).norm();
        }
        
        LandmarkDistanceNormalizer LandmarkDistanceNormalizer::createInterocularNormalizerIBug() {
            /*  The average point-to-point Euclidean error normalized by the inter-ocular distance 
                (measured as the Euclidean distance between the outer corners of the eyes) will be 
                used as the error measure.
                See http://ibug.doc.ic.ac.uk/resources/300-W/
             */
            return LandmarkDistanceNormalizer(36, 45);
        }
        
        LandmarkDistanceNormalizer LandmarkDistanceNormalizer::createInterocularNormalizerIMM() {
            // http://doi.ieeecomputersociety.org/cms/Computer.org/dl/trans/tp/2008/03/figures/ttp20080305411.gif
            return LandmarkDistanceNormalizer(21, 13);
        }
        

        TestResult testTracker(SampleData &td, const Tracker &t, const DistanceNormalizer &norm) {
            TestResult r;
            r.meanNormalizedDistance = 0.f;
            
            double sumDevs = 0.0;
            long elements = 0;
            for (size_t i = 0; i < td.samples.size(); ++i) {
                
                dest::core::Shape estimateInImageSpace = t.predict(td.input->images[td.samples[i].inputIdx], td.samples[i].shapeToImage);
                td.samples[i].estimate = td.samples[i].shapeToImage.inverse() * estimateInImageSpace.colwise().homogeneous();
                
                double dev = (td.samples[i].target - td.samples[i].estimate).colwise().norm().sum();
                sumDevs += dev * norm(td.samples[i]);
                elements += static_cast<long>(td.samples[i].target.cols());
                
                
                if (i % 100 == 0)
                    DEST_LOG("Processing " << i << "/" << td.samples.size() << " elements.\r" << std::flush);
            }
            
            r.meanNormalizedDistance = static_cast<float>(sumDevs / (double)elements);
            return r;
        }
        
    }
}
