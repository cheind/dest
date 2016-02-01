/**
    This file is part of Deformable Shape Tracking (DEST).

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/

#include <dest/core/tester.h>
#include <dest/util/log.h>
#include <numeric>

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
            r.medianNormalizedDistance = 0.f;
            r.stddevNormalizedDistance = 0.f;
            r.worstNormalizedDistance = 0.f;
            
            const int nLandmarks = static_cast<int>(td.samples.front().target.cols());
            std::vector<float> d;
            
            for (size_t i = 0; i < td.samples.size(); ++i) {
                
                dest::core::Shape estimateInImageSpace = t.predict(td.input->images[td.samples[i].inputIdx], td.samples[i].shapeToImage);
                td.samples[i].estimate = td.samples[i].shapeToImage.inverse() * estimateInImageSpace.colwise().homogeneous();
                
                const float normalizer = norm(td.samples[i]);
                Eigen::VectorXf dev = (td.samples[i].target - td.samples[i].estimate).colwise().norm() * normalizer;
                for (int j  = 0; j < nLandmarks; ++j) {
                    if (dev(j) > 1.9f)
                        std::cout << i << std::endl;
                    d.push_back(dev(j));
                }
            
                if (i % 100 == 0)
                    DEST_LOG("Processing " << i << "/" << td.samples.size() << " elements.\r" << std::flush);
            }
            
            std::sort(d.begin(), d.end());
            
            r.meanNormalizedDistance = std::accumulate(d.begin(), d.end(), 0.f) / (float)(d.size());                        
            float var = 0.f;
            for (size_t i = 0; i < d.size(); ++i) {
                var += (d[i] - r.meanNormalizedDistance) * (d[i] - r.meanNormalizedDistance);
            }
            r.medianNormalizedDistance = (d.size() % 2 == 0) ? d[d.size() / 2] : (d[d.size() / 2 - 1] + d[d.size() / 2]) * 0.5f;
            r.stddevNormalizedDistance = std::sqrt(var / (float)(d.size()));            
            r.worstNormalizedDistance = d.back();
            
            const int nbins = 20;
            const float binSize = 1.f / nbins;
            r.histNormalizedDistance = std::vector<float>(nbins + 1, 0.f);
            for (size_t i = 0; i < d.size(); ++i) {
                int bin = static_cast<int>(floor(d[i] / binSize));
                if (bin < nbins)
                    r.histNormalizedDistance[bin] += 1.f;
                else
                    r.histNormalizedDistance.back() += 1.f; // extra large values
            }            
            for (size_t i = 0; i < r.histNormalizedDistance.size(); ++i) {
                r.histNormalizedDistance[i] /= (float)(d.size());
            }
             
            return r;
        }
        
    }
}
