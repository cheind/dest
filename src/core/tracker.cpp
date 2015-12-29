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
#include <dest/util/log.h>
#include <dest/util/draw.h>
#include <dest/io/matrix_io.h>
#include <fstream>
#include <iomanip>
#include <opencv2/opencv.hpp>

namespace dest {
    namespace core {
        
        struct Tracker::data {
            typedef std::vector<Regressor> RegressorVector;            
            RegressorVector cascade;
            Shape meanShape;
            Shape meanShapeRectCorners;            

            flatbuffers::Offset<io::Tracker> save(flatbuffers::FlatBufferBuilder &fbb) const {
                flatbuffers::Offset<io::MatrixF> lmeans = io::toFbs(fbb, meanShape);
                flatbuffers::Offset<io::MatrixF> lbounds = io::toFbs(fbb, meanShapeRectCorners);

                std::vector< flatbuffers::Offset<io::Regressor> > lregs;
                for (size_t i = 0; i < cascade.size(); ++i) {
                    lregs.push_back(cascade[i].save(fbb));
                }

                auto vregs = fbb.CreateVector(lregs);

                io::TrackerBuilder b(fbb);
                b.add_cascade(vregs);
                b.add_meanShape(lmeans);
                b.add_meanShapeRectCorners(lbounds);

                return b.Finish();
            }

            void load(const io::Tracker &fbs) {

                io::fromFbs(*fbs.meanShape(), meanShape);
                io::fromFbs(*fbs.meanShapeRectCorners(), meanShapeRectCorners);

                cascade.resize(fbs.cascade()->size());
                for (flatbuffers::uoffset_t i = 0; i < fbs.cascade()->size(); ++i) {
                    cascade[i].load(*fbs.cascade()->Get(i));
                }
            }
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

        flatbuffers::Offset<io::Tracker> Tracker::save(flatbuffers::FlatBufferBuilder &fbb) const
        {
            return _data->save(fbb);
        }

        void Tracker::load(const io::Tracker &fbs)
        {
            _data->load(fbs);
        }

        bool Tracker::save(const std::string &path) const
        {
            std::ofstream ofs(path, std::ofstream::binary);
            if (!ofs.is_open()) return false;

            flatbuffers::FlatBufferBuilder fbb;
            io::FinishTrackerBuffer(fbb, save(fbb));

            ofs.write(reinterpret_cast<char*>(fbb.GetBufferPointer()), fbb.GetSize());
            return !ofs.bad();
        }

        bool Tracker::load(const std::string &path)
        {
            std::ifstream ifs(path, std::ifstream::binary);
            if (!ifs.is_open()) return false;

            std::string buf;
            ifs.seekg(0, std::ios::end);
            buf.resize(static_cast<size_t>(ifs.tellg()));
            ifs.seekg(0, std::ios::beg);
            ifs.read(&buf[0], buf.size());

            if (ifs.bad())
                return false;

            flatbuffers::Verifier v(reinterpret_cast<const uint8_t*>(buf.data()), buf.size());
            if (!io::VerifyTrackerBuffer(v)) {
                return false;
            }

            const io::Tracker *t = io::GetTracker(buf.data());
            load(*t);

            return true;
        }
        
        bool Tracker::fit(TrainingData &t) {
            eigen_assert(!t.samples.empty());

            Tracker::data &data = *_data;
            
            const int numSamples = static_cast<int>(t.samples.size());

            RegressorTraining rt;
            rt.training = &t;
            rt.numLandmarks = static_cast<int>(t.samples.front().estimateInShapeSpace.cols());
            rt.input = t.input;
            
            rt.meanShape = Shape::Zero(2, rt.numLandmarks);
            for (int i = 0; i < numSamples; ++i) {
                rt.meanShape += t.samples[i].estimateInShapeSpace;
            }
            rt.meanShape /= static_cast<float>(numSamples);

            // Build cascade
            data.cascade.resize(t.params.numCascades);
            
            for (int i = 0; i < t.params.numCascades; ++i) {
                DEST_LOG("Building cascade " << i + 1 << std::endl);
                
                // Fit gradient boosted trees.
                data.cascade[i].fit(rt);
                
                // Update shape estimate
                for (int s = 0; s < numSamples; ++s) {
                    t.samples[s].estimateInShapeSpace +=
                        data.cascade[i].predict(t.input->images[t.samples[s].inputIdx],
                                                t.samples[s].estimateInShapeSpace,
                                                t.samples[s].shapeToImage);
                }
            }

            // Update internal data
            data.meanShape = rt.meanShape;
            data.meanShapeRectCorners = shapeBounds(data.meanShape);

            return true;

        }
        
        Shape Tracker::predict(const Image &img, const ShapeTransform &shapeToImage, std::vector<Shape> *stepResults) const
        {
            Tracker::data &data = *_data;

            Shape estimate = data.meanShape;
            const int numCascades = static_cast<int>(data.cascade.size());
            for (int i = 0; i < numCascades; ++i) {
                if (stepResults) {
                    stepResults->push_back(shapeToImage * estimate.colwise().homogeneous());
                }
                estimate += data.cascade[i].predict(img, estimate, shapeToImage);
            }

            Shape final = shapeToImage * estimate.colwise().homogeneous();

            if (stepResults) {
                stepResults->push_back(final);
            }

            return final;
        }        
    }
}
