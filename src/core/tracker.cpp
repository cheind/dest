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

            if (!io::VerifyTrackerBuffer(flatbuffers::Verifier(reinterpret_cast<const uint8_t*>(buf.data()), buf.size()))) {
                return false;
            }

            const io::Tracker *t = io::GetTracker(buf.data());
            load(*t);

            return true;
        }
        
        bool Tracker::fit(TrainingData &t) {
            Tracker::data &data = *_data;
            
            const int numShapes = static_cast<int>(t.shapes.size());
            const int numSamples = numShapes * t.params.numInitializationsPerImage;
            
            RegressorTraining rt;
            rt.trainingData = &t;
            rt.samples.resize(numSamples);
            rt.numLandmarks = static_cast<int>(t.shapes.front().cols());
            
            rt.meanShape = Shape::Zero(2, rt.numLandmarks);
            for (int i = 0; i < numShapes; ++i) {
                rt.meanShape += t.shapes[i];
            }
            rt.meanShape /= static_cast<float>(numShapes);            
            
            // Generate training triplets
            std::uniform_int_distribution<int> dist(0, numShapes - 1);
            for (int i = 0; i < numSamples; ++i) {
                int id = dist(t.rnd);
                
                rt.samples[i].idx = i % numShapes;
                rt.samples[i].estimate = t.shapes[id];
            }
            
            // Build cascade           
            data.cascade.resize(t.params.numCascades);
            
            for (int i = 0; i < t.params.numCascades; ++i) {
                DEST_LOG("Building cascade " << i << std::endl);
                data.cascade[i].fit(rt);
                
                // Update shape estimate
                for (int s = 0; s < numSamples; ++s) {
                    if (s < 20) {
                        cv::Mat tmp = util::drawShape(t.images[rt.samples[s].idx], rt.samples[s].estimate, cv::Scalar(0,255,0));
                        cv::imshow("x", tmp);
                        cv::waitKey();
                        
                        DEST_LOG( i << " " <<  (t.shapes[rt.samples[s].idx] - rt.samples[s].estimate).norm() << std::endl);
                    }
                    
                    rt.samples[s].estimate += data.cascade[i].predict(t.images[rt.samples[s].idx], rt.samples[s].estimate);
                    
                }
            }

            // Update internal data
            data.meanShape = rt.meanShape;
            data.meanShapeRectCorners = boundingBoxCornersOfShape(data.meanShape);
            
            return true;

        }
        
        Shape Tracker::predict(const Image &img, const Shape &shape) const
        {
            Tracker::data &data = *_data;
            
            Shape estimate = shape;

            const int numCascades = static_cast<int>(data.cascade.size());
            for (int i = 0; i < numCascades; ++i) {
                estimate += data.cascade[i].predict(img, estimate);               
            }

            return estimate;
        }

        Shape Tracker::initialShapeFromRect(const Shape &rect) const
        {
            Tracker::data &data = *_data;
            Eigen::AffineCompact2f t = estimateSimilarityTransform(data.meanShapeRectCorners, rect);
            return t * data.meanShape.colwise().homogeneous();
        }


        Shape Tracker::boundingBoxCornersOfShape(const Shape &s) const
        {
            const Eigen::Vector2f minC = s.rowwise().minCoeff();
            const Eigen::Vector2f maxC = s.rowwise().maxCoeff();

            Shape rect(2, 4);
            rect.col(0) = minC;
            rect.col(1) = Eigen::Vector2f(maxC(0), minC(1));
            rect.col(2) = Eigen::Vector2f(minC(0), maxC(1));
            rect.col(3) = maxC;

            return rect;
        }

        
    }
}
