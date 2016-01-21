/**
    This file is part of Deformable Shape Tracking (DEST).

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/

#ifndef DEST_MATRIX_IO_H
#define DEST_MATRIX_IO_H

#include <Eigen/Core>
#include "dest_io_generated.h"

namespace dest {
    namespace io {
        
        /**
            Convert generic matrix to flatbuffers.
        */
        template<int Rows, int Cols, int Options, int MaxRows, int MaxCols>
        flatbuffers::Offset<MatrixF> toFbs(flatbuffers::FlatBufferBuilder &fbb, const Eigen::Matrix<float, Rows, Cols, Options, MaxRows, MaxCols> &m)
        {
            flatbuffers::Offset< flatbuffers::Vector<float> > od = fbb.CreateVector(m.array().data(), m.array().size());
            
            MatrixFBuilder mb(fbb);
            mb.add_rows(static_cast<int>(m.rows()));
            mb.add_cols(static_cast<int>(m.cols()));
            mb.add_data(od);
            
            return mb.Finish();
        }
        
        /**
            Convert generic matrix to flatbuffers.
        */
        template<int Rows, int Cols, int Options, int MaxRows, int MaxCols>
        flatbuffers::Offset<MatrixI> toFbs(flatbuffers::FlatBufferBuilder &fbb, const Eigen::Matrix<int, Rows, Cols, Options, MaxRows, MaxCols> &m)
        {
            flatbuffers::Offset< flatbuffers::Vector<int> > od = fbb.CreateVector(m.array().data(), m.array().size());
            
            MatrixIBuilder mb(fbb);
            mb.add_rows(static_cast<int>(m.rows()));
            mb.add_cols(static_cast<int>(m.cols()));
            mb.add_data(od);
            
            return mb.Finish();
        }
        
        /**
            Convert flatbuffers to generic matrix.
        */
        template<int Rows, int Cols, int Options, int MaxRows, int MaxCols>
        void fromFbs(const MatrixF &fbsValue, Eigen::Matrix<float, Rows, Cols, Options, MaxRows, MaxCols> &m)
        {
            typedef Eigen::Matrix<float, Rows, Cols, Options, MaxRows, MaxCols> MatrixType;
            
            Eigen::Map<const MatrixType> map(fbsValue.data()->data(), fbsValue.rows(), fbsValue.cols());
            m = map;
        }
        
        /**
            Convert flatbuffers to generic matrix.
        */
        template<int Rows, int Cols, int Options, int MaxRows, int MaxCols>
        void fromFbs(const MatrixI &fbsValue, Eigen::Matrix<int, Rows, Cols, Options, MaxRows, MaxCols> &m)
        {
            typedef Eigen::Matrix<int, Rows, Cols, Options, MaxRows, MaxCols> MatrixType;
            
            Eigen::Map<const MatrixType> map(fbsValue.data()->data(), fbsValue.rows(), fbsValue.cols());
            m = map;
        }


        
    }
}

#endif