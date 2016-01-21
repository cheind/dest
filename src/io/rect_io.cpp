/**
    This file is part of Deformable Shape Tracking (DEST).

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/

#include <dest/io/rect_io.h>
#include <fstream>

namespace dest {
    namespace io {
        
        bool importRectangles(const std::string &pathToCSV, std::vector<core::Rect> &rects) {
            std::ifstream file(pathToCSV);
            if (!file.is_open())
                return false;
            
            std::string line;
            while (std::getline(file, line)) {
                
                if (line.empty())
                    break;
                
                core::Rect r(2, 4);
                std::istringstream str(line);
                str >> r(0, 0) >> r(0, 1) >> r(0, 2) >> r(0, 3) >> r(1, 0) >> r(1, 1) >> r(1, 2) >> r(1, 3);
                rects.push_back(r);
            }
            
            return true;
        }
        
        bool exportRectangles(const std::string &pathToCSV, const std::vector<core::Rect> &rects) {
            std::ofstream ofs(pathToCSV);
            if (!ofs.is_open())
                return false;
            
            Eigen::IOFormat csvFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", " ");
            for(size_t i = 0; i < rects.size(); ++i) {
                ofs << rects[i].format(csvFormat) << std::endl;
            }
            
            ofs.close();
            return true;
        }
    }
}