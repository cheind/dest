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

#include <dest/util/glob.h>
#include <tinydir/tinydir.h>

namespace dest {
    namespace util {
        
        std::vector<std::string> findFilesInDir(const std::string &directory, const std::string &extension, bool stripExtension)
        {
            std::vector<std::string> files;
            
            tinydir_dir dir;
            if (tinydir_open(&dir, directory.c_str()) != 0)
                return files;
            
            while (dir.has_next) {
                tinydir_file file;
                
                if (tinydir_readfile(&dir, &file) != 0 || file.is_dir) {
                    tinydir_next(&dir);
                    continue;
                }
                
                if (extension != file.extension) {
                    tinydir_next(&dir);
                    continue;

                }
                
                std::string path(file.path);
                
                if (stripExtension) {
                    size_t lastindex = path.find_last_of(".");
                    std::string raw = path.substr(0, lastindex);
                    files.push_back(raw);
                } else {
                    files.push_back(path);
                }
                
                tinydir_next(&dir);
            }
            
            tinydir_close(&dir);
            return files;
            
        }
        
    }
}