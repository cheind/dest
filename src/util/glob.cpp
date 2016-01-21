/**
    This file is part of Deformable Shape Tracking (DEST).

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
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