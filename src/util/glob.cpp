/**
    This file is part of Deformable Shape Tracking (DEST).

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/

#include <dest/util/glob.h>
#include <tinydir/tinydir.h>
#include <stack>

namespace dest {
    namespace util {
        
        std::vector<std::string> findFilesInDir(const std::string &directory, const std::string &extension, bool stripExtension, bool recursive)
        {
            std::vector<std::string> files;
            
            std::stack<std::string> dirsLeft;
            dirsLeft.push(directory);

            tinydir_dir dir;
            while (!dirsLeft.empty()) {
                std::string dirPath = dirsLeft.top(); dirsLeft.pop();

                // Try to open directory
                if (tinydir_open_sorted(&dir, dirPath.c_str()) != 0)
                    continue;
                
                for (unsigned i = 0; i < dir.n_files; i++) {
                    tinydir_file file;

                    if (tinydir_readfile_n(&dir, &file, i) != 0) {
                        continue;
                    }

                    if (file.is_dir) {
                        if (recursive && file.name != std::string(".") && file.name != std::string("..")) {
                            dirsLeft.push(file.path);
                        }
                        continue;
                    }

                    if (extension != file.extension) {
                        continue;
                    }

                    std::string path(file.path);

                    if (stripExtension) {
                        size_t lastindex = path.find_last_of(".");
                        std::string raw = path.substr(0, lastindex);
                        files.push_back(raw);
                    }
                    else {
                        files.push_back(path);
                    }
                }

                tinydir_close(&dir);
            }
            
            return files;
            
        }
        
    }
}