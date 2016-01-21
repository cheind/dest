/**
    This file is part of Deformable Shape Tracking (DEST).

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/

#ifndef DEST_GLOB_H
#define DEST_GLOB_H

#include <vector>
#include <string>

namespace dest {
    namespace util {
        
        /**
            Find all files in directory with options.

            \param Directory to search in.
            \param extension Necessary file extension.
            \param stripExtension Whether or not to strip extension in results.
            \param recursive Traverse sub-directories too.
            \returns List of found files.
        */
        std::vector<std::string> findFilesInDir(const std::string &directory, const std::string &extension, bool stripExtension, bool recursive);
        
    }
}

#endif