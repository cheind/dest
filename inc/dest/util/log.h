/**
    This file is part of Deformable Shape Tracking (DEST).

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/

#ifndef DEST_LOG_H
#define DEST_LOG_H

#include <dest/core/config.h>


#ifdef DEST_VERBOSE
    #include <iostream>
    #include <iomanip>
    #define DEST_LOG(x) do { std::cout << x; } while (0)
    #define DEST_LOG_MARK DEST_LOG(__FILE__ << " : " << __LINE__ << std::endl)
#else
    #define DEST_LOG(x)
    #define DEST_LOG_MARK DEST_LOG
#endif

#endif