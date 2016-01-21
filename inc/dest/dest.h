/**
    This file is part of Deformable Shape Tracking (DEST).

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/

#ifndef DEST_H
#define DEST_H

#include <dest/core/config.h>
#include <dest/core/shape.h>
#include <dest/core/image.h>
#include <dest/core/tracker.h>
#include <dest/core/training_data.h>
#include <dest/core/tester.h>
#include <dest/io/rect_io.h>

#ifdef DEST_WITH_OPENCV
#include <dest/util/convert.h>
#include <dest/util/draw.h>
#include <dest/io/database_io.h>
#include <dest/face/face_detector.h>
#endif

#endif