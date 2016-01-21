/**
This file is part of Deformable Shape Tracking (DEST).

Copyright(C) 2015/2016 Christoph Heindl
All rights reserved.

This software may be modified and distributed under the terms
of the BSD license.See the LICENSE file for details.
*/

#include "catch.hpp"

#include <dest/core/shape.h>
#include <dest/io/rect_io.h>
#include <iostream>

TEST_CASE("rect-io")
{
    std::vector<dest::core::Rect> rects;
    dest::core::Rect r0;
    r0 << 0.f, 1.f, 2.f, 3.f,
          5.f, 6.f, 7.f, 8.f;
    rects.push_back(r0);
    
    dest::core::Rect r1;
    r1 << 10.f, 11.f, 12.f, 13.f,
          15.f, 16.f, 17.f, 18.f;
    rects.push_back(r1);
    
    REQUIRE(dest::io::exportRectangles("rects.csv", rects));
    rects.clear();
    REQUIRE(dest::io::importRectangles("rects.csv", rects));
    REQUIRE(rects.size() == 2);
    
    REQUIRE(r0.isApprox(rects[0]));
    REQUIRE(r1.isApprox(rects[1]));
}