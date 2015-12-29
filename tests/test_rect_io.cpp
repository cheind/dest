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

#include "catch.hpp"

#include <dest/core/shape.h>
#include <dest/face/database_importers.h>
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
    
    REQUIRE(dest::face::exportRectangles("rects.csv", rects));
    rects.clear();
    REQUIRE(dest::face::importRectangles("rects.csv", rects));
    REQUIRE(rects.size() == 2);
    
    REQUIRE(r0.isApprox(rects[0]));
    REQUIRE(r1.isApprox(rects[1]));
}