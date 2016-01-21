/**
This file is part of Deformable Shape Tracking (DEST).

Copyright(C) 2015/2016 Christoph Heindl
All rights reserved.

This software may be modified and distributed under the terms
of the BSD license.See the LICENSE file for details.
*/

#include "catch.hpp"

#include <dest/core/shape.h>
#include <iostream>


TEST_CASE("shape-relative-coords")
{
    dest::core::Shape s(2, 4);
    s << 0.f, 2.f, 2.f, 0.f,
         0.f, 0.f, 2.f, 2.f;
    
    dest::core::PixelCoordinates abscoords(2, 3);
    abscoords << -0.5f, 1.6f, 3.f,
                 -0.5f, 0.1f, 3.f;
    
    dest::core::PixelCoordinates relcoords;
    Eigen::VectorXi closest;
    
    dest::core::shapeRelativePixelCoordinates(s, abscoords, relcoords, closest);
    
    dest::core::PixelCoordinates expectedRelcoords(2, 3);
    expectedRelcoords << -0.5f, -0.4f, 1.f,
                         -0.5f, 0.1f, 1.f;
    
    Eigen::VectorXi expectedClosest(3);
    expectedClosest << 0, 1, 2;
    
    REQUIRE(relcoords.isApprox(expectedRelcoords));
    REQUIRE(closest.isApprox(expectedClosest));
    
}

TEST_CASE("shape-bounds")
{
    dest::core::Shape s(2, 4);
    s << 0.f, 2.f, 2.f, 0.f,
         0.f, 0.f, 2.f, 2.f;

    dest::core::Rect r = dest::core::shapeBounds(s);
    dest::core::Rect expected = dest::core::createRectangle(Eigen::Vector2f(0.f, 0.f), Eigen::Vector2f(2.f, 2.f));

    REQUIRE(r.isApprox(expected));
}