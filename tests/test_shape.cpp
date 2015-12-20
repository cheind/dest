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