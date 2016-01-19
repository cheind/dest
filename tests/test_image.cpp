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

#include <dest/core/image.h>

TEST_CASE("image-readpixels")
{
    dest::core::Image img(2, 2);
    img << 0, 64,
           128, 255;
    
    dest::core::PixelCoordinates coords(2, 6);
    coords << -1.f, 0.f, 0.f, 0.5f, 0.5f, 2.f,
              -1.f, 0.f, 0.5f, 0.0f, 0.5f, 2.f;
    
    dest::core::PixelIntensities expected(6);
    expected << 0.f, 0.f, 64.f, 32.f, 111.75f, 255.f;
    
    dest::core::PixelIntensities intensities;
    dest::core::readImage(img, coords, intensities);
    
    REQUIRE(intensities.isApprox(expected));

}