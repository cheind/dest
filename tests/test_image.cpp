/**
    This file is part of Deformable Shape Tracking (DEST).

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
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