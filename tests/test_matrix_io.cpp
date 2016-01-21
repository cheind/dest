/**
This file is part of Deformable Shape Tracking (DEST).

Copyright(C) 2015/2016 Christoph Heindl
All rights reserved.

This software may be modified and distributed under the terms
of the BSD license.See the LICENSE file for details.
*/

#include "catch.hpp"

#include <dest/core/shape.h>
#include <dest/io/matrix_io.h>

TEST_CASE("matrix")
{
    dest::core::Shape s = dest::core::Shape::Random(2, 4);
    
    flatbuffers::FlatBufferBuilder fbb;
    auto off = dest::io::toFbs(fbb, s);
}