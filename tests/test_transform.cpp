/**
This file is part of Deformable Shape Tracking (DEST).

Copyright(C) 2015/2016 Christoph Heindl
All rights reserved.

This software may be modified and distributed under the terms
of the BSD license.See the LICENSE file for details.
*/

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <dest/core/shape.h>
#include <Eigen/Geometry>

TEST_CASE("similarity-transform-translate")
{
    dest::core::Shape to(2, 4);
    to << 0.f, 2.f, 2.f, 0.f,
          0.f, 0.f, 2.f, 2.f;
    
    Eigen::AffineCompact2f t;
    t = Eigen::Translation2f(1.f, 1.f);
    
    dest::core::Shape from = t.matrix() * to.colwise().homogeneous();
    
    Eigen::AffineCompact2f s = dest::core::estimateSimilarityTransform(from, to);
    
    Eigen::AffineCompact2f expected;
    expected = Eigen::Translation2f(-1.f, -1.f);

    REQUIRE(s.isApprox(expected));

}

TEST_CASE("similarity-transform-compound")
{
    dest::core::Shape to(2, 4);
    to << 0.f, 2.f, 2.f, 0.f,
    0.f, 0.f, 2.f, 2.f;
    
    Eigen::AffineCompact2f t;
    t = Eigen::Translation2f(1.f, 1.f) * Eigen::Rotation2Df(0.17f) * Eigen::Scaling(1.8f);
    
    dest::core::Shape from = t.matrix() * to.colwise().homogeneous();
    
    Eigen::AffineCompact2f s = dest::core::estimateSimilarityTransform(from, to);
    
    Eigen::AffineCompact2f expected = t.inverse();
    
    REQUIRE(s.isApprox(expected));    
}


TEST_CASE("similarity-transform-between-rects")
{
    dest::core::Rect r = dest::core::createRectangle(Eigen::Vector2f(-2.f, -2.f), Eigen::Vector2f(2.f, 2.f));

    Eigen::AffineCompact2f t;
    t = Eigen::Rotation2Df(0.17f);

    r = t.matrix() * r.colwise().homogeneous();

    dest::core::Rect n = dest::core::unitRectangle();
    Eigen::AffineCompact2f s = dest::core::estimateSimilarityTransform(r, n);

    r = s.matrix() * r.colwise().homogeneous();
    REQUIRE(r.isApprox(n));
}