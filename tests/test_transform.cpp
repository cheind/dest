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