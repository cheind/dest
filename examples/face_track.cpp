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

#include <dest/dest.h>

#include <dest/face/database_importers.h>
#include <dest/util/draw.h>
#include <random>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv)
{
    std::vector<dest::core::Image> images;
    std::vector<dest::core::Shape> shapes;

    dest::face::importIMMFaceDatabase(argv[1], images, shapes);

    dest::core::Tracker t;
    if (!t.load(argv[2])) {
        std::cout << "Failed to load tracker." << std::endl;
        return false;
    }

    std::mt19937 rnd;
    std::uniform_int_distribution<int> d(0, (int)images.size() - 1);
    std::normal_distribution<float> dg(0, 2.f);

    bool done = false;
    while (!done) {
        int imageId = d(rnd);
        int shapeId = d(rnd);

        dest::core::Shape s = shapes[shapeId];
        dest::core::Image i = images[imageId];

        dest::core::Shape spred = t.predict(i, s);

        cv::Mat img = dest::util::drawShape(i, s, cv::Scalar(0, 0, 255));
        dest::util::drawShape(img, spred, cv::Scalar(0, 255, 0));

        cv::imshow("prediction", img);
        int key = cv::waitKey();
        if (key == 'x')
            done = true;
    }


    
    return 0;
}
