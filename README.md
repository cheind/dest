## About this library

**Deformable Shape Tracking (DEST)** is a C++ library providing high performance 2D shape tracking utilizing
machine learning methods. The video below shows the real-time capabilities of DEST in annotating video sequences
or still images using with facial landmarks. The **DEST** tracker was previously trained on thousands of
training samples from available face databases.

https://youtu.be/Hewjc0oyqPQ

## Building from source
**Deformable Shape Tracking** requires the following pre-requisites

 - [CMake](www.cmake.org) - for generating cross platform build files
 - [OpenCV 2.x / 3.x](www.opencv.org) - for image processing related functions

To build from source

 1. Point CMake to the cloned git repository
 1. Click CMake Configure
 1. Point `OpenCV_DIR` to the directory containing the file `OpenCVConfig.cmake`
 1. Click CMake Generate

Although **Deformable Shape Tracking** should build across multiple platforms and architectures, tests are carried out on these systems
 - Windows 8/10 MSVC10 / MSVC12 x64
 - OS X 10.10 XCode 7.x x64

If the build should fail for a specific platform, don't hesitate to create an issue.

# References

 1. <a name="Kazemi14"></a>Kazemi, Vahdat, and Josephine Sullivan. "One millisecond face alignment with an ensemble of regression trees." Computer Vision and Pattern Recognition (CVPR), 2014 IEEE Conference on. IEEE, 2014.

# License

```
This file is part of Deformable Shape Tracking.

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
along with Deformable Shape Tracking.  If not, see <http://www.gnu.org/licenses/>.
```
