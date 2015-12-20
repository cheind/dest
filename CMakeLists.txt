cmake_minimum_required(VERSION 2.8)

project(deformable-shape-tracking)

if (WIN32)
	add_definitions("-D_SCL_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_WARNINGS")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LIBRARY "libc++")
endif()

set(DEST_LINK_TARGETS)

set(DEST_EIGEN_DIR "../eigen" CACHE PATH "Where is the include directory of Eigen located")
set(DEST_WITH_OPENCV OFF CACHE BOOL "Build DEST with OpenCV support")
if(DEST_WITH_OPENCV)
	if (WIN32)
    	set(OpenCV_STATIC OFF)
		set(OpenCV_SHARED ON)
	endif()
	find_package(OpenCV REQUIRED)
	include_directories(${OpenCV_INCLUDE_DIRS})
	list(APPEND DEST_LINK_TARGETS ${OpenCV_LIBRARIES})
endif()

include_directories(${CMAKE_CURRENT_BINARY_DIR} ${DEST_EIGEN_DIR} "inc")

# Library
set(DEST_VERBOSE ON CACHE BOOL "Build DEST in verbose mode.")
configure_file(inc/dest/core/config.h.in dest/core/config.h)
add_library(dest
    inc/dest/dest.h
    inc/dest/core/config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/dest/core/config.h
    inc/dest/core/shape.h
    inc/dest/core/image.h
    inc/dest/core/training_data.h
    inc/dest/core/tracker.h
    inc/dest/core/regressor.h
    inc/dest/core/tree.h
    inc/dest/face/database_importers.h
    inc/dest/io/dest_io.fbs
    inc/dest/io/dest_io_generated.h
    inc/dest/io/matrix_io.h
    inc/dest/util/draw.h
    inc/dest/util/log.h
    src/core/training_data.cpp
    src/core/tracker.cpp
    src/core/regressor.cpp
    src/core/tree.cpp
    src/core/shape.cpp
    src/core/image.cpp
    src/face/database_importers.cpp
    src/util/draw.cpp
)
	
target_link_libraries(dest ${DEST_LINK_TARGETS})
	
# Samples

add_executable(dest_facetrack examples/face_track.cpp)
target_link_libraries(dest_facetrack dest ${DEST_LINK_TARGETS})

# Tests

add_executable(dest_tests
    tests/catch.hpp
    tests/test_transform.cpp
    tests/test_image.cpp
    tests/test_shape.cpp
    tests/test_matrix_io.cpp
)
target_link_libraries(dest_tests dest ${DEST_LINK_TARGETS})