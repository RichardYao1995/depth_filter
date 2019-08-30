
cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "-std=c++11 -O3")

# Define project name
project(depth_filter)

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI

FIND_PACKAGE(RTABMap REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL 1.7 REQUIRED)
include_directories(${PROJECT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
include_directories(${RTABMap_INCLUDE_DIRS})

file(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/*.cpp)

## Declare the executable target built from your sources
add_executable(main ${SRC_FILES})

# Link your application with OpenCV libraries
target_link_libraries(main ${OpenCV_LIBS} ${PCL_LIBRARIES} ${RTABMap_LIBRARIES})

