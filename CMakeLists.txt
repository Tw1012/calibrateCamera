cmake_minimum_required(VERSION 3.0.0)

project(calibrateCamera VERSION 0.1.0)

#找opencv库
find_package(OpenCV 4.2.0 REQUIRED)
#添加头文件
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(calibrateCamera calibrateCamera.cpp)

#连接OpenCV库
target_link_libraries(calibrateCamera ${OpenCV_LIBS})
