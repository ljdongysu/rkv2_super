cmake_minimum_required(VERSION 3.10)
project(rkv2_super)

set(CMAKE_CXX_STANDARD 14)

# opencv
find_package(OpenCV REQUIRED)

# npu
include_directories(/usr/include/)
include_directories(/root/ljdong/yolo_Deepsort_rknn/include)
# /root/ljdong/yolo_Deepsort_rknn/include/rknn_fp.h

add_executable(rkv2_super rknn_fp.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} )
target_link_libraries(${PROJECT_NAME}  /usr/lib/librknn_api.so)
target_link_libraries(${PROJECT_NAME}  /usr/lib/librknnrt.so)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} -lrknnrt -lrga -lpthread)