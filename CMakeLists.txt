cmake_minimum_required(VERSION 3.0.0)
set(PROJECT_NAME rkv2_super)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 14)

# opencv
find_package(OpenCV REQUIRED)

# npu
include_directories(/usr/include/)
include_directories(/root/ljdong/yolo_Deepsort_rknn/include)
# /root/ljdong/yolo_Deepsort_rknn/include/rknn_fp.h

add_executable(${PROJECT_NAME} rknn_fp.cpp SpRun.cpp SpRun.h)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} )

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} /root/runtime/RK356X/Linux/librknn_api/aarch64/librknnrt.so /root/runtime/RK356X/Linux/librknn_api/aarch64/librknn_api.so  /root/ljdong/RK3566_lib/rga/lib/librga.so -lpthread)