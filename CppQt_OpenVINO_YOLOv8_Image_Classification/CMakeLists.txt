cmake_minimum_required(VERSION 3.5)

project(CppQt_OpenVINO_YOLOv8_Image_Classification LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(${PROJECT_NAME} main.cpp)

# OpenCV
# 下载链接为https://github.com/opencv/opencv/releases/download/4.8.0/opencv-4.8.0-windows.exe
set(OpenCV_DIR G:/OpenSource/OpenCV/4_8_0/build)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# OpenVINO
# 下载链接为https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.2/windows/w_openvino_toolkit_windows_2023.2.0.13089.cfd42bd2cb0_x86_64.zip
set(OpenVINO_RUNTIME G:/OpenSource/OpenVINO/2023/w_openvino_toolkit_windows_2023.2.0.13089.cfd42bd2cb0_x86_64/runtime)
set(OpenVINO_DIR ${OpenVINO_RUNTIME}/cmake)
find_package(OpenVINO REQUIRED)
include_directories(${OpenVINO_RUNTIME}/include)
find_library(OpenVINO_LIBS NAMES openvino PATHS ${OpenVINO_RUNTIME}/lib/intel64/Release)
target_link_libraries(${PROJECT_NAME} ${OpenVINO_LIBS})


#设置可执行文件输出路径
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../bin)

#install(TARGETS ${PROJECT_NAME}
#    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

