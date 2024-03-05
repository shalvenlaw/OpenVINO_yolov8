TEMPLATE = app
CONFIG += console c++17
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        main.cpp

## OpenCV 4.8.0 world
# 下载链接为https://github.com/opencv/opencv/releases/download/4.8.0/opencv-4.8.0-windows.exe
OPENCV_BUILD = G:/OpenSource/OpenCV/4_8_0/build/install
win32:CONFIG(release, debug|release): LIBS += -L$${OPENCV_BUILD}/x64/vc15/lib/ -lopencv_world480
else:win32:CONFIG(debug, debug|release): LIBS += -L$${OPENCV_BUILD}/x64/vc15/lib/ -lopencv_world480d
INCLUDEPATH += $${OPENCV_BUILD}/include

## OpenVINO 2023.2
# 下载链接为https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.2/windows/w_openvino_toolkit_windows_2023.2.0.13089.cfd42bd2cb0_x86_64.zip
OPENVINO_RUNTIME = G:/OpenSource/OpenVINO/2023/w_openvino_toolkit_windows_2023.2.0.13089.cfd42bd2cb0_x86_64/runtime
CONFIG(release, debug|release): LIBS += -L$${OPENVINO_RUNTIME}/lib/intel64/release/ \
    -lopenvino \
    -lopenvino_onnx_frontend
else:CONFIG(debug, debug|release): LIBS += -L$${OPENVINO_RUNTIME}/lib/intel64/debug/ \
    -lopenvinod \
    -lopenvino_onnx_frontendd
INCLUDEPATH += \
    $${OPENVINO_RUNTIME}/include

## 指定exe或dll的生成目录
DESTDIR = ../bin




