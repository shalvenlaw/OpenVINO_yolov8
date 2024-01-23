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
# 链接为https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.2/windows/w_openvino_toolkit_windows_2023.2.0.13089.cfd42bd2cb0_x86_64.zip
OPENVINO_DIR = G:/OpenSource/OpenVINO/2023/w_openvino_toolkit_windows_2023.2.0.13089.cfd42bd2cb0_x86_64
OPENVINO_RUNTIME = $${OPENVINO_DIR}/runtime
OPENVINO_COMMON = $${OPENVINO_DIR}/samples/cpp/common
CONFIG(release, debug|release): LIBS += -L$${OPENVINO_RUNTIME}/lib/intel64/release/ \
    -lopenvino \
    -lopenvino_paddle_frontend \
#    -lopenvino_ir_frontend \
    -lopenvino_onnx_frontend
else:CONFIG(debug, debug|release): LIBS += -L$${OPENVINO_RUNTIME}/lib/intel64/debug/ \
    -lopenvinod \
    -lopenvino_paddle_frontendd \
#    -lopenvino_ir_frontendd \
    -lopenvino_onnx_frontendd
INCLUDEPATH += \
    $${OPENVINO_RUNTIME}/include \
    $${OPENVINO_RUNTIME}/include/ie \
    $${OPENVINO_RUNTIME}/include/ngraph \
    $${OPENVINO_RUNTIME}/include/openvino \
    $${OPENVINO_COMMON}/format_reader/include \
    $${OPENVINO_COMMON}/utils/include
HEADERS += \
    $${OPENVINO_COMMON}/format_reader/MnistUbyte.h \
    $${OPENVINO_COMMON}/format_reader/bmp.h \
    $${OPENVINO_COMMON}/format_reader/format_reader.h \
    $${OPENVINO_COMMON}/format_reader/format_reader_ptr.h \
    $${OPENVINO_COMMON}/format_reader/opencv_wrapper.h \
    $${OPENVINO_COMMON}/format_reader/register.h \
    $${OPENVINO_COMMON}/format_reader/yuv_nv12.h \
#    $${OPENVINO_COMMON}/utils/include/samples/args_helper.hpp \
    $${OPENVINO_COMMON}/utils/include/samples/classification_results.h \
    $${OPENVINO_COMMON}/utils/include/samples/common.hpp \
    $${OPENVINO_COMMON}/utils/include/samples/console_progress.hpp \
    $${OPENVINO_COMMON}/utils/include/samples/csv_dumper.hpp \
    $${OPENVINO_COMMON}/utils/include/samples/ocv_common.hpp \
    $${OPENVINO_COMMON}/utils/include/samples/os/windows/w_dirent.h \
    $${OPENVINO_COMMON}/utils/include/samples/slog.hpp \
    $${OPENVINO_COMMON}/utils/include/samples/vpu/vpu_tools_common.hpp
SOURCES += \
    $${OPENVINO_COMMON}/format_reader/src/MnistUbyte.cpp \
    $${OPENVINO_COMMON}/format_reader/src/bmp.cpp \
    $${OPENVINO_COMMON}/format_reader/src/format_reader.cpp \
    $${OPENVINO_COMMON}/format_reader/src/npy.cpp \
    $${OPENVINO_COMMON}/format_reader/src/opencv_wrapper.cpp \
    $${OPENVINO_COMMON}/format_reader/src/yuv_nv12.cpp \
#    $${OPENVINO_COMMON}/utils/src/args_helper.cpp \
#    $${OPENVINO_COMMON}/utils/src/common.cpp \
    $${OPENVINO_COMMON}/utils/src/slog.cpp

## 指定exe或dll的生成目录
DESTDIR = ../bin




