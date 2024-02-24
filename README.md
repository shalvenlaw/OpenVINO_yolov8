# OpenVINO_yolov8
用OpenVINO对yolov8导出的onnx模型进行C++的推理. 任务包括图像分类, 目标检测, 实例分割, 人体姿态检测, 和旋转矩形框目标预测. 步骤包括图片前处理, 推理, NMS等.

### 编译环境
- Windows 10
- Qt 6.2.4 MSVC2019 64bit
- OpenCV 4.8.0
- OpenVINO 2023.2

### 各个项目简介
- CppQt_OpenVINO_YOLOv8_Image_Classification: 图像分类
- CppQt_OpenVINO_YOLOv8_Image_Detection: 目标检测
- CppQt_OpenVINO_YOLOv8_Image_Segmentation: 实例分割
- CppQt_OpenVINO_YOLOv8_Image_Pose: 人体姿态检测
- CppQt_OpenVINO_YOLOv8_Image_OBB: 旋转矩形框目标预测

### 预告
- 新建一个分支使用CMake代替QMake
- 将[openvino_notebooks](https://github.com/openvinotoolkit/openvino_notebooks)中的所有模型都用C++推理一遍
