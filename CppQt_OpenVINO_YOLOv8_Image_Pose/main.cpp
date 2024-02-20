#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include "samples/args_helper.hpp"
#include "samples/common.hpp"
#include "samples/classification_results.h"
#include "format_reader_ptr.h"

// COCO数据集的标签
static const std::vector<std::string> class_names = {
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

// 模型文件路径
static const std::string model_file = "../model/yolov8n-pose.onnx";
// 测试图片路径
//static const std::string image_file = "../data/zidane.jpg";
//static const std::string image_file = "../data/dog_512.bmp";
//static const std::string image_file = "../data/car.bmp";
static const std::string image_file = "../data/bus.jpg";

// 定义skeleton的连接关系以及color mappings
// skeleton[0]={16, 14}表示objects_keypoints[16]和objects_keypoints[14]连一条线, posePalette[limbColorIndices[0]]表示这条线的颜色
static const std::vector<std::vector<int>> skeleton = { {16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12}, {7, 13}, {6, 7},
    {6, 8}, {7, 9}, {8, 10}, {9, 11}, {2, 3}, {1, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}, {5, 7}
};
static const std::vector<cv::Scalar> posePalette = {
    cv::Scalar(255, 128, 0), cv::Scalar(255, 153, 51), cv::Scalar(255, 178, 102), cv::Scalar(230, 230, 0), cv::Scalar(255, 153, 255),
    cv::Scalar(153, 204, 255), cv::Scalar(255, 102, 255), cv::Scalar(255, 51, 255), cv::Scalar(102, 178, 255), cv::Scalar(51, 153, 255),
    cv::Scalar(255, 153, 153), cv::Scalar(255, 102, 102), cv::Scalar(255, 51, 51), cv::Scalar(153, 255, 153), cv::Scalar(102, 255, 102),
    cv::Scalar(51, 255, 51), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255), cv::Scalar(255, 0, 0), cv::Scalar(255, 255, 255)
};
static const std::vector<int> limbColorIndices = { 9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16 };
static const std::vector<int> kptColorIndices = { 16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9 };


/// 转换图像数据: 先转换元素类型, (可选)然后归一化到[0, 1], (可选)然后交换RB通道
void convert(const cv::Mat &input, cv::Mat &output, const bool normalize, const bool exchangeRB)
{
    input.convertTo(output, CV_32F);
    if (normalize) {
        output = output / 255.0; // 归一化到[0, 1]
    }
    if (exchangeRB) {
        cv::cvtColor(output, output, cv::COLOR_BGR2RGB);
    }
}

/*!
 * \brief fill_tensor_data_image 对网络的输入为图片数据的节点进行赋值，实现图片数据输入网络
 * \param input_tensor 输入节点的tensor
 * \param input_image 输入图片的数据
 * \return 缩放因子, 该缩放是为了将input_image塞进input_tensor
 */
float fill_tensor_data_image(ov::Tensor &input_tensor, const cv::Mat &input_image)
{
    /// letterbox变换: 不改变宽高比(aspect ratio), 将input_image缩放并放置到blob_image左上角
    const ov::Shape tensor_shape = input_tensor.get_shape();
    const size_t num_channels = tensor_shape[1];
    const size_t height = tensor_shape[2];
    const size_t width = tensor_shape[3];
    // 缩放因子
    const float scale = std::min(height / float(input_image.rows),
                                 width / float(input_image.cols));
    const cv::Matx23f matrix{
        scale, 0.0, 0.0,
        0.0, scale, 0.0,
    };
    cv::Mat blob_image;
    // 下面根据scale范围进行数据转换, 这只是为了提高一点速度(主要是提高了交换通道的速度)
    // 如果不在意这点速度提升的可以固定一种做法(两个if分支随便一个都可以)
    if (scale < 1.0 - FLT_EPSILON) {
        // 要缩小, 那么先缩小再交换通道
        cv::warpAffine(input_image, blob_image, matrix, cv::Size(width, height));
        convert(blob_image, blob_image, true, true);
    } else {
        // 要放大, 那么先交换通道再放大
        convert(input_image, blob_image, true, true);
        cv::warpAffine(blob_image, blob_image, matrix, cv::Size(width, height));
    }
//    cv::imshow("input_image", input_image);
//    cv::imshow("blob_image", blob_image);
//    cv::waitKey(0);

    /// 将图像数据填入input_tensor
    float *const input_tensor_data = input_tensor.data<float>();
    // 原有图片数据为 HWC格式，模型输入节点要求的为 CHW 格式
    for (size_t c = 0; c < num_channels; c++) {
        for (size_t h = 0; h < height; h++) {
            for (size_t w = 0; w < width; w++) {
                input_tensor_data[c * width * height + h * width + w] = blob_image.at<cv::Vec<float, 3>>(h, w)[c];
            }
        }
    }
    return 1 / scale;
}

// 打印模型信息, 这个函数是从$${OPENVINO_COMMON}/utils/src/args_helper.cpp复制过来的
void printInputAndOutputsInfo(const ov::Model &network)
{
    slog::info << "model name: " << network.get_friendly_name() << slog::endl;

    const std::vector<ov::Output<const ov::Node>> inputs = network.inputs();
    for (const ov::Output<const ov::Node> &input : inputs) {
        slog::info << "    inputs" << slog::endl;

        const std::string name = input.get_names().empty() ? "NONE" : input.get_any_name();
        slog::info << "        input name: " << name << slog::endl;

        const ov::element::Type type = input.get_element_type();
        slog::info << "        input type: " << type << slog::endl;

        const ov::Shape shape = input.get_shape();
        slog::info << "        input shape: " << shape << slog::endl;
    }

    const std::vector<ov::Output<const ov::Node>> outputs = network.outputs();
    for (const ov::Output<const ov::Node> &output : outputs) {
        slog::info << "    outputs" << slog::endl;

        const std::string name = output.get_names().empty() ? "NONE" : output.get_any_name();
        slog::info << "        output name: " << name << slog::endl;

        const ov::element::Type type = output.get_element_type();
        slog::info << "        output type: " << type << slog::endl;

        const ov::Shape shape = output.get_shape();
        slog::info << "        output shape: " << shape << slog::endl;
    }
}

int main(int argc, char **argv)
{
    try {
        /// 创建OpenVINO Runtime Core对象
        ov::Core core;
        std::shared_ptr<ov::Model> model = core.read_model(model_file);
        printInputAndOutputsInfo(*model); // 打印模型信息

//        return 0;

        /// 载入并编译模型
        ov::CompiledModel compiled_model = core.compile_model(model, "AUTO");

        /// 创建推理请求
        ov::InferRequest infer_request = compiled_model.create_infer_request();

        /// 设置模型输入
        // 获取模型输入节点
        ov::Tensor input_tensor = infer_request.get_input_tensor();

        const int64 start = cv::getTickCount();
        // 读取图片并按照模型输入要求进行预处理
        cv::Mat image = cv::imread(image_file, cv::IMREAD_COLOR);
        const float factor = fill_tensor_data_image(input_tensor, image);

        /// 执行推理计算
        infer_request.infer();

        /// 处理推理计算结果
        // 获得推理结果
        const ov::Tensor output = infer_request.get_output_tensor();
        const ov::Shape output_shape = output.get_shape();
        const float *output_buffer = output.data<const float>();

        // 解析推理结果
        const int out_rows = output_shape[1]; //获得"output"节点的rows
        const int out_cols = output_shape[2]; //获得"output"节点的cols
        const cv::Mat det_output(out_rows, out_cols, CV_32F, (float *)output_buffer);

        std::vector<cv::Rect> boxes;
        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<std::vector<float>> objects_keypoints;
        // 输出格式是[56,8400], 每列代表一个框(即最多有8400个框), 前面4行分别是[cx, cy, ow, oh], 中间score, 最后17*3关键点(3代表每个关键点的信息, 包括[x, y, visibility])
        // 56=4+1+17*3
        std::cout << std::endl << std::endl;
        for (int i = 0; i < det_output.cols; ++i) {
            const float score = det_output.at<float>(4, i);
            // 置信度 0～1之间
            if (score > 0.3f) {
                const float cx = det_output.at<float>(0, i);
                const float cy = det_output.at<float>(1, i);
                const float ow = det_output.at<float>(2, i);
                const float oh = det_output.at<float>(3, i);
                cv::Rect box;
                box.x = static_cast<int>((cx - 0.5 * ow) * factor);
                box.y = static_cast<int>((cy - 0.5 * oh) * factor);
                box.width = static_cast<int>(ow * factor);
                box.height = static_cast<int>(oh * factor);

                boxes.push_back(box);
                class_ids.push_back(0); //{0:"person"} 姿态估计只有人这一类
                confidences.push_back(score);

                // 获取关键点
                std::vector<float> keypoints;
                cv::Mat kpts = det_output.col(i).rowRange(5, 56);
                for (int j = 0; j < 17; j++) {
                    float x = kpts.at<float>(j * 3 + 0, 0) * factor;
                    float y = kpts.at<float>(j * 3 + 1, 0) * factor;
                    float s = kpts.at<float>(j * 3 + 2, 0);
                    keypoints.push_back(x);
                    keypoints.push_back(y);
                    keypoints.push_back(s);
                }
                objects_keypoints.push_back(keypoints);
            }
        }

        const int radius = 5;
        const cv::Size &shape = image.size();
        std::vector<cv::Scalar> limbColorPalette;
        std::vector<cv::Scalar> kptColorPalette;
        for (int index : limbColorIndices) {
            limbColorPalette.push_back(posePalette[index]);
        }
        for (int index : kptColorIndices) {
            kptColorPalette.push_back(posePalette[index]);
        }

        // NMS, 消除具有较低置信度的冗余重叠框
        std::vector<int> indexes;
        cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indexes);
        for (size_t i = 0; i < indexes.size(); i++) {
            const int index = indexes[i];
            const int idx = class_ids[index];
            cv::rectangle(image, boxes[index], cv::Scalar(0, 0, 255), 2, 8);
            cv::rectangle(image, cv::Point(boxes[index].tl().x, boxes[index].tl().y - 20),
                          cv::Point(boxes[index].br().x, boxes[index].tl().y), cv::Scalar(0, 255, 255), -1);
            cv::putText(image, class_names[idx] + " " + std::to_string(confidences[index]), cv::Point(boxes[index].tl().x, boxes[index].tl().y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0, 0, 0));

            // 绘制关键点和连线
            const std::vector<float> &keypoint = objects_keypoints[index];
            // draw points
            for (int i = 0; i < 17; i++) {
                int idx = i * 3;
                int x_coord = static_cast<int>(keypoint[idx]);
                int y_coord = static_cast<int>(keypoint[idx + 1]);

                if ((x_coord % shape.width) != 0 && (y_coord % shape.height) != 0) {
                    const float conf = keypoint[2];
                    if (conf < 0.5) {
                        continue;
                    }
                    cv::circle(image, cv::Point(x_coord, y_coord), radius, kptColorPalette[i], -1, cv::LINE_AA);
                }
            }
            // draw lines
            for (int i = 0; i < skeleton.size(); i++) {
                const std::vector<int> &sk = skeleton[i];
                int idx1 = sk[0] - 1;
                int idx2 = sk[1] - 1;

                int idx1_x_pos = idx1 * 3;
                int idx2_x_pos = idx2 * 3;

                int x1 = static_cast<int>(keypoint[idx1_x_pos]);
                int y1 = static_cast<int>(keypoint[idx1_x_pos + 1]);
                int x2 = static_cast<int>(keypoint[idx2_x_pos]);
                int y2 = static_cast<int>(keypoint[idx2_x_pos + 1]);

                float conf1 = keypoint[idx1_x_pos + 2];
                float conf2 = keypoint[idx2_x_pos + 2];

                // Check confidence thresholds
                if (conf1 < 0.5 || conf2 < 0.5) {
                    continue;
                }

                // Check if positions are within bounds
                if (x1 % shape.width == 0 || y1 % shape.height == 0 || x1 < 0 || y1 < 0 ||
                        x2 % shape.width == 0 || y2 % shape.height == 0 || x2 < 0 || y2 < 0) {
                    continue;
                }
                // Draw a line between keypoints
                cv::line(image, cv::Point(x1, y1), cv::Point(x2, y2), limbColorPalette[i], 2, cv::LINE_AA);
            }
        }

        // 计算FPS
        const float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
        std::cout << "Infer time(ms): " << t * 1000 << "ms; Detections: " << indexes.size() << std::endl;
        cv::putText(image, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
        cv::imshow("CppQt_OpenVINO_YOLOv8_Image_Pose", image);

        cv::waitKey(0);
        cv::destroyAllWindows();

    } catch (const std::exception &e) {
        std::cerr << "exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "unknown exception" << std::endl;
    }
    return 0;
}
