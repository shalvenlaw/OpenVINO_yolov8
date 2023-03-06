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
static const std::string model_file = "../model/yolov8n.onnx";
// 测试图片路径
//static const std::string image_file = "../data/zidane.jpg";
static const std::string image_file = "../data/dog_512.bmp";
//static const std::string image_file = "../data/car.bmp";
//static const std::string image_file = "../data/bus.jpg";


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
    // 下面根据scale范围进行数据转换, 这只是为了提高一点速度(主要是提高了交换通道的速度), 但可读性很差
    // 如果不在意这点速度提升的可以固定一种做法(前两个if分支都可以)
    if (scale > 1.0 + FLT_EPSILON) {
        // 要放大, 那么先交换通道再放大
        convert(input_image, blob_image, true, true);
        cv::warpAffine(blob_image, blob_image, matrix, cv::Size(width, height));
    } else if (scale < 1.0 - FLT_EPSILON) {
        // 要缩小, 那么先缩小再交换通道
        cv::warpAffine(input_image, blob_image, matrix, cv::Size(width, height));
        convert(blob_image, blob_image, true, true);
    } else {
        convert(input_image, blob_image, true, true);
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

int main(int argc, char **argv)
{
    try {
        /// 创建OpenVINO Runtime Core对象
        ov::Core core;
        std::shared_ptr<ov::Model> model = core.read_model(model_file);
        printInputAndOutputsInfo(*model); // 打印模型信息
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
        // 输出格式是[84,8400], 每列代表一个框(即最多有8400个框), 前面4行分别是cx, cy, ow, oh, 后面80行是每个类别的置信度
        std::cout << std::endl << std::endl;
        for (int i = 0; i < det_output.cols; ++i) {
            const cv::Mat classes_scores = det_output.col(i).rowRange(4, 84);
            cv::Point class_id_point;
            double score;
            cv::minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

            // 置信度 0～1之间
            if (score > 0.3) {
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
                class_ids.push_back(class_id_point.y);
                confidences.push_back(score);
            }
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
            cv::putText(image, class_names[idx], cv::Point(boxes[index].tl().x, boxes[index].tl().y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }

        // 计算FPS
        const float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
        std::cout << "Infer time(ms): " << t * 1000 << "ms; Detections: " << indexes.size() << std::endl;
        cv::putText(image, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
        cv::imshow("CppQt_OpenVINO202203_YOLOv8_Image_Detection", image);

        cv::waitKey(0);
        cv::destroyAllWindows();

    } catch (const std::exception &e) {
        std::cerr << "exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "unknown exception" << std::endl;
    }
    return 0;
}
