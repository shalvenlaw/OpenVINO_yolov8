#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include "samples/args_helper.hpp"
#include "samples/common.hpp"
#include "samples/classification_results.h"
#include "format_reader_ptr.h"

//// 将x限制在[x1, x2]: 若x<x1, 则取x1; 若x>x2, 则取x2; 若x\in [x1, x2], 则取x
//#define CLIP(x, x1, x2) std::max(x1, std::min(x, x2))

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
static const std::string model_file = "../model/yolov8n-seg.onnx";
// 测试图片路径
static const std::string image_file = "../data/zidane.jpg";
//static const std::string image_file = "../data/dog_512.bmp";
//static const std::string image_file = "../data/car.bmp";
//static const std::string image_file = "../data/bus.jpg";

struct SegmentOutput {
    int _id; // 结果类别id
    float _confidence; // 结果置信度
    cv::Rect2f _box; // 矩形框
    cv::Mat _boxMask; // 矩形框内mask, 节省内存空间和加快速度
};

void draw(cv::Mat &image, std::vector<SegmentOutput> &results)
{
    // 生成随机颜色
    std::vector<cv::Scalar> colors;
    std::srand(std::time(nullptr));
    for (int i = 0; i < class_names.size(); ++i) {
        const int b = std::rand() % 256;
        const int g = std::rand() % 256;
        const int a = std::rand() % 256;
        colors.push_back(cv::Scalar(b, g, a));
    }
    cv::Mat mask = image.clone();
    for (const SegmentOutput &result : results) {
        cv::rectangle(image, result._box, colors[result._id], 2, 8);

        mask(result._box).setTo(colors[result._id], result._boxMask);

        const std::string label = cv::format("%s:%.2f", class_names[result._id].c_str(), result._confidence);

        int baseLine;
        const cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        cv::putText(image, label,
                    cv::Point(result._box.x, std::max(result._box.y, float(labelSize.height))),
                    cv::FONT_HERSHEY_SIMPLEX, 1, colors[result._id], 2);
    }
    cv::addWeighted(image, 0.5, mask, 0.8, 1, image); // 把mask加在原图上面
}

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
    // 先缩放再交换通道
    cv::warpAffine(input_image, blob_image, matrix, cv::Size(width, height));
    convert(blob_image, blob_image, true, true);

//    // 下面根据scale范围进行数据转换, 这只是为了提高一点速度(主要是提高了交换通道的速度), 但可读性很差
//    // 如果不在意这点速度提升的可以固定一种做法(前两个if分支都可以)
//    if (scale > 1.0 + FLT_EPSILON) {
//        // 要放大, 那么先交换通道再放大
//        convert(input_image, blob_image, true, true);
//        cv::warpAffine(blob_image, blob_image, matrix, cv::Size(width, height));
//    } else if (scale < 1.0 - FLT_EPSILON) {
//        // 要缩小, 那么先缩小再交换通道
//        cv::warpAffine(input_image, blob_image, matrix, cv::Size(width, height));
//        convert(blob_image, blob_image, true, true);
//    } else {
//        convert(input_image, blob_image, true, true);
//    }
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

cv::Rect toBox(const cv::Mat &input, const cv::Rect &range)
{
    const float cx = input.at<float>(0);
    const float cy = input.at<float>(1);
    const float ow = input.at<float>(2);
    const float oh = input.at<float>(3);
    cv::Rect box;
    box.x = cvRound(cx - 0.5f * ow);
    box.y = cvRound(cy - 0.5f * oh);
    box.width = cvRound(ow);
    box.height = cvRound(oh);
    return box & range;
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
        const float scale_factor = fill_tensor_data_image(input_tensor, image);

        /// 执行推理计算
        infer_request.infer();

        /// 处理推理计算结果
        // 获得推理结果, 输出结点是[116,8400], 一共8400个结果, 每个结果116个维度.
        // 116=4+80+32, 4是预测框的[cx, cy, w, h], 80是每个类别的置信度, 32是分割需要用到的
        const ov::Tensor output0 = infer_request.get_tensor("output0");
        const float *output0_buffer = output0.data<const float>();
        const ov::Shape output0_shape = output0.get_shape();
        const int output0_rows = output0_shape[1];
        const int output0_cols = output0_shape[2];
        std::cout << "The shape of Detection tensor:" << output0_shape << std::endl;

        const ov::Tensor output1 = infer_request.get_tensor("output1");
        const ov::Shape output1_shape = output1.get_shape();
        std::cout << "The shape of Proto tensor:" << output1_shape << std::endl;
        std::cout << std::endl << std::endl;

        // Detect Matrix: 116 x 8400 -> 8400 x 116
        // 一共8400个结果, 每个结果116个维度.
        // 116=4+80+32, 4是预测框的[cx, cy, w, h]; 80是每个类别的置信度; 32需要与Proto Matrix相乘得到分割mask, 所以这里转置了矩阵
        const cv::Mat detect_buffer = cv::Mat(output0_rows, output0_cols, CV_32F, (float *)output0_buffer).t();
        // Proto Matrix: 1x32x160x160 -> 32x25600
        const cv::Mat proto_buffer(output1_shape[1], output1_shape[2]*output1_shape[3], CV_32F, output1.data());

        const float conf_threshold = 0.5;
        const float nms_threshold = 0.5;
        std::vector<cv::Rect> mask_boxes;
        std::vector<cv::Rect> boxes;
        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Mat> masks;
        for (int i = 0; i < detect_buffer.rows; ++i) {
            const cv::Mat result = detect_buffer.row(i);
            /// 处理检测部分的结果
            // 取置信度最大的那个标签
            const cv::Mat classes_scores = result.colRange(4, 84);
            cv::Point class_id_point;
            double score;
            cv::minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

            if (score > conf_threshold) {
                // score\in [0, 1), 置信度太小的结果可以舍弃掉
                class_ids.push_back(class_id_point.x);
                confidences.push_back(score);
                // 预测框是在640x640的图片上预测的, 但是分割结果只有160x160
                const float mask_scale = 0.25f; // 160/640 = 0.25

                const cv::Mat detection_box = result.colRange(0, 4);
                const cv::Rect mask_box = toBox(detection_box * mask_scale,
                                                cv::Rect(0, 0, 160, 160));
                const cv::Rect image_box = toBox(detection_box * scale_factor,
                                                 cv::Rect(0, 0, image.cols, image.rows));
                mask_boxes.push_back(mask_box);
                boxes.push_back(image_box);

                /// 处理分割部分的结果
                masks.push_back(result.colRange(84, 116));
            }
        }
        // NMS, 消除具有较低置信度的冗余重叠框
        std::vector<int> nms_indexes;
        cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, nms_indexes);

        std::vector<SegmentOutput> segmentOutputs;
        for (const int index : nms_indexes) {
            SegmentOutput segmentOutput;
            segmentOutput._id = class_ids[index];
            segmentOutput._confidence = confidences[index];
            segmentOutput._box = boxes[index];
            // sigmoid运算
            cv::Mat m;
            cv::exp(-masks[index] * proto_buffer, m);
            m = 1.0f / (1.0f + m);
            m = m.reshape(1, 160); // 1x25600 -> 160x160
            cv::resize(m(mask_boxes[index]) > 0.5f,
                       segmentOutput._boxMask, segmentOutput._box.size());

            segmentOutputs.push_back(segmentOutput);
        }
        draw(image, segmentOutputs);

        // 计算FPS
        const float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
        std::cout << "Infer time(ms): " << t * 1000 << "ms; Detections: " << nms_indexes.size() << std::endl;
        cv::putText(image, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
        cv::imshow("CppQt_OpenVINO202203_YOLOv8_Image_Segmentation", image);

        cv::waitKey(0);
        cv::destroyAllWindows();

    } catch (const std::exception &e) {
        std::cerr << "exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "unknown exception" << std::endl;
    }
    return 0;
}
