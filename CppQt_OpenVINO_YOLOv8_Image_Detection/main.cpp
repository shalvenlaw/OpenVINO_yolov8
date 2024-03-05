#include "../common.hpp"

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
        cv::dnn::NMSBoxes(boxes, confidences, 0.25f, 0.45f, indexes);
        for (size_t i = 0; i < indexes.size(); i++) {
            const int index = indexes[i];
            const cv::Rect &box = boxes[index];
            // 绘制矩形框
            cv::rectangle(image, box, cv::Scalar(0, 0, 255), 2, 8);
            // 绘制标签
            const std::string label = class_names[class_ids[index]] + ":" + std::to_string(confidences[index]).substr(0, 4);
            const cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, nullptr);
            const cv::Rect textBox(box.tl().x, box.tl().y - 15, textSize.width, textSize.height + 5);
            cv::rectangle(image, textBox, cv::Scalar(0, 255, 255), cv::FILLED);
            cv::putText(image, label, cv::Point(box.tl().x, box.tl().y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0, 0, 0));
        }
        // 计算FPS
        const float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
        std::cout << "Infer time(ms): " << t * 1000 << "ms; Detections: " << indexes.size() << std::endl;
        cv::putText(image, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
        /// 获取程序名称
        const std::string programName{extractedProgramName(argv[0])};
        cv::imshow(programName, image);
        /// 保存结果图
        save(programName, image);

        cv::waitKey(0);
        cv::destroyAllWindows();

    } catch (const std::exception &e) {
        std::cerr << "exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "unknown exception" << std::endl;
    }
    return 0;
}
