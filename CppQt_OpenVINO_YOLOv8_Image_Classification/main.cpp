#include <fstream>
#include "../common.hpp"

// 模型文件路径
static const std::string model_file = "../model/yolov8n-cls.onnx";
// 分类标签文件路径
static const std::string label_file = "../model/imagenet_2012.txt";
// 测试图片路径
//static const std::string image_file = "../data/zidane.jpg";
static const std::string image_file = "../data/dog_512.bmp";
//static const std::string image_file = "../data/car.bmp";
//static const std::string image_file = "../data/bus.jpg";

int main(int argc, char **argv)
{
    try {
        /// 读取标签文件, 这个文件必须和训练集的标签相符
        std::vector<std::string> labels;
        std::ifstream inputFile;
        inputFile.open(label_file, std::ios::in);
        if (inputFile.is_open()) {
            std::string strLine;
            while (std::getline(inputFile, strLine)) {
                labels.push_back(strLine);
            }
        } else {
            throw "open labels file failed!";
        }
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
        fill_tensor_data_image(input_tensor, image);

        /// 执行推理计算
        infer_request.infer();

        /// 处理推理计算结果
        // 获得推理结果
        const ov::Tensor output = infer_request.get_output_tensor();
        const ov::Shape output_shape = output.get_shape();
        const float *output_buffer = output.data<const float>();

        // 解析推理结果
        const int out_rows = output_shape[1]; //获得"output"节点的rows
        const cv::Mat classes_scores(out_rows, 1, CV_32F, (float *)output_buffer);

        cv::Point class_id_point;
        double score;
        cv::minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);
        const std::string classification = labels.at(class_id_point.y);

        // 计算FPS
        const float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
        std::cout << "Infer time(ms): " << t * 1000 << "ms" << std::endl;
        std::cout << "Classification: "  << classification << std::endl;
        std::cout << "Score: "  << score << std::endl;

        const cv::Point pos(10, 30);
        cv::putText(image, cv::format("FPS: %.2f", 1.0 / t), pos, cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
        // 绘制标签
        const std::string label = classification + ": " + std::to_string(score).substr(0, 4);
        const cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, nullptr);
        const cv::Rect textBox(pos.x, pos.y + 5, textSize.width, textSize.height + 10);
        cv::rectangle(image, textBox, cv::Scalar(0, 255, 255), cv::FILLED);
        cv::putText(image, label, cv::Point(pos.x, pos.y + 20), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 0));

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
