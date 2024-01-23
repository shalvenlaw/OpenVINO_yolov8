#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
//#include "samples/args_helper.hpp"
#include "samples/common.hpp"
#include "samples/classification_results.h"
#include "format_reader_ptr.h"

// 模型文件路径
static const std::string model_file = "../model/yolov8n-cls.onnx";
// 分类标签文件路径
static const std::string label_file = "../model/imagenet_2012.txt";
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
        // 读取标签文件, 这个文件必须和训练集的标签相符
        std::vector<std::string> labels;
        std::ifstream inputFile;
        inputFile.open(label_file, std::ios::in);
        if (inputFile.is_open()) {
            std::string strLine;
            while (std::getline(inputFile, strLine)) {
                trim(strLine);
                labels.push_back(strLine);
            }
        } else {
            throw "open labels file failed!";
        }
        /// 创建OpenVINO Runtime Core对象
        ov::Core core;
        std::shared_ptr<ov::Model> model = core.read_model(model_file);
//        std::cout << model->get_name() << std::endl;
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
        const cv::Mat classes_scores(out_rows, 1, CV_32F, (float *)output_buffer);

        cv::Point class_id_point;
        double score;
        cv::minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

        // 计算FPS
        const float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
        std::cout << "Infer time(ms): " << t * 1000 << "ms" << std::endl;
        std::cout << "Classification: "  << labels.at(class_id_point.y) << std::endl;
        std::cout << "Score: "  << score << std::endl;

        system("pause");
    } catch (const std::exception &e) {
        std::cerr << "exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "unknown exception" << std::endl;
    }
    return 0;
}
