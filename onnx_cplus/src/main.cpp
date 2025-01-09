
#include "sam.h"
#include <chrono>

int main()
{
    std::string encoder_path = "../../model/sam_vit_h_encoder_static_quntized.onnx";
    std::string decoder_path = "../../model/sam_vit_h_decoder.onnx";
    SamOnnxModel *sam = new SamOnnxModel(encoder_path, decoder_path, 0, 4);

    cv::Mat input_image = cv::imread("../../images/truck.jpg");
    cv::cvtColor(input_image, input_image, cv::COLOR_BGR2RGB);
    input_image.convertTo(input_image, CV_32FC3, 1.0f);

    auto start = std::chrono::system_clock::now();
    std::vector<float> imgEmbedding = sam->encode_image(input_image);

    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    std::cout << "Encode Time: " << double(duration.count()) << "s" << std::endl;

    std::vector<std::tuple<float, float>> coords = {{425.0, 600.0}, {700.0, 875.0}};
    // std::vector<std::tuple<float, float>> coords = {{660.0, 750.0}};
    cv::Mat originImage = cv::imread("../../images/truck.jpg");
    cv::Mat outputImage = sam->decode_image(coords, imgEmbedding, input_image, originImage);

    // cv::circle(outputImage, cv::Point(660, 750), 3, cv::Scalar(0, 255, 0), -1);
    cv::rectangle(outputImage, cv::Rect(425, 600, 275, 275), cv::Scalar(0, 255, 0), 3);

    cv::imwrite("./result.jpg", outputImage);

    return 0;
}