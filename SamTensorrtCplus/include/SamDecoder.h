#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <string>


class SamDecoder 
{
public:
    SamDecoder();
    ~SamDecoder();

    cv::Mat runDecoder(std::vector<float> img_embading, std::vector<std::tuple<float, float>>);

private:
    std::string model_file;
    cv::Size img_size;
    Ort::SessionOptions s_opt;
    Ort::Session *session;

    bool isExist(std::string modlel_file);
    void loadModel(std::string model_file);
    std::vector<std::tuple<float, float>> getBoxPoint(const std::vector<std::tuple<float, float>> &coords);
};