#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <utility>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>


class SamOnnxModel {
public:
    int originWidth, originHeight;
    cv::Size targetSize = cv::Size(1024, 1024);

    Ort::Session* encoderSession;
    Ort::Session* decoderSession;
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "SamOnnxModel"};
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    SamOnnxModel(std::string &encoderModelPath, std::string &decoderModelPath, int mask_threshold=0.0, int nums_cpu_threads=0.0);
    ~SamOnnxModel();

    std::vector<float> encode_image(const cv::Mat &image);
    cv::Mat decode_image(std::vector<std::tuple<float, float>> coords, std::vector<float> image_embedding, cv::Mat img, cv::Mat original_img);

private:
    Ort::RunOptions runOptions;
    Ort::AllocatorWithDefaultOptions ortAlloc;

    std::vector<float> imageMean = {123.675, 116.28, 103.53};
    std::vector<float> imageStd = {58.395, 57.12, 57.375};

    //set input, output
    const int64_t numEncoderInputElements = 3 * 1024 * 1024;
    const int64_t numEncoderOutputElements = 256 * 64 * 64;

    size_t encoderInputCount, encoderOutputCount, decoderInputCount, decoderOutputCount;

    // encoder shape
    const std::array<int64_t, 4> encoderInputShape = {1, 3, 1024, 1024};
    const std::array<int64_t, 4> encoderOutputShape = {1, 256, 64, 64};

    // image transform
    cv::Mat transform_image(const cv::Mat &image);

    // point get
    std::vector<std::tuple<float, float>> getBoxPoint(const std::vector<std::tuple<float, float>> &coords, const std::tuple<int, int> &originSize);
};
