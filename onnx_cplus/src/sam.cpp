#include "sam.h"
#include <chrono>

SamOnnxModel::SamOnnxModel(std::string &encoderModelPath, std::string &decoderModelPath, int mask_threshold, int num_cpu_threads)
{
    Ort::SessionOptions session_opt;

    if (num_cpu_threads != 0)
        // session_opt.SetInterOpNumThreads(num_cpu_threads);
        session_opt.SetIntraOpNumThreads(num_cpu_threads);
    
    // session_opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    this->encoderSession = new Ort::Session(env, encoderModelPath.c_str(), session_opt);
    this->decoderSession = new Ort::Session(env, decoderModelPath.c_str(), session_opt);

    this->encoderInputCount = this->encoderSession->GetInputCount();
    this->encoderOutputCount = this->encoderSession->GetOutputCount();
}

SamOnnxModel::~SamOnnxModel(){}

cv::Mat SamOnnxModel::transform_image(const cv::Mat &image)
{
    cv::Mat NormImage, paddingImage, resizedImage, chwImage;
    cv::Mat channels[3];
    cv::split(image, channels);
    for (int i = 0; i < 3; ++i) {
        channels[i] = (channels[i] - this->imageMean[i]) / this->imageStd[i];
    }
    cv::merge(channels, 3, NormImage);

    int width = image.cols;
    int height = image.rows;
    int max_length = std::max(width, height);
    cv::copyMakeBorder(NormImage, paddingImage, 0, max_length - height, 0, max_length - width, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    cv::resize(paddingImage, resizedImage, this->targetSize, cv::INTER_LINEAR);

    // HWC -> CHW
    cv::dnn::blobFromImage(resizedImage, chwImage);

    return chwImage;
}

std::vector<std::tuple<float, float>> SamOnnxModel::getBoxPoint(const std::vector<std::tuple<float, float>> &coords, const std::tuple<int, int> &originSize)
{
    int oldW = std::get<0>(originSize);
    int oldH = std::get<1>(originSize);
    float scale = 1024.0f / static_cast<float>(std::max(oldH, oldW));
    int newW = static_cast<int>(oldW * scale + 0.5);
    int newH = static_cast<int>(oldH * scale + 0.5);

    std::vector<std::tuple<float, float>> transformedCoords = coords;
    for (auto &coord : transformedCoords) {
        std::get<0>(coord) = std::get<0>(coord) * (newW / static_cast<float>(oldW));
        std::get<1>(coord) = std::get<1>(coord) * (newH / static_cast<float>(oldH));
    }

    return transformedCoords;
}

std::vector<float> SamOnnxModel::encode_image(const cv::Mat &input_image)
{
    cv::Mat transedImg = this->transform_image(input_image);

    std::vector<float> transedImgVec;
    float *transedImgData = reinterpret_cast<float*> (transedImg.data);
    transedImgVec.assign(transedImgData, transedImgData + transedImg.total() * transedImg.channels());

    std::vector<float> encoderInput(this->numEncoderInputElements);
    std::vector<float> encoderOutput(this->numEncoderOutputElements);

    // define tensor
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(this->memoryInfo, transedImgVec.data(), transedImgVec.size(), this->encoderInputShape.data(), this->encoderInputShape.size());
    Ort::Value outputTensor = Ort::Value::CreateTensor<float>(this->memoryInfo, encoderOutput.data(), encoderOutput.size(), this->encoderOutputShape.data(), this->encoderOutputShape.size());

    // define names
    Ort::AllocatedStringPtr inputName = this->encoderSession->GetInputNameAllocated(0, this->ortAlloc);
    Ort::AllocatedStringPtr outputName = this->encoderSession->GetOutputNameAllocated(0, this->ortAlloc);
    const std::array<const char*, 1> inputNames = {inputName.get()};
    const std::array<const char*, 1> outputNames = {outputName.get()};
    inputName.release();
    outputName.release();

    // run encoder
    this->encoderSession->Run(this->runOptions, inputNames.data(), &inputTensor, this->encoderInputCount, outputNames.data(), &outputTensor, this->encoderOutputCount);
    
    return encoderOutput;
}

cv::Mat SamOnnxModel::decode_image(std::vector<std::tuple<float, float>> coords, std::vector<float> imageEmbedding, cv::Mat img, cv::Mat originalImage)
{
    const std::array<int64_t, 4> imageEmbeddingShape = {1, 256, 64, 64};

    std::tuple<int, int> originSize = {originalImage.cols, originalImage.rows};
    std::vector<int64_t> coordShape =  this->decoderSession->GetInputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();
    auto transformedCoords = this->getBoxPoint(coords, originSize);
    coordShape[1] = transformedCoords.size();
    std::cout<<coordShape[1]<<"size point"<<std::endl;
    const std::array<int64_t, 3> onnxCoordShape = {coordShape[0], coordShape[1], coordShape[2]};
    std::vector<float> onnxCoords;
    for (auto &point : transformedCoords) {
        onnxCoords.push_back(std::get<0>(point));
        onnxCoords.push_back(std::get<1>(point));
    }

    const std::array<int64_t, 2> onnxLabelShape = {1, 2};
    std::vector<float> OnnxLabel = {1.0, 1.0};

    const std::array<int64_t, 4> onnxMaskInputShape = {1, 1, 256, 256};
    std::vector<float> onnxMaskInput(1 * 1 * 256 * 256, 0.0f);

    const std::array<int64_t, 1> hasMaskInputShape = {1};
    std::vector<float> hasMaskInput(1, 0.0f);

    const std::array<int64_t, 1> origImgSizeShape = {2};
    std::vector<float> origImgSize = {static_cast<float>(img.rows), static_cast<float>(img.cols)};

    this->decoderInputCount = this->decoderSession->GetInputCount();
    std::vector<Ort::Value> inputTensors;

    inputTensors.emplace_back(Ort::Value::CreateTensor<float>(this->memoryInfo, imageEmbedding.data(), imageEmbedding.size(), imageEmbeddingShape.data(), imageEmbeddingShape.size()));
    inputTensors.emplace_back(Ort::Value::CreateTensor<float>(this->memoryInfo, onnxCoords.data(), onnxCoords.size(), onnxCoordShape.data(), onnxCoordShape.size()));
    inputTensors.emplace_back(Ort::Value::CreateTensor<float>(this->memoryInfo, OnnxLabel.data(), OnnxLabel.size(), onnxLabelShape.data(), onnxLabelShape.size()));
    inputTensors.emplace_back(Ort::Value::CreateTensor<float>(this->memoryInfo, onnxMaskInput.data(), onnxMaskInput.size(), onnxMaskInputShape.data(), onnxMaskInputShape.size()));
    inputTensors.emplace_back(Ort::Value::CreateTensor<float>(this->memoryInfo, hasMaskInput.data(), hasMaskInput.size(), hasMaskInputShape.data(), hasMaskInputShape.size()));
    inputTensors.emplace_back(Ort::Value::CreateTensor<float>(this->memoryInfo, origImgSize.data(), origImgSize.size(), origImgSizeShape.data(), origImgSizeShape.size()));

    int pixels = img.rows * img.cols;
    const std::array<int64_t, 4> masksShape = {1, 1, img.rows, img.cols};
    std::vector<float> output(pixels);

    const std::array<int64_t, 2> scoreShape = {1, 1};
    std::vector<float> score(1);

    const std::array<int64_t, 4> lowResLogits = {1, 1, 256, 256};
    std::vector<float> lowResLogitsOutput(1 * 1 * 256 * 256);

    std::vector<Ort::Value> outputTensors;
    outputTensors.emplace_back(Ort::Value::CreateTensor(this->memoryInfo, output.data(), output.size(), masksShape.data(), masksShape.size()));
    outputTensors.emplace_back(Ort::Value::CreateTensor(this->memoryInfo, score.data(), score.size(), scoreShape.data(), scoreShape.size()));
    outputTensors.emplace_back(Ort::Value::CreateTensor(this->memoryInfo, lowResLogitsOutput.data(), lowResLogitsOutput.size(), lowResLogits.data(), lowResLogits.size()));

    //set input names
    Ort::AllocatedStringPtr inputName0 = this->decoderSession->GetInputNameAllocated(0, this->ortAlloc);
    Ort::AllocatedStringPtr inputName1 = this->decoderSession->GetInputNameAllocated(1, this->ortAlloc);
    Ort::AllocatedStringPtr inputName2 = this->decoderSession->GetInputNameAllocated(2, this->ortAlloc);
    Ort::AllocatedStringPtr inputName3 = this->decoderSession->GetInputNameAllocated(3, this->ortAlloc);
    Ort::AllocatedStringPtr inputName4 = this->decoderSession->GetInputNameAllocated(4, this->ortAlloc);
    Ort::AllocatedStringPtr inputName5 = this->decoderSession->GetInputNameAllocated(5, this->ortAlloc);
    const std::array<const char*, 6> inputNames = {inputName0.get(), inputName1.get(), inputName2.get(), inputName3.get(), inputName4.get(), inputName5.get()};
    inputName0.release();
    inputName1.release();
    inputName2.release();
    inputName3.release();
    inputName4.release();
    inputName5.release();
    
    //set output names
    Ort::AllocatedStringPtr outputName0 = this->decoderSession->GetOutputNameAllocated(0, this->ortAlloc);
    Ort::AllocatedStringPtr outputName1 = this->decoderSession->GetOutputNameAllocated(1, this->ortAlloc);
    Ort::AllocatedStringPtr outputName2 = this->decoderSession->GetOutputNameAllocated(2, this->ortAlloc);
    const std::array<const char*, 3> outputNames = {outputName0.get(), outputName1.get(), outputName2.get()};
    outputName0.release();
    outputName1.release();
    outputName2.release();

    this->decoderSession->Run(this->runOptions, inputNames.data(), inputTensors.data(), 6, outputNames.data(), outputTensors.data(), 3);

    //convert mat
    cv::Mat outputMask = cv::Mat(img.rows, img.cols, CV_32FC1, output.data());
    outputMask.convertTo(outputMask, CV_8UC1);

    cv::normalize(outputMask, outputMask, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    cv::resize(outputMask, outputMask, originalImage.size());

    originalImage.convertTo(originalImage, CV_8UC1);

    cv::threshold(outputMask, outputMask, 0, 255, cv::THRESH_BINARY);

    cv::Mat coloredMask = cv::Mat::zeros(originalImage.size(), CV_8UC3);
    coloredMask.setTo(cv::Scalar(255, 229, 204), outputMask);

    cv::imwrite("mask.jpg", outputMask);

    cv::Mat result;
    std::cout<< "originalImage size: " << originalImage.size() << std::endl;
    std::cout<< "colored mask size" << coloredMask.size() << std::endl;
    cv::addWeighted(originalImage, 0.5, coloredMask, 0.5, 0.0, result);

    return result;
}
