#include "SamEncoder.h"


SamEncoder::SamEncoder(const std::string& engine_file, const int device_id):m_engine_file(engine_file), device_id(device_id) {
    cudaSetDevice(device_id);
    cudaStreamCreate(&m_stream);
    LoadModel();
}

SamEncoder::~SamEncoder() {
    cudaStreamSynchronize(m_stream);
    cudaStreamDestroy(m_stream);
    if (d_input_ptr != nullptr) cudaFree(d_input_ptr);
    if (d_output_ptr != nullptr) cudaFree(d_output_ptr);
}

const std::vector<float> SamEncoder::runEncoder(const cv::Mat& img) {
    const cv::Mat mattmp = img.clone();
    preprocess(mattmp);
    Forward();
    auto res = postProcess();

    return std::move(res);
}

bool SamEncoder::LoadModel()
{
    if (IsExists(m_engine_file))
        return LoadTRTModel();
    return false;
}

bool SamEncoder::LoadTRTModel()
{
    std::ifstream fgie(m_engine_file, std::ios_base::in | std::ios_base::binary);
    if (!fgie)
        return false;

    std::stringstream buffer;
    buffer << fgie.rdbuf();

    std::string stream_model(buffer.str());

    deserializeCudaEngine(stream_model.data(), stream_model.size());
    return true;
}

bool SamEncoder::deserializeCudaEngine(const void *blob, std::size_t size)
{
    m_runtime = nvinfer1::createInferRuntime(my_loger);
    assert(m_runtime != nullptr);

    bool didInitPlugs = initLibNvInferPlugins(nullptr, "");
    m_engine = m_runtime->deserializeCudaEngine(blob, size, nullptr);
    assert(m_engine != nullptr);

    m_context = m_engine->createExecutionContext();
    assert(m_context != nullptr);

    mallocInputOutput();

    return true;
}

bool SamEncoder::mallocInputOutput()
{
    m_buffers.clear();
    int nb_bind = m_engine->getNbBindings();

    nvinfer1::Dims input_dim = m_engine->getBindingDimensions(0);
    nvinfer1::Dims output_dim = m_engine->getBindingDimensions(1);

    // Ctreate GPU buffers on device
    cudaMalloc((void **)&d_input_ptr, m_max_batchsize * input_dim.d[1] * input_dim.d[2] * input_dim.d[3] * sizeof(float));
    cudaMalloc((void **)&d_output_ptr, m_max_batchsize * output_dim.d[1] * output_dim.d[2] * output_dim.d[3] * sizeof(float));

    m_buffers.emplace_back(d_input_ptr);
    m_buffers.emplace_back(d_output_ptr);
    return true;
}

void SamEncoder::Forward()
{
    assert(m_engine != nullptr);
    m_context->enqueueV2(m_buffers.data(), m_stream, nullptr);
    cudaStreamSynchronize(m_stream);
}

const std::vector<float> SamEncoder::postProcess()
{
    std::vector<float> result(ouput_size);
    cudaMemcpyAsync(result.data(), d_output_ptr, m_batchsize * ouput_size * sizeof(float), cudaMemcpyDeviceToHost, m_stream);
    cudaStreamSynchronize(m_stream);

    return std::move(result);
}

void SamEncoder::preprocess(const cv::Mat &image)
{
    cv::Mat img = image.clone();
    img = transform(img);

    std::vector<cv::Mat> channels;
    int offset = 0;

    for (auto &channel : channels) {
        cudaMemcpy(d_input_ptr + offset, channel.data, channel.total() * sizeof(float), cudaMemcpyHostToDevice);
        offset += channel.total();
    }
}

const cv::Mat SamEncoder::transform(const cv::Mat &imageBGR)
{
    cv::Mat img, padding_img;
    int height = imageBGR.rows;
    int width = imageBGR.cols;
    int max_length = std::max(width, height);

    cv::copyMakeBorder(imageBGR, padding_img, 0, max_length - height, 0, max_length - width, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    cv::resize(padding_img, img, cv::Size(1024, 1024), cv::INTER_LINEAR);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    img.convertTo(img, CV_32F);

    const cv::Scalar m_mean = cv::Scalar(123.675, 116.28, 103.53);
    const cv::Scalar m_std = cv::Scalar(58.395, 57.12, 57.375);
    const cv::Mat mean = cv::Mat(img.size(), img.type(), m_mean);
    const cv::Mat std_mat = cv::Mat(img.size(), img.type(), m_std);
    img = (img - mean) / std_mat;

    return std::move(img);
}
