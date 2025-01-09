#include <iostream>
#include "SamEncoder.h"
#include <chrono>


int main() {
    std::vector<float> res;

    cv::Mat img = cv::imread("../images/dog.jpg");
    SamEncoder encoder("../model/sam_vit_h_encoder.engine", 0);

    auto start = std::chrono::system_clock::now();
    res = encoder.runEncoder(img);
    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Encoder time: " << static_cast<double>(duration.count()) / 1000 << "s" << std::endl;
    return 0;
}
