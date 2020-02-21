#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "caffe/caffe.hpp"

#include "TRTNetwork/TRTNetwork.hpp"
#include "TRTNetwork/Transformer.hpp"

static std::vector<std::string> readLabels(std::string label_file)
{
    std::ifstream labels_st(label_file.c_str());
    if (!labels_st) {
        TRTLog(trt::ERROR) << "Unable to open labels file " << label_file;
        exit(1);
    }

    std::vector<std::string> labels;
    std::string line;
    while (std::getline(labels_st, line))
        labels.push_back(std::string(line));
    return labels;
}

static int volumeOf(const std::vector<int>& shape)
{
    int vol = 1;
    for (int i = 0; i < (int)shape.size(); ++i)
        vol *= shape.at(i);
    return vol;
}

int main(int argc, char** argv)
{
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0]
                  << " deploy.prototxt network.caffemodel"
                  << " mean.binaryproto labels.txt img.jpg" << std::endl;
        return 1;
    }

    ::google::InitGoogleLogging(argv[0]);

    std::string model_file   = argv[1];
    std::string trained_file = argv[2];
    std::string mean_file    = argv[3];
    std::string label_file   = argv[4];
    std::string img_file     = argv[5];

    std::vector<std::string> labels = readLabels(label_file);

    trt::TRTNetwork caffenet("caffenet", model_file, trained_file, {"prob"}, {"data"});

    trt::Transformer transformer;
    transformer.set_transpose({2, 0, 1});
    transformer.set_mean({104.0069879317889, 116.66876761696767, 122.6789143406786});
    transformer.set_raw_scale(255.0f);
    transformer.set_channel_swap({2, 1, 0});
    transformer.set_input_shape(caffenet.getBlobShape("data"));

    float *data_ptr = new float[volumeOf(caffenet.getBlobShape("data"))];
    float *prob_ptr = new float[volumeOf(caffenet.getBlobShape("prob"))];

    cv::Mat img = cv::imread(img_file, -1);
    if (img.empty()) {
        TRTLog(trt::ERROR) << "Unable to decode image " << img;
        exit(1);
    }

    transformer.preprocess(data_ptr, img);
    caffenet.forward(1, {{"data", data_ptr}, {"prob", prob_ptr}});

    std::vector<float> predScore(prob_ptr, prob_ptr + volumeOf(caffenet.getBlobShape("prob")));
    std::pair<float, int> topPred(0.0f, 0);
    for (int i = 0; i < (int)predScore.size(); ++i)
        if (predScore.at(i) > topPred.first)
            topPred = std::pair<float, int>(predScore.at(i), i);

    TRTLog(trt::INFO) << "Predicted score: " << topPred.first
                      << " index: " << topPred.second
                      << " label: " << labels.at(topPred.second);

    delete[] data_ptr;
    delete[] prob_ptr;

    return 0;
}
