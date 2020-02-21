#pragma once

#include <vector>

#include "opencv2/opencv.hpp"

namespace trt {

/**
 * @brief This class serves as a helper for data transformation before
 *        feeding it into the network.
 *
 *        The general transformation is currently not implemented. Only
 *        OpenCV to Caffe conversion is supported currently.
 *
 * @todo Implement general transformation in function preprocess.
 */
class Transformer
{
public:
    Transformer();

    /**
     * @brief Set transformation parameters
     */
    bool set_transpose(const std::vector<int>& order);    // Not functional
    bool set_channel_swap(const std::vector<int>& order); // Not functional
    bool set_mean(const std::vector<float>& vec);
    bool set_raw_scale(float value);
    bool set_input_shape(const std::vector<int>& shape);

    /**
     * @brief Process data for network input. Currently only OpenCV to
     *        Caffe conversion is implemented.
     *
     *        This implementation copies from Caffe's example:
     *
     *        https://github.com/BVLC/caffe/blob/master/examples/cpp_classification/classification.cpp
     */
    bool preprocess(float* data_ptr, const cv::Mat& img);

private:
    std::vector<int> dim_order;
    std::vector<int> channel_order;
    std::vector<float> mean_vec;
    float raw_scale;
    std::vector<int> input_shape;
    cv::Scalar mean_;
    int num_channels_;
    cv::Size input_geometry_;
};

} // namespace trt
