#include "Transformer.hpp"

namespace trt {

/**
 * @brief Validate the order with the following rules:
 *
 *        1) Lenth is N
 *        2) Each index is within [0, N-1]
 *        3) No duplicated indice
 */
static bool valid_order(const std::vector<int> &order, int N)
{
    if ((int)order.size() != N)
        return false;
    for (int i = 0; i < N; ++i) {
        if (order.at(i) > N-1 || 0 > order.at(i))
            return false;
        for (int j = 0; j < N; ++j)
            if (i != j && order.at(i) == order.at(j))
                return false;
    }

    return true;
}


Transformer::Transformer()
{
    dim_order = {2, 0, 1};
    channel_order = {2, 1, 0};
    mean_vec = {0.f, 0.f, 0.f};
    raw_scale = 255.f;
}

bool Transformer::set_transpose(const std::vector<int> &order)
{
    if (!valid_order(order, 3))
        return false;
    dim_order = order;
    return true;
}

bool Transformer::set_channel_swap(const std::vector<int> &order)
{
    if (!valid_order(order, 3))
        return false;
    channel_order = order;
    return true;
}

bool Transformer::set_mean(const std::vector<float> &vec)
{
    if ((int)vec.size() != 3)
        return false;
    mean_vec = vec;
    mean_ = cv::Scalar(vec.at(0), vec.at(1), vec.at(2));
    return true;
}

bool Transformer::set_raw_scale(float value)
{
    raw_scale = value;
    return true;
}

bool Transformer::set_input_shape(const std::vector<int>& shape)
{
    if ((int)shape.size() != 3)
        return false;
    for (int i = 0; i < 3; ++i)
        if (shape.at(i) <= 0)
            return false;

    input_shape = shape;
    num_channels_ = shape.at(0);
    input_geometry_ = cv::Size(shape.at(2), shape.at(1));
    return true;
}

bool Transformer::preprocess(float *input_data, const cv::Mat &img)
{
    /* Wrap the input layer of the network in separate cv::Mat objects
     * (one per channel). This way we save one memcpy operation and we
     * don't need to rely on cudaMemcpy2D. The last preprocessing
     * operation will write the separate channels directly to the input
     * layer. */

    std::vector<cv::Mat> input_channels;
    for (int i = 0; i < num_channels_; ++i) {
        cv::Mat channel(input_geometry_, CV_32FC1, input_data + i * input_geometry_.area());
        input_channels.push_back(channel);
    }

    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;

    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    cv::Mat sample_normalized;
    sample_normalized = sample_float - mean_;

    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    cv::split(sample_normalized, input_channels);
    return true;
}

} // namespace trt
