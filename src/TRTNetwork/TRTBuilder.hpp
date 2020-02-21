#pragma once

#include <string>
#include <iostream>
#include <vector>
#include <mutex>
#include <memory>

#include "TensorRT/NvInfer.h"
#include "TensorRT/NvCaffeParser.h"

#include "Logger.hpp"

namespace trt {

/**
 * @brief Helper ostream function for nvinfer1::Dims
 */
std::ostream& operator<< (std::ostream &os, const nvinfer1::Dims &dims);

/**
 * @brief This is the data holder of the IO blobs of a neural
 *        network and their associated indices and dims.
 */
class IOBlob
{
public:
    IOBlob() = default;
    IOBlob(const IOBlob& other) = delete;
    IOBlob& operator =(const IOBlob& other) = delete;
    ~IOBlob();

    bool isOutput;
    std::string name;
    int index;
    nvinfer1::Dims dims;
    size_t sizePerBatch;
    void *gpuPtr = nullptr; // On-Gpu pointer for the data
};

/**
 * @brief Helper ostream function for TRTIOBlobs.
 */
std::ostream& operator<< (std::ostream& os, const IOBlob& blob);

/**
 * @brief TRTBuilder is used to create neural network instance
 */
class TRTBuilder
{
public:
    static nvinfer1::ICudaEngine* createEngine(
        const std::string &deploy,
        const std::string &model,
        const std::vector<std::string> &outputNames,
        int maxBatchSize = 1, int inputHeight = 0, int inputWidth = 0,
        size_t maxWorkspaceSize = 1 << 25);
};

} // namespace trt
