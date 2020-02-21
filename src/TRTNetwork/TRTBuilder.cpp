#include "TRTBuilder.hpp"

#include "cuda_runtime.h"

namespace trt {

/**
 * @brief Helper ostream function for nvinfer1::Dims
 */
std::ostream& operator<< (std::ostream &os, const nvinfer1::Dims &dims)
{
    os << "[";
    for (int i = 0; i < dims.nbDims - 1; i++)
        os << dims.d[i] << ", ";
    os << dims.d[dims.nbDims - 1] << "]";
    return os;
}

/**
 * @brief Helper ostream function for TRTIOBlobs.
 */
std::ostream& operator<< (std::ostream& os, const IOBlob& blob)
{
    os << "\t" << blob.name << "\t" << blob.index << "\t" << blob.dims;
    return os;
}

IOBlob::~IOBlob()
{
    // Cuda api knows if ptr == nullptr
    cudaFree(gpuPtr);
}

nvinfer1::ICudaEngine* TRTBuilder::createEngine(
    const std::string &deploy,
    const std::string &model,
    const std::vector<std::string> &outputNames,
    int maxBatchSize, int inputHeight, int inputWidth,
    size_t maxWorkspaceSize)
{
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(Logger::globalInstance());
    nvinfer1::INetworkDefinition *network = builder->createNetwork();
    nvcaffeparser1::ICaffeParser *parser = nvcaffeparser1::createCaffeParser();

    const nvcaffeparser1::IBlobNameToTensor *mapping =
        parser->parse(deploy.c_str(), model.c_str(), *network, nvinfer1::DataType::kFLOAT);
    for (const auto& outputName : outputNames)
        network->markOutput(*mapping->find(outputName.c_str()));

    /**
     * For image workload, we often need to resize the input blob to the desired size.
     * By convention, the first input blob is the image blob to resize so we just need to
     * retrieve and modify the shape of the tensor accordingly.
     */
    if (inputHeight && inputWidth) {
        nvinfer1::ITensor *inputTensor = network->getInput(0);
        nvinfer1::Dims dims = inputTensor->getDimensions();
        dims.d[1] = inputHeight; dims.d[2] = inputWidth;
        inputTensor->setDimensions(dims);
    }

    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(maxWorkspaceSize);
    nvinfer1::ICudaEngine *engine = builder->buildCudaEngine(*network);

    parser->destroy();
    network->destroy();
    builder->destroy();
    return engine;
}

} // namespace trt
