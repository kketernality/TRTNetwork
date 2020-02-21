#include "TRTNetwork.hpp"

#include "cuda_runtime.h"

namespace trt {

/**
 * @brief TRTNetwork constructor
 */
TRTNetwork::TRTNetwork(
           const std::string &name,
           const std::string &deploy,
           const std::string &model,
           const std::vector< std::string > &outputBlobs,
           const std::vector< std::string > &inputBlobs,
           int maxBatchSize, int inputHeight, int inputWidth,
           size_t maxWorkspaceSize)
    : name(name),
      outputBlobNames(outputBlobs),
      inputBlobNames(inputBlobs)
{
    for (const std::string& blob : outputBlobs) {
        blobMapping[blob].name = blob;
        blobMapping[blob].isOutput = true;
    }
    for (const std::string& blob : inputBlobs) {
        blobMapping[blob].name = blob;
        blobMapping[blob].isOutput = false;
    }

    engine = TRTBuilder::createEngine(deploy, model, outputBlobs,
                                      maxBatchSize, inputHeight, inputWidth,
                                      maxWorkspaceSize);
    contex = engine->createExecutionContext();

    for (std::pair<const std::string, IOBlob> &kv : blobMapping) {
        kv.second.index = engine->getBindingIndex(kv.second.name.c_str());
        kv.second.dims = engine->getBindingDimensions(kv.second.index);

        size_t size = 1;
        for (int i = 0; i < kv.second.dims.nbDims; i++)
            size *= kv.second.dims.d[i];
        kv.second.sizePerBatch = size;

        cudaMalloc(&kv.second.gpuPtr, maxBatchSize * size * sizeof(float));
    }
}

TRTNetwork::~TRTNetwork()
{
    if (contex)
        contex->destroy();
    if (engine)
        engine->destroy();
}

bool TRTNetwork::forward(int batchSize, const std::vector< std::pair<std::string, void*> > &feedDict)
{
    typedef std::map<std::string, IOBlob>::iterator it_t;

    for (const std::pair<const std::string, void*> &kv : feedDict) {
        it_t it = blobMapping.find(kv.first);
        if (it == blobMapping.end())
            return false;
        if (!it->second.isOutput)
            cudaMemcpy(it->second.gpuPtr, kv.second, batchSize * it->second.sizePerBatch * sizeof(float),
                       cudaMemcpyHostToDevice);
        bindings[it->second.index] = it->second.gpuPtr;
    }

    if (!contex->execute(batchSize, bindings))
        return false;

    for (const std::pair<const std::string, void*> &kv : feedDict) {
        it_t it = blobMapping.find(kv.first);
        if (it->second.isOutput)
            cudaMemcpy(kv.second, it->second.gpuPtr, batchSize * it->second.sizePerBatch * sizeof(float),
                       cudaMemcpyDeviceToHost);
    }

    return true;
}

bool TRTNetwork::forward(int batchSize, const std::vector<std::pair<std::string, void*> > &feedDict, cudaStream_t stream)
{
    typedef std::map<std::string, IOBlob>::iterator it_t;

    for (const std::pair<const std::string, void*> &kv : feedDict) {
        it_t it = blobMapping.find(kv.first);
        if (it == blobMapping.end())
            return false;
        if (!it->second.isOutput)
            bindings[it->second.index] = kv.second;
    }

    return contex->enqueue(batchSize, bindings, stream, nullptr);
}

std::string TRTNetwork::getName() const
{
    return name;
}

std::string TRTNetwork::getBindingInfoString() const
{
    std::stringstream ss;

    ss << "Network " << name << " - Index Mapping & Dimensions" << std::endl;
    for (const std::pair<const std::string, IOBlob> &kv : blobMapping)
        ss << "\t" << kv.first << "\t" << kv.second << std::endl;
    char c; ss.get(c); // Remove the last newline '\n' charater

    return ss.str();
}

std::vector<int> TRTNetwork::getBlobShape(const std::string& name) const
{
    typedef std::map<std::string, IOBlob>::const_iterator it_t;

    it_t it = blobMapping.find(name);
    if (it == blobMapping.cend())
        return std::vector<int>();

    nvinfer1::Dims dims = it->second.dims;
    std::vector<int> shape;
    for (int i = 0; i < dims.nbDims; ++i)
        shape.push_back(dims.d[i]);
    return shape;
}

} // namespace trt
