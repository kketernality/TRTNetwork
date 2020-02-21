#pragma once

#include <string>
#include <vector>
#include <map>
#include <sstream>

#include "TRTBuilder.hpp"

/** @note Assume there are at most 6 input & output blobs **/
#define TRT_MAX_BINDINGS 6

namespace trt {

/**
 * @brief Neural network instance on TensorRT.
 *
 *        This class is neighter copyable nor movable due to the constraint
 *        of TensorRT. We cannot manipulate the internal structure of TensorRT
 *        runtime objects which are hidden inside the TensorRT library.
 */
class TRTNetwork
{
public:
    /**
     * @brief TRTNetwork constructor
     * @param name              Name of the network instance.
     * @param deploy            Path of the prototxt file of the network.
     * @param model             Path of the caffemodel file of the network.
     * @param outputBlobs       List of the output blob names.
     *                          The names are defined in the prototxt of the network.
     *
     *                          { blob_name_1, blob_name_2, ... }
     *
     * @param inputBlobs        List of the input blob names.
     *                          The names are defined in the prototxt of the network.
     *
     *                          { blob_name_1, blob_name_2, ... }
     *
     * @param maxBatchSize      Max batch size of network inference.
     * @param inputHeight       Used in image inference workload.
     *                          Resize the height of the first input blob (usually image) to inputHeight.
     *                          Set as 0 to use the default value defined in prototxt.
     * @param inputWidth        Used in image inference workload.
     *                          Resize the width of the first input blob (usually image) to inputWidth.
     *                          Set as 0 to use the default value defined in prototxt.
     * @param maxWorkspaceSize  The maximum workspace size specified in TensorRT.
     */
    TRTNetwork(const std::string &name,
               const std::string &deploy,
               const std::string &model,
               const std::vector< std::string > &outputBlobs,
               const std::vector< std::string > &inputBlobs,
               int maxBatchSize = 1, int inputHeight = 0, int inputWidth = 0,
               size_t maxWorkspaceSize = 1 << 25);

    TRTNetwork(const TRTNetwork& other) = delete;
    TRTNetwork& operator= (const TRTNetwork& other) = delete;
    TRTNetwork(TRTNetwork&& other) = delete;

    ~TRTNetwork();

    /**
     * @brief Trigger neural network to do inference with the bound input & output.
     * @param batchSize  Batch size of this inference
     * @param feedDict   The binding of input and output. The is the list of the tuple in the form of
     *
     *                   { blob_name, blob_data_pointer }
     *
     *                   as the feeding data. This resembles the Python style dictionary in TensorFlow
     *                   to ease the usage and provides great flexibility. For example, one can pass
     *
     *                   { {"data", data_ptr}, {"prob", prob_ptr} }
     *
     *                   as the feedDict argument to the function.
     *
     * @return success   True if successfully trigger the inference.
     */
    bool forward(int batchSize, const std::vector< std::pair<std::string, void*> > &feedDict);
    /**
     * @brief An overloaded function to do the asynchronous execution of Cuda.
     *        It differs from the original as it passes the on-Gpu pointers
     *        instead of on-Cpu pointers.
     *
     * @param batshSize  Batch size of this inference
     * @param feedDict   The same structure as described above
     *
     *                   { {"data", data_ptr}, {"prob", prob_ptr} }
     *
     *                   Note the pointers should reside on Gpu instead of Cpu.
     *
     * @param stream     Cuda stream used to asynchronous execution.
     */
    bool forward(int batchSize, const std::vector< std::pair<std::string, void*> > &feedDict, cudaStream_t stream);

    std::string getName() const;
    std::string getBindingInfoString() const;
    std::vector<int> getBlobShape(const std::string& name) const;

protected:
    const std::string name;

    std::vector<std::string> outputBlobNames;
    std::vector<std::string> inputBlobNames;
    std::map<std::string, IOBlob> blobMapping;

    void *bindings[TRT_MAX_BINDINGS];

    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IExecutionContext *contex = nullptr;
};

} // namespace trt
