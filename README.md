# TRTNetwork

TensorRT C++ wrapper for Caffe-based neural network.

## Usage

This C++ wrapper is designed with simplicity in mind. A neural network instance can be easily created within a few lines of code. The inference operation of the network uses a dictionary-style argument which resembles the python API of the Tensorflow 1.0. 

```cpp
#include "TRTNetwork/TRTNetwork.hpp"

// Instantiate the network instance
trt::TRTNetwork network("caffenet", proto, weights, {"prob"}, {"data"});

// Trigger network inference
network.forward(batch, {{"prob", prob_ptr}, {"data", data_ptr}});

```
TensorRT wrapper API is detailedly documented in the header files.

## Build

The build of this repo relies on CMake. Execute the script:

```bash
./scripts/make.sh
```

## Run Examples

The examples in this repo require an environmental variable `CAFFE_ROOT` pointing to the Caffe repo path, because the examples use the model and test data included in Caffe to ensure the correctness of this TensorRT wrapper API.

```bash
export CAFFE_ROOT=${CAFFE_REPO_PATH}
```

The test data and model should be prepared using the scripts provided in Caffe. To be specific, the caffenet model and ilsvrc2012 dataset. The download scripts can be easily found in their folders.

### Classification_01

This code is derived and rewritten from Caffe's cpp-classification [example](https://github.com/BVLC/caffe/tree/master/examples/cpp_classification). The example is executed using:

```bash
./scripts/run_01.sh
```

The terminal output should be identical with Caffe's example.

```bash
---------- Prediction for examples/images/cat.jpg ----------
0.3134 - "n02123045 tabby, tabby cat"
0.2380 - "n02123159 tiger cat"
0.1235 - "n02124075 Egyptian cat"
0.1003 - "n02119022 red fox, Vulpes vulpes"
0.0715 - "n02127052 lynx, catamount"
```

### Classification_02

The minimal example. It uses the same model and test data as the previous example. The example is executed using:

```bash
./scripts/run_02.sh
```
The output should look like:
```bash
[I][02/21|22:39:15] Predicted score: 0.313388 index: 281 label: n02123045 tabby, tabby cat
```

## Todo

Supporting PReLU layer in C++ directly by either plugin layer or layer transformation, which transforms the PReLU layer into combination of ReLU, scale and sum layer.

Other TensorRT components such as PluginFactory, Profiler, ....
