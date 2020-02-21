# TRTNetwork

TensorRT C++ wrapper for Caffe-based neural network.

## Usage

This C++ wrapper is designed with simplicity in mind. A neural network instance can be easily created within a few lines of code.

```cpp
#include "TRTNetwork/TRTNetwork.hpp"

// Instantiate the network instance
trt::TRTNetwork network("caffenet", proto, weights, {"prob"}, {"data"});

// Trigger network inference
network.forward(batch, {{"prob", prob_ptr}, {"data", data_ptr}});

```
TensorRT wrapper API is detailedly documented in the header files.

## Build

```bash
./scripts/make.sh
```

## Run Examples

The examples in this repo require an environmental variable `CAFFE_ROOT` pointing to the Caffe repo path.  

```bash
export CAFFE_ROOT=${CAFFE_REPO_PATH}
```

Besides, the test data and model should be prepared using the scripts provided in Caffe. To be specific, the caffenet model and ilsvrc2012 dataset.

### Classification_01

This code is derived and rewritten from Caffe's cpp-classification [example][https://github.com/BVLC/caffe/tree/master/examples/cpp_classification]. The terminal output should be identical with Caffe's example.

```bash
./scripts/run_01.sh
```

### Classification_02

The minimal example. The code corresponds to Caffe's ipython notebook [example][https://github.com/BVLC/caffe/blob/master/examples/00-classification.ipynb].

```bash
./scripts/run_02.sh
```
