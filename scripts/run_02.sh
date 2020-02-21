# Execute cpp-classification 
./bin/classification_02 \
    ${CAFFE_ROOT}/models/bvlc_reference_caffenet/deploy.prototxt \
    ${CAFFE_ROOT}/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel \
    ${CAFFE_ROOT}/data/ilsvrc12/imagenet_mean.binaryproto \
    ${CAFFE_ROOT}/data/ilsvrc12/synset_words.txt \
    ${CAFFE_ROOT}/examples/images/cat.jpg
