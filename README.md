# caffe_layers
useful caffe layer in python, cpp, or cuda


##normalize_layer and Aggregate_layer used for image retrieval
```
cp layers/normalize_layer/normalize_layer.hpp $CAFFE_HOME/include/caffe/layers/normalize_layer.hpp
cp layers/normalize_layer/normalize_layer.cpp $CAFFE_HOME/src/caffe/layers/normalize_layer.cpp
cp layers/Aggregate_layer/Aggregate_layer.hpp $CAFFE_HOME/include/caffe/layers/Aggregate_layer.hpp
cp layers/Aggregate_layer/Aggregate_layer.cpp $CAFFE_HOME/src/caffe/layers/Aggregate_layer.cpp
```


python implementation of customer layer produces the same results, but is less efficient
