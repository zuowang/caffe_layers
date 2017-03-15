#include <vector>

#include "caffe/layers/aggregate_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AggregateLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype * top_data = top[0]->mutable_gpu_data();
  int channels = bottom[0]->channels();
  caffe_gpu_set(top[0]->count(), Dtype(0), top[0]->mutable_gpu_data());
  for(int i = 0; i < bottom[0]->num(); ++i){
    caffe_gpu_add(channels, bottom[0]->gpu_data() + i*channels, top_data, top_data);
  }
}

template <typename Dtype>
void AggregateLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int channels = bottom[0]->channels();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  for(int i = 0;i < bottom[0]->num(); ++i){
    caffe_gpu_memcpy(channels, top[0]->gpu_diff(), bottom_diff + i*channels);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(AggregateLayer);

}  // namespace caffe
