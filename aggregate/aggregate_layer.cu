#include <vector>

#include "caffe/layers/aggregate_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AggregateLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    if (bottom.size() == 1) { return; }
    caffe_set(sizeof(Dtype) * top[0]->count(), Dtype(0), top[0]->mutable_gpu_data());
    for(int i = 0;i < bottom.size();++i){
        caffe_add(sizeof(Dtype) * bottom[i]->count(), bottom[i]->gpu_data(), top[0]->mutable_gpu_data(), top[0]->mutable_gpu_data());
    }
}

template <typename Dtype>
void AggregateLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (bottom.size() == 1) { return; }
    int offset = 0;
    for(int i = 0;i < bottom.size();++i){
        caffe_gpu_memcpy(sizeof(Dtype) * bottom[i]->count(), top[0]->gpu_diff() + offset, bottom[i]->mutable_gpu_diff());
        offset += bottom[i]->count();
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(AggregateLayer);

}  // namespace caffe
