#include <algorithm>
#include <vector>

#include "caffe/layers/aggregate_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AggregateLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void AggregateLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(1, bottom[0]->channels(), 1, 1);
}

template <typename Dtype>
void AggregateLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype * top_data = top[0]->mutable_cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();

  vector<float>tmp;

  for (int i = 0; i < bottom[0]->channels(); i++) {
    float sum = 0;
    for (int j = 0; j < bottom[0]->num(); j++) {
      int idx = j*bottom[0]->channels() + i;
      sum += bottom_data[idx];
      if (j == bottom[0]->num() - 1) {
        tmp.push_back(sum);
      }
    }
  }
  for (int i = 0; i < bottom[0]->channels(); i++) {
    top_data[i] = tmp[i];
  }
  tmp.clear();
}

template <typename Dtype>
void AggregateLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype * top_diff = top[0]->cpu_diff();
  int channels = bottom[0]->channels();
  for (int i = 0; i < bottom[0]->num(); i++){
    caffe_copy(channels, top_diff, bottom_diff + i*channels);
  }
}

#ifdef CPU_ONLY
STUB_GPU(AggregateLayer);
#endif

INSTANTIATE_CLASS(AggregateLayer);
REGISTER_LAYER_CLASS(Aggregate);

}  // namespace caffe
