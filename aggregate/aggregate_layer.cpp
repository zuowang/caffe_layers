
#include "caffe/layers/aggregate_layer.hpp"
#include <caffe/util/math_functions.hpp>
#include <vector>
namespace caffe{
template <typename Dtype>
void AggregateLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	num_channels_ = bottom[0]->channels();
	num_images_ = bottom[0]->num();
}
template <typename Dtype>
void AggregateLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	top[0]->Reshape(1,num_channels_,1,1);
}
template <typename Dtype>
void AggregateLayer<Dtype>::Forward_cpu(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	Dtype * top_data = top[0]->mutable_cpu_data();
	const Dtype* bottom_data = bottom[0]->cpu_data();
	
	vector<float>tmp;
	
	for (int i = 0; i < bottom[0]->channels(); i++){
		float sum = 0;
		for (int j = 0; j < bottom[0]->num(); j++){
			int idx = j*bottom[0]->channels() + i;
			sum += bottom_data[idx];
			if (j == bottom[0]->num() - 1){
				tmp.push_back(sum);
			}
		}
	}
	for (int i = 0; i < bottom[0]->channels(); i++){
		top_data[i] = tmp[i];
	}
	tmp.clear();
}
INSTANTIATE_CLASS(AggregateLayer);
REGISTER_LAYER_CLASS(Aggregate);
}
