#include <algorithm>
#include <set>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/coupled_cluster_loss_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CoupledClusterLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  N = this->layer_param_.coupled_cluster_loss_param().group_size();
  margin = this->layer_param_.coupled_cluster_loss_param().margin();
  scale = this->layer_param_.coupled_cluster_loss_param().scale();
  log_flag = this->layer_param_.coupled_cluster_loss_param().log_flag();
  LOG(INFO) << "Set loss scale is " << scale;
  // batch size must be multiple times of N
  CHECK_EQ(bottom[0]->num()%N, 0);
  group_num = bottom[0]->num()/N;

  feat_len = bottom[0]->channels();

  CHECK_EQ(bottom.size(), 2);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);

  pos_center_.Reshape(group_num, feat_len, 1, 1);
  diff_.Reshape(N*group_num, feat_len, 1, 1);
  dist_sq_.Reshape(N*group_num, 1, 1, 1);
}

template <typename Dtype>
void CoupledClusterLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->num(), N*group_num);
}

template <typename Dtype>
void CoupledClusterLossLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    pos_ids = std::vector<std::vector<int> >(group_num, std::vector<int>());
    neg_ids = std::vector<std::vector<int> >(group_num, std::vector<int>());
    pos_backward = std::vector<bool>(group_num*N, false);
    neg_backward = std::vector<bool>(group_num*N, false);
    const Dtype *feat_ptr = bottom[0]->cpu_data();
    const Dtype *label_ptr = bottom[1]->cpu_data();
    Dtype *diff_ptr_ = diff_.mutable_cpu_data();
    Dtype loss(0);

    caffe_set(feat_len*group_num, Dtype(0), pos_center_.mutable_cpu_data());

    int cnt = 0;
    /* i -> group index */
    for(int i=0; i<group_num; ++i) {
        /* search for the positive id */
        std::set<Dtype> labels;
        Dtype anchor_id = -1;
        for(int j=0; j<N; ++j) {
            Dtype tmp = label_ptr[N*i+j];
            if(labels.count(tmp)>0) {
                anchor_id = tmp;
                break;
            }
            else
                labels.insert(tmp);
        }
        // CHECK_NE(anchor_id, -1);
        /* collect for positive and negative ids, compute the center of positive samples */
        for(int j=0; j<N; ++j) {
            if(label_ptr[i*N+j]==anchor_id){
                pos_ids[i].push_back(j);
                caffe_add(feat_len, feat_ptr+feat_len*(i*N+j), pos_center_.mutable_cpu_data()+feat_len*i, pos_center_.mutable_cpu_data()+feat_len*i);
            }
            else neg_ids[i].push_back(j);
        }
        caffe_cpu_scale(feat_len, Dtype(1)/pos_ids[i].size(), pos_center_.mutable_cpu_data()+feat_len*i, pos_center_.mutable_cpu_data()+feat_len*i);

        if(neg_ids[i].size()==0 || pos_ids[i].size()<=1) continue;

        Dtype pos_mdist = Dtype(0);
        Dtype neg_min_val = -1;
        int neg_min_ind = -1;
        Dtype pos_max_val = -1;
        for(int j=0; j<N; ++j) {
            // f[j]-center
            caffe_sub(feat_len, feat_ptr+feat_len*(i*N+j), pos_center_.cpu_data()+feat_len*i, diff_ptr_+feat_len*(i*N+j));
            if(scale!=1)
                caffe_cpu_scale(feat_len, scale, diff_ptr_+feat_len*(i*N+j), diff_ptr_+feat_len*(i*N+j));
            Dtype d = caffe_cpu_dot(feat_len, diff_ptr_+feat_len*(i*N+j), diff_ptr_+feat_len*(i*N+j));
            if(log_flag)
                LOG(INFO) << "i " << i << ", j " << j << ", d " << d;
            dist_sq_.mutable_cpu_data()[i*N+j] = d;
            if(std::count(neg_ids[i].begin(), neg_ids[i].end(), j)>0 && (neg_min_val==-1 || d<neg_min_val)) {
                neg_min_val = d;
                neg_min_ind = i*N+j;
            }
            else if(std::count(neg_ids[i].begin(), neg_ids[i].end(), j)==0 && (pos_max_val==-1 || d>pos_max_val)) pos_max_val = d;
        }
        neg_backward[neg_min_ind] = true;
        for(int j=0; j<N; ++j) {
            if(std::count(neg_ids[i].begin(), neg_ids[i].end(), j)>0)
                continue;
            else {
                Dtype d = dist_sq_.cpu_data()[i*N+j];
                Dtype mdist = std::max(d+margin-neg_min_val, Dtype(0));
                if(log_flag)
                    LOG(INFO) << "j=" << j << ", d=" << d << ", neg_min_val=" << neg_min_val << ", mdist=" << mdist;
                if(mdist>0) pos_backward[i*N+j] = true;
                pos_mdist += mdist;
            }
        }
        /* average punishment */
        pos_mdist /= pos_ids[i].size();
        // pos_mdist *= 2;

        if(log_flag)
            LOG(INFO) << "pos_mdist " << pos_mdist << ", neg_min_val " << neg_min_val;

        CHECK_GE(pos_ids[i].size(), 2);
        CHECK_GE(neg_ids[i].size(), 1);

        loss += pos_mdist;
        ++cnt;
    }
    loss = loss / cnt;
    top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void CoupledClusterLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    /* loss_weight */
    const Dtype alpha = top[0]->cpu_diff()[0]/group_num;
    CHECK_EQ(feat_len, bottom[0]->channels());
    CHECK_EQ(N*group_num*feat_len, bottom[0]->count());
    if(propagate_down[0]) {
        Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();
        caffe_set(N*group_num*feat_len, Dtype(0), bottom_diff);
        for(int i=0; i<group_num; ++i) {
            for(int j=0; j<N; ++j) {
                if(pos_backward[i*N+j])
                    /* for positive samples */
                    caffe_cpu_axpby(feat_len, scale*alpha, diff_.cpu_data()+feat_len*(i*N+j), Dtype(0), bottom_diff+feat_len*(i*N+j));
                else if(neg_backward[i*N+j]) {
                    /* for hard negative sample */
                    caffe_cpu_axpby(feat_len, -scale*alpha, diff_.cpu_data()+feat_len*(i*N+j), Dtype(0), bottom_diff+feat_len*(i*N+j));
                }
            }
        }
    }
}

#ifdef CPU_ONLY
STUB_GPU(CoupledClusterLossLayer);
#endif
INSTANTIATE_CLASS(CoupledClusterLossLayer);
REGISTER_LAYER_CLASS(CoupledClusterLoss);

}
