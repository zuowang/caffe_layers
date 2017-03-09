#ifndef CAFFE_SET_LOSS_LAYER_HPP_
#define CAFFE_SET_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {


template <typename Dtype>
class CoupledClusterLossLayer : public LossLayer<Dtype> {
    public:
        explicit CoupledClusterLossLayer(const LayerParameter& param)
            : LossLayer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "CoupledClusterLoss"; }
        virtual inline int ExactNumTopBlobs() const { return 1; }
        virtual inline int ExactNumBottomBlobs() const { return 2; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

        // probe id and gallery ids
        std::vector<std::vector<int> > pos_ids, neg_ids;
        std::vector<bool> neg_backward;
        std::vector<bool> pos_backward;
        Dtype margin;
        Dtype scale;
        int N;
        int group_num;
        int feat_len;
        bool log_flag;

        /* center of all positive samples */
        Blob<Dtype> pos_center_;
        Blob<Dtype> diff_;
        Blob<Dtype> dist_sq_;
};
}
#endif
