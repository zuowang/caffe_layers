// ------------------------------------------------------------------
// Subcategory CNN
// Copyright (c) 2015 CVGL Stanford
// Licensed under The MIT License
// Written by Yu Xiang
// ------------------------------------------------------------------

# include <cfloat>
# include <ctime>
# include <stdio.h>
# include <math.h>
# include <string>

#include "caffe/roi_generating_layers.hpp"

namespace caffe {

template <typename Dtype>
void FeatureExtrapolatingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  FeatureExtrapolatingParameter feature_extrapolating_param = this->layer_param_.feature_extrapolating_param();

  CHECK_GT(feature_extrapolating_param.num_per_octave(), 0)
      << "number of scales per octave must be > 0";

  num_per_octave_ = feature_extrapolating_param.num_per_octave();
  scale_string_ = feature_extrapolating_param.scale_string();
  num_scale_base_ = feature_extrapolating_param.num_scale_base();
  num_scale_ = (num_scale_base_ - 1) * num_per_octave_ + 1;
}

template <typename Dtype>
void FeatureExtrapolatingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  // parse scale string
  scales_base_.Reshape(num_scale_base_, 1, 1, 1);
  double *scales_base = scales_base_.mutable_cpu_data();
  std::size_t sz;
  string str = scale_string_;
  for(int i = 0; i < num_scale_base_; i++)
  {
    scales_base[i] = std::stod(str, &sz);
    str = str.substr(sz);
  }

  // compute scales
  scales_.Reshape(num_scale_, 1, 1, 1);
  double* scales = scales_.mutable_cpu_data();
  for(int i = 0; i < num_scale_; i++)
  {
    int index_scale_base = i / num_per_octave_;
    double sbase = scales_base[index_scale_base];
    int j = i % num_per_octave_;
    double step = 0;
    if(j == 0)
      scales[i] = sbase;
    else
    {
      double sbase_next = scales_base[index_scale_base+1];
      step = (sbase_next - sbase) / num_per_octave_;
      scales[i] = sbase + j * step;
    }
  }

  // counting
  num_image_ = num_ / num_scale_base_;
  num_top_ = num_image_ * num_scale_;

  // extrapolated features
  top[0]->Reshape(num_top_, channels_, height_, width_);

  // tracing information
  channels_trace_ = 8;
  trace_.Reshape(num_top_, channels_trace_, height_, width_);

  // flags of real scales or approximated scales
  is_real_scales_.Reshape(num_scale_, 1, 1, 1);
  int* flags = is_real_scales_.mutable_cpu_data();
  for(int i = 0; i < num_scale_; i++)
    flags[i] = 0;
  for(int i = 0; i < num_scale_base_; i++)
    flags[i * num_per_octave_] = 1;

  // scale mapping
  which_base_scales_.Reshape(num_scale_, 1, 1, 1);
  int* mapping = which_base_scales_.mutable_cpu_data();
  for(int i = 0; i < num_scale_; i++)
    mapping[i] = int(roundf(float(i) / float(num_per_octave_)));

  // rescaling factors
  rescaling_factors_.Reshape(num_scale_, 1, 1, 1);
  double* factors = rescaling_factors_.mutable_cpu_data();
  for(int i = 0; i < num_scale_; i++)
  {
    int scale_base_index = mapping[i];
    double scale_base = scales_base[scale_base_index];
    double scale = scales[i];
    factors[i] = scale / scale_base;
  }
}

template <typename Dtype>
void FeatureExtrapolatingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(0), top_data);

  double* trace_data = trace_.mutable_cpu_data();

  const int* flags = is_real_scales_.cpu_data();
  const int* mapping = which_base_scales_.cpu_data();
  const double* factors = rescaling_factors_.cpu_data();

  // compute extrapolated features
  for(int n = 0; n < num_top_; n++)
  {
    int index_image = n / num_scale_;
    int index_scale = n % num_scale_;
    // flag for approximation or not
    int flag = flags[index_scale];
    // which base scale to use
    int index_scale_base = mapping[index_scale];
    // rescaling factor
    double factor = factors[index_scale];
    // bottom batch image
    int index_batch = index_image * num_scale_base_ + index_scale_base;
    const Dtype* batch_data = bottom_data + bottom[0]->offset(index_batch);

    for (int c = 0; c < channels_; ++c)
    {
      for(int h = 0; h < height_; h++)
      {
        for(int w = 0; w < width_; w++)
        {
          const int index = h * width_ + w;
          if(flag == 1) // no approximation
          {
            top_data[index] = batch_data[index];
            // set tracing info
            if(c == 0)
            {
              for(int i = 0; i < channels_trace_ / 2; i++)
              {
                trace_data[n * channels_trace_ * height_ * width_ + 2 * i * height_ * width_ + index] = index_batch * channels_ * height_ * width_ + index;
                trace_data[n * channels_trace_ * height_ * width_ + (2 * i + 1) * height_ * width_ + index] = 0.25;
              }
            }
          }
          else
          {
            // bilinear interpolation
            double xp = w / factor;
            double yp = h / factor;
            double cx[2], cy[2], ux, uy;
            int xi, yi, dx, dy, i;
            Dtype val;
            if(xp >= 0 && xp < width_ && yp >= 0 && yp < height_)
            {
              xi = (int)floor(xp); 
              yi = (int)floor(yp);
              ux = xp - (double)xi;
              uy = yp - (double)yi;
              cx[0] = ux;
              cx[1] = 1 - ux;
              cy[0] = uy;
              cy[1] = 1 - uy;

              val = 0;
              i = 0;
              for(dx = 0; dx <= 1; dx++)
              {
                for(dy = 0; dy <= 1; dy++)
                {
                  if(xi+dx >= 0 && xi+dx < width_ && yi+dy >= 0 && yi+dy < height_)
                  {
                    val += cx[1-dx] * cy[1-dy] * batch_data[(yi+dy) * width_ + (xi+dx)];
                    if(c == 0)
                    {
                      trace_data[n * channels_trace_ * height_ * width_ + 2 * i * height_ * width_ + index] = index_batch * channels_ * height_ * width_ + (yi+dy) * width_ + (xi+dx);
                      trace_data[n * channels_trace_ * height_ * width_ + (2 * i + 1) * height_ * width_ + index] = cx[1-dx] * cy[1-dy];
                    }
                  }
                  else
                  {
                    if(c == 0)
                    {
                      trace_data[n * channels_trace_ * height_ * width_ + 2 * i * height_ * width_ + index] = -1;
                      trace_data[n * channels_trace_ * height_ * width_ + (2 * i + 1) * height_ * width_ + index] = 0;
                    }
                  }
                  i++;
                }
              }
              top_data[index] = val;
            }
            else
            {
              // set tracing info
              if(c == 0)
              {
                for(int i = 0; i < channels_trace_ / 2; i++)
                {
                  trace_data[n * channels_trace_ * height_ * width_ + 2 * i * height_ * width_ + index] = -1;
                  trace_data[n * channels_trace_ * height_ * width_ + (2 * i + 1) * height_ * width_ + index] = 0;
                }
              }
            }
          }
        }
      }

      // Increment all data pointers by one channel
      batch_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);

    }
  }
}

template <typename Dtype>
void FeatureExtrapolatingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
  if (propagate_down[0]) 
  {
    const Dtype* top_diff = top[0]->cpu_diff();
    const double* trace_data = trace_.cpu_data();
    const int* mapping = which_base_scales_.cpu_data();
    const double* factors = rescaling_factors_.cpu_data();

    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);

    for(int n = 0; n < num_; n++)
    {
      int index_image = n / num_scale_base_;
      int index_scale_base = n % num_scale_base_;

      for (int c = 0; c < channels_; ++c)
      {
        for(int h = 0; h < height_; h++)
        {
          for(int w = 0; w < width_; w++)
          {
            Dtype val = 0;
            for(int i = 0; i < num_scale_; i++)
            {
              if(mapping[i] == index_scale_base)
              {
                int index_batch = index_image * num_scale_ + i;
                double factor = factors[i];
                double xp = w * factor;
                double yp = h * factor;
                int xi = (int)floor(xp); 
                int yi = (int)floor(yp);
        
                for(int dx = -2; dx <= 2; dx++)
                {
                  for(int dy = -2; dy <= 2; dy++)
                  {
                    if(xi+dx >= 0 && xi+dx < width_ && yi+dy >= 0 && yi+dy < height_)
                    {
                      for(int j = 0; j < channels_trace_ / 2; j++)
                      {
                        int index_trace = int(trace_data[index_batch * channels_trace_ * height_ * width_ + 2 * j * height_ * width_ + (yi+dy) * width_ + (xi+dx)]);
                        double weight_trace = trace_data[index_batch * channels_trace_ * height_ * width_ + (2 * j + 1) * height_ * width_ + (yi+dy) * width_ + (xi+dx)];
                        if(index_trace == n * channels_ * height_ * width_ + h * width_ + w)
                          val += weight_trace * top_diff[index_batch * channels_ * height_ * width_ + c * height_ * width_ + (yi+dy) * width_ + (xi+dx)];
                      }
                    }
                  }
                }
              }
            }

            // assign value
            bottom_diff[h * width_ + w] = val;
          }
        }

        // Increment all data pointers by one channel
        bottom_diff += bottom[0]->offset(0, 1);
      }
    }
  }
}

INSTANTIATE_CLASS(FeatureExtrapolatingLayer);
REGISTER_LAYER_CLASS(FeatureExtrapolating);

}  // namespace caffe
