#ifndef CAFFE_PIXELCONV_LAYER_HPP_
#define CAFFE_PIXELCONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/proto/caffe.pb.h"
#include <math.h>

namespace caffe {

/**
  @brief Convolution operation of two input Blobs
   Input 1: learned per-pixel kernel (weight) bottom[0];
   Input 2: Input Data or Input Label bottom[1];
**/

template <typename Dtype>
class PixelConvLayer : public Layer<Dtype> {
 public:
  explicit PixelConvLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "PixelConv"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int num_, channels_, height_, width_;
  int kernel_size, pad_size;
  bool is_pad_;
  bool is_bpk_;
  bool is_bpd_;

  Blob<Dtype> col_buffer_;
  Blob<Dtype> dat_buffer_;
  Blob<Dtype> weight_buffer_;
};

}  // namespace caffe

#endif  // CAFFE_PIXELCONV_LAYER_HPP_
