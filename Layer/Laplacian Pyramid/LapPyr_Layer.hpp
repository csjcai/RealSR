#ifndef CAFFE_LAPPYR_LAYER_HPP_
#define CAFFE_LAPPYR_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/proto/caffe.pb.h"
#include <math.h>

namespace caffe {

template <typename Dtype>
class LapPyrLayer : public Layer<Dtype> {
 public:
  explicit LapPyrLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "LapPyr"; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  //virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  //    const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //   const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int kernel, pad;
  bool _is_down_; 

  int num_, channels_;
  int height1_, width1_;
  int height2_, width2_;
  int height3_, width3_;

  Blob<Dtype> col_buffer1_;
  Blob<Dtype> col_buffer2_;

  Blob<Dtype> weight1_;
  Blob<Dtype> weight2_;

  Blob<Dtype> temp1_;
  Blob<Dtype> temp2_;
  Blob<Dtype> temp_up1_;
  Blob<Dtype> temp_up2_;

};

}  // namespace caffe

#endif
