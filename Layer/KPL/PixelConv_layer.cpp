#include <algorithm>
#include <vector>

#include "caffe/layers/PixelConv_layer.hpp"

namespace caffe {

template <typename Dtype>
void PixelConvLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    PixelConvParameter pixelconv_param = this->layer_param_.pixelconvolution_param();
    is_pad_ = pixelconv_param.is_pad();
    is_bpk_ = pixelconv_param.is_bpk();
    is_bpd_ = pixelconv_param.is_bpd();

}

template <typename Dtype>
void PixelConvLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    // weight & data should have the same size 
    CHECK_EQ(bottom[0]->height(), bottom[1]->height())
          << "The weight and data must have the same size.";

    CHECK_EQ(bottom[0]->width(), bottom[1]->width())
          << "The weight and data must have the same size.";
  
    // data shape
    num_ = bottom[1]->num();
    channels_ = bottom[1]->channels();
    height_ = bottom[1]->height();
    width_ = bottom[1]->width();

     // the channels of weight is [power(n) * channels_], where n is the size of kernel
     kernel_size = sqrt((bottom[0]->channels()/channels_));

    // padding or not
    if (is_pad_) {
        pad_size = static_cast<int>(floor(static_cast<float>(kernel_size) / 2));
    }

    // shape
    col_buffer_.Reshape(1, 1, kernel_size * kernel_size, height_ * width_);
    dat_buffer_.Reshape(1, 1, kernel_size * kernel_size, height_ * width_);
    weight_buffer_.Reshape(1, 1, 1, kernel_size * kernel_size);
    top[0]->Reshape(num_, channels_, height_, width_);

    // initialization
    caffe_set(weight_buffer_.count(), Dtype(1), weight_buffer_.mutable_cpu_data());
}

template <typename Dtype>
void PixelConvLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    const Dtype* weights = bottom[0]->cpu_data();         // learned weight 
    const Dtype* bottom_data = bottom[1]->cpu_data();     // input data
    Dtype* top_data = top[0]->mutable_cpu_data();

    for(int n = 0; n < num_; ++n) {
        for (int c = 0; c < channels_; ++c) {      
            im2col_cpu(bottom_data, 1, height_, width_, kernel_size, kernel_size, 
                pad_size, pad_size, 1, 1, 1, 1, col_buffer_.mutable_cpu_data());
      
            caffe_mul(dat_buffer_.count(), weights, 
                col_buffer_.cpu_data(), dat_buffer_.mutable_cpu_data());

            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, (Dtype)1.,
                height_ * width_, kernel_size * kernel_size,
                (Dtype)1., weight_buffer_.cpu_data(), dat_buffer_.cpu_data(), (Dtype)0., top_data);

            bottom_data += bottom[1]->offset(0, 1);
            top_data += top[0]->offset(0, 1);
            weights += bottom[0]->offset(0, kernel_size * kernel_size);
        }
    }
}

template <typename Dtype>
void PixelConvLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    const Dtype* weights = bottom[0]->cpu_data();          // learned weight
    Dtype* weights_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* bottom_data = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[1]->mutable_cpu_diff(); 
    const Dtype* top_diff = top[0]->cpu_diff();

    caffe_set(bottom[0]->count(), Dtype(0), weights_diff);
    caffe_set(bottom[1]->count(), Dtype(0), bottom_diff);

    for (int n = 0; n < num_; ++n) {
        for (int c = 0; c < channels_; ++c) {
            // gradient w.r.t dat_buffer_
            caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_size * kernel_size,
                height_ * width_, (Dtype)1., (Dtype)1., weight_buffer_.cpu_data(), top_diff, 
                (Dtype)0., dat_buffer_.mutable_cpu_diff());

            // gradient w.r.t weight
            im2col_cpu(bottom_data, 1, height_, width_, kernel_size, kernel_size, 
                pad_size, pad_size, 1, 1, 1, 1, col_buffer_.mutable_cpu_data());
            
	    if (is_bpk_) {
              caffe_mul(dat_buffer_.count(), col_buffer_.cpu_data(), 
                  dat_buffer_.cpu_diff(), weights_diff); 
	    }

            // gradient w.r.t bottom_data  
	    if (is_bpd_) {
              caffe_mul(dat_buffer_.count(), weights, 
                  dat_buffer_.cpu_diff(), col_buffer_.mutable_cpu_diff()); 

              col2im_cpu(col_buffer_.cpu_diff(), 1, height_, width_, kernel_size, kernel_size,
                  pad_size, pad_size, 1, 1, 1, 1, bottom_diff);
	    }

            bottom_data += bottom[1]->offset(0, 1);
            bottom_diff += bottom[1]->offset(0, 1);
            top_diff += top[0]->offset(0, 1);
            weights += bottom[0]->offset(0, kernel_size * kernel_size);
            weights_diff += bottom[0]->offset(0, kernel_size * kernel_size);
        }
    }
}


#ifdef CPU_ONLY
STUB_GPU(PixelConvLayer);
#endif

INSTANTIATE_CLASS(PixelConvLayer);
REGISTER_LAYER_CLASS(PixelConv);

}  // namespace caffe
