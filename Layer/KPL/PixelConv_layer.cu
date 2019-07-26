#include <algorithm>
#include <vector>

#include "caffe/layers/PixelConv_layer.hpp"

namespace caffe {

template <typename Dtype>
void PixelConvLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {    

    const Dtype* weights = bottom[0]->gpu_data();         // learned weight
    const Dtype* bottom_data = bottom[1]->gpu_data();     // input data
    Dtype* top_data = top[0]->mutable_gpu_data();

    for(int n = 0; n < num_; ++n) {
        for (int c = 0; c < channels_; ++c) {      
            im2col_gpu(bottom_data, 1, height_, width_, kernel_size, kernel_size, 
                pad_size, pad_size, 1, 1, 1, 1, col_buffer_.mutable_gpu_data());

            caffe_gpu_mul(dat_buffer_.count(), weights, 
                col_buffer_.gpu_data(), dat_buffer_.mutable_gpu_data());

            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, (Dtype)1.,
                height_ * width_, kernel_size * kernel_size,
                (Dtype)1., weight_buffer_.gpu_data(), dat_buffer_.gpu_data(), (Dtype)0., top_data);

            bottom_data += bottom[1]->offset(0, 1);
            top_data += top[0]->offset(0, 1);
            weights += bottom[0]->offset(0, kernel_size * kernel_size);
        }
    }
}

template <typename Dtype>
void PixelConvLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    const Dtype* weights = bottom[0]->gpu_data();          // learned weight
    Dtype* weights_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* bottom_data = bottom[1]->gpu_data();
    Dtype* bottom_diff = bottom[1]->mutable_gpu_diff(); 
    const Dtype* top_diff = top[0]->gpu_diff();

    caffe_gpu_set(bottom[0]->count(), Dtype(0), weights_diff);
    caffe_gpu_set(bottom[1]->count(), Dtype(0), bottom_diff);

    for (int n = 0; n < num_; ++n) {
        for (int c = 0; c < channels_; ++c) {
            // gradient w.r.t dat_buffer_
            caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_size * kernel_size,
                height_ * width_, (Dtype)1., (Dtype)1., weight_buffer_.gpu_data(), top_diff, 
                (Dtype)0., dat_buffer_.mutable_gpu_diff());

            // gradient w.r.t weight
            im2col_gpu(bottom_data, 1, height_, width_, kernel_size, kernel_size, 
                pad_size, pad_size, 1, 1, 1, 1, col_buffer_.mutable_gpu_data());

	    if (is_bpk_) {
              caffe_gpu_mul(dat_buffer_.count(), col_buffer_.gpu_data(), 
                  dat_buffer_.gpu_diff(), weights_diff);
	    }

            // gradient w.r.t bottom_data  
	    if (is_bpd_) {
              caffe_gpu_mul(dat_buffer_.count(), weights, 
                  dat_buffer_.gpu_diff(), col_buffer_.mutable_gpu_diff());


              col2im_gpu(col_buffer_.gpu_diff(), 1, height_, width_, kernel_size, kernel_size,
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

INSTANTIATE_LAYER_GPU_FUNCS(PixelConvLayer);

}  // namespace caffe
