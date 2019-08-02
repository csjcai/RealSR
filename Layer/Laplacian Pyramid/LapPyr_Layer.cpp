#include <algorithm>
#include <vector>

#include "caffe/layers/LapPyr_Layer.hpp"

namespace caffe {

template <typename Dtype>
void LapPyrLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    LapPyrParameter lapPyr = this->layer_param_.laplacianpyramid_param();
    _is_down_ = lapPyr.is_down();
}

template <typename Dtype>
void LapPyrLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    weight1_.Reshape(1, 1, 5, 5);
    weight2_.Reshape(1, 1, 5, 5);

    // kernel
    kernel = 5;
    pad = static_cast<int>(floor(static_cast<float>(kernel) / 2));

    // LOG(INFO) << kernel;

    if (_is_down_) {
        // data shape
        num_       = bottom[0]->num();
        channels_  = bottom[0]->channels();
        height1_   = bottom[0]->height();
        width1_    = bottom[0]->width();

        height2_   = height1_/2;
        width2_    = width1_/2;

        height3_   = height1_/4;
        width3_    = width1_/4;

        // shape
        col_buffer1_.Reshape(1, 1, kernel * kernel, height1_ * width1_);
        col_buffer2_.Reshape(1, 1, kernel * kernel, height2_ * width2_);

        temp1_.Reshape(1, 1, height1_, width1_);
        temp2_.Reshape(1, 1, height2_, width2_);
        
        temp_up1_.Reshape(1, 1, height1_, width1_);
        temp_up2_.Reshape(1, 1, height2_, width2_);

        top[0]->Reshape(num_, channels_, height1_, width1_);
        top[1]->Reshape(num_, channels_, height2_, width2_);
        top[2]->Reshape(num_, channels_, height3_, width3_);
    }
    else {
        // data shape
        num_       = bottom[0]->num();
        channels_  = bottom[0]->channels();
        height1_   = bottom[0]->height();
        width1_    = bottom[0]->width();
	
	//LOG(INFO) << height1_;
        
	height2_   = bottom[1]->height();
        width2_    = bottom[1]->width();

	//LOG(INFO) << height2_;

        height3_   = bottom[2]->height();
        width3_    = bottom[2]->width();

	//LOG(INFO) << height3_;

        // shape
        col_buffer1_.Reshape(1, 1, kernel * kernel, height1_ * width1_);
        col_buffer2_.Reshape(1, 1, kernel * kernel, height2_ * width2_);

        temp1_.Reshape(1, 1, height1_, width1_);
        temp2_.Reshape(1, 1, height2_, width2_);
        
        temp_up1_.Reshape(1, 1, height1_, width1_);
        temp_up2_.Reshape(1, 1, height2_, width2_);

        top[0]->Reshape(num_, channels_, height1_, width1_);
    }
}

template <typename Dtype>
void LapPyrLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    Dtype* weight1 = weight1_.mutable_cpu_data();
    Dtype* weight2 = weight2_.mutable_cpu_data();

    Dtype tmp[25] = {1, 4,  6,  4,  1,
                     4, 16, 24, 16, 4,
                     6, 24, 36, 24, 6,
                     4, 16, 24, 16, 4,
                     1, 4,  6,  4,  1};

    for (int h = 0; h < 5; h++) {
        for (int w = 0; w < 5; w++) {
            weight1[h * 5 + w] = tmp[h * 5 + w] / 256;
            weight2[h * 5 + w] = tmp[h * 5 + w] / 64;
        }
    }

    Dtype* col_buff1 = col_buffer1_.mutable_cpu_data();
    Dtype* temp1 = temp1_.mutable_cpu_data();
    Dtype* col_buff2 = col_buffer2_.mutable_cpu_data();
    Dtype* temp2 = temp2_.mutable_cpu_data();

    Dtype* temp_up1 = temp_up1_.mutable_cpu_data();
    Dtype* temp_up2 = temp_up2_.mutable_cpu_data();

    if (_is_down_) {
        int count = bottom[0]->count();
        const Dtype* bottom_data = bottom[0]->cpu_data();        //input data
        caffe_copy(count, bottom[0]->cpu_data(), top[0]->mutable_cpu_data());

        Dtype* top_data1 = top[0]->mutable_cpu_data();
        Dtype* top_data2 = top[1]->mutable_cpu_data();
        Dtype* top_data3 = top[2]->mutable_cpu_data();

        for(int n = 0; n < num_; ++n) {
            for (int c = 0; c < channels_; ++c) {

                // scale 2
                im2col_cpu(bottom_data, 1, height1_, width1_, kernel, kernel, 
                    pad, pad, 1, 1, 1, 1, col_buffer1_.mutable_cpu_data());
      
                col_buff1 = col_buffer1_.mutable_cpu_data();

                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, (Dtype)1.,
                    height1_ * width1_, kernel * kernel,
                    (Dtype)1., weight1, col_buff1, (Dtype)0., temp1_.mutable_cpu_data());

                temp1 = temp1_.mutable_cpu_data();

                for (int i = 0; i < height2_; i++) {
                    for (int j = 0; j < width2_; j++) {
                        top_data2[i * width2_ + j] = temp1[(i * 2) * (width2_ * 2) + (j * 2)];
                    }
                }
                // scale 2
                

                // scale 3
                im2col_cpu(top_data2, 1, height2_, width2_, kernel, kernel, 
                    pad, pad, 1, 1, 1, 1, col_buffer2_.mutable_cpu_data());
      
                col_buff2 = col_buffer2_.mutable_cpu_data();

                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, (Dtype)1.,
                    height2_ * width2_, kernel * kernel,
                    (Dtype)1., weight1, col_buff2, (Dtype)0., temp2_.mutable_cpu_data());

                temp2 = temp2_.mutable_cpu_data();

                for (int i = 0; i < height3_; i++) {
                    for (int j = 0; j < width3_; j++) {
                        top_data3[i * width3_ + j] = temp2[(i * 2) * (width3_ * 2) + (j * 2)];
                    }
                }
                // scale 3

                // scale 1
                // upsample scale 2
                temp_up1 = temp_up1_.mutable_cpu_data();
                for (int i = 0; i < height2_; i++) {
                    for (int j = 0; j < width2_; j++) {
                        temp_up1[(i * 2) * (width2_ * 2) + (j * 2)] = top_data2[i * width2_ + j];
                        temp_up1[(i * 2) * (width2_ * 2) + ((j * 2) + 1)] = 0;
                        temp_up1[((i * 2) + 1) * (width2_ * 2) + (j * 2)] = 0;
                        temp_up1[((i * 2) + 1) * (width2_ * 2) + ((j * 2) + 1)] = 0;
                    }
                }

                im2col_cpu(temp_up1, 1, height1_, width1_, kernel, kernel, 
                    pad, pad, 1, 1, 1, 1, col_buffer1_.mutable_cpu_data());
      
                col_buff1 = col_buffer1_.mutable_cpu_data();

                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, (Dtype)1.,
                    height1_ * width1_, kernel * kernel,
                    (Dtype)1., weight2, col_buff1, (Dtype)0., temp1_.mutable_cpu_data());

                temp1 = temp1_.mutable_cpu_data();

                for (int i = 0; i < height1_; i++) {
                    for (int j = 0; j < width1_; j++) {
                        top_data1[i * width1_ + j] = top_data1[i * width1_ + j] - temp1[i * width1_ + j];
                    }
                }
                // scale 1
                

                // scale 2
                // upsample scale 3
                temp_up2 = temp_up2_.mutable_cpu_data();
                for (int i = 0; i < height3_; i++) {
                    for (int j = 0; j < width3_; j++) {
                        temp_up2[(i * 2) * (width3_ * 2) + (j * 2)] = top_data3[i * width3_ + j];
                        temp_up2[(i * 2) * (width3_ * 2) + ((j * 2) + 1)] = 0;
                        temp_up2[((i * 2) + 1) * (width3_ * 2) + (j * 2)] = 0;
                        temp_up2[((i * 2) + 1) * (width3_ * 2) + ((j * 2) + 1)] = 0;
                    }
                }

                im2col_cpu(temp_up2, 1, height2_, width2_, kernel, kernel, 
                    pad, pad, 1, 1, 1, 1, col_buffer2_.mutable_cpu_data());
      
                col_buff2 = col_buffer2_.mutable_cpu_data();

                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, (Dtype)1.,
                    height2_ * width2_, kernel * kernel,
                    (Dtype)1., weight2, col_buff2, (Dtype)0., temp2_.mutable_cpu_data());

                temp2 = temp2_.mutable_cpu_data();

                for (int i = 0; i < height2_; i++) {
                    for (int j = 0; j < width2_; j++) {
                        top_data2[i * width2_ + j] = top_data2[i * width2_ + j] - temp2[i * width2_ + j];
                    }
                }
                // scale 2

                bottom_data += bottom[0]->offset(0,1);
                top_data1 += top[0]->offset(0,1);
                top_data2 += top[1]->offset(0,1);
                top_data3 += top[2]->offset(0,1);
            }
        }
    }
    else {
        const Dtype* bottom_data1 = bottom[0]->cpu_data(); //input data
        const Dtype* bottom_data2 = bottom[1]->cpu_data(); //input data
        const Dtype* bottom_data3 = bottom[2]->cpu_data(); //input data

        Dtype* top_data = top[0]->mutable_cpu_data();

        for(int n = 0; n < num_; ++n) {
            for (int c = 0; c < channels_; ++c) {
                // scale 2
                // upsample scale 3
                Dtype* temp_up2 = temp_up2_.mutable_cpu_data();
                for (int i = 0; i < height3_; i++) {
                    for (int j = 0; j < width3_; j++) {
                        temp_up2[(i * 2) * (width3_ * 2) + (j * 2)] = bottom_data3[i * width3_ + j];
                        temp_up2[(i * 2) * (width3_ * 2) + ((j * 2) + 1)] = 0;
                        temp_up2[((i * 2) + 1) * (width3_ * 2) + (j * 2)] = 0;
                        temp_up2[((i * 2) + 1) * (width3_ * 2) + ((j * 2) + 1)] = 0;
                    }
                }

                im2col_cpu(temp_up2, 1, height2_, width2_, kernel, kernel, 
                    pad, pad, 1, 1, 1, 1, col_buffer2_.mutable_cpu_data());
      
                col_buff2 = col_buffer2_.mutable_cpu_data();

                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, (Dtype)1.,
                    height2_ * width2_, kernel * kernel,
                    (Dtype)1., weight2, col_buff2, (Dtype)0., temp2_.mutable_cpu_data());

                temp2 = temp2_.mutable_cpu_data();

                for (int i = 0; i < height2_; i++) {
                    for (int j = 0; j < width2_; j++) {
                        temp2[i * width2_ + j] = bottom_data2[i * width2_ + j] + temp2[i * width2_ + j];
                    }
                }
                // scale 2
                // 
                // scale 1
                // upsample scale 2
                temp_up1 = temp_up1_.mutable_cpu_data();
                for (int i = 0; i < height2_; i++) {
                    for (int j = 0; j < width2_; j++) {
                        temp_up1[(i * 2) * (width2_ * 2) + (j * 2)] = temp2[i * width2_ + j];
                        temp_up1[(i * 2) * (width2_ * 2) + ((j * 2) + 1)] = 0;
                        temp_up1[((i * 2) + 1) * (width2_ * 2) + (j * 2)] = 0;
                        temp_up1[((i * 2) + 1) * (width2_ * 2) + ((j * 2) + 1)] = 0;
                    }
                }

                im2col_cpu(temp_up1, 1, height1_, width1_, kernel, kernel, 
                    pad, pad, 1, 1, 1, 1, col_buffer1_.mutable_cpu_data());
      
                col_buff1 = col_buffer1_.mutable_cpu_data();

                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, (Dtype)1.,
                    height1_ * width1_, kernel * kernel,
                    (Dtype)1., weight2, col_buff1, (Dtype)0., temp1_.mutable_cpu_data());

                temp1 = temp1_.mutable_cpu_data();

                for (int i = 0; i < height1_; i++) {
                    for (int j = 0; j < width1_; j++) {
                        top_data[i * width1_ + j] = bottom_data1[i * width1_ + j] + temp1[i * width1_ + j];
                    }
                }
                // scale 1
                bottom_data1 += bottom[0]->offset(0,1);
                bottom_data2 += bottom[1]->offset(0,1);
                bottom_data3 += bottom[2]->offset(0,1);
                top_data += top[0]->offset(0,1);
            }
        }


    } 


}

template <typename Dtype>
void LapPyrLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    NOT_IMPLEMENTED;

}


#ifdef CPU_ONLY
STUB_GPU(LapPyrLayer);
#endif

INSTANTIATE_CLASS(LapPyrLayer);
REGISTER_LAYER_CLASS(LapPyr);

}  // namespace caffe
