#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/dtow_layer.hpp"

namespace caffe {
	template <typename Dtype>
	__global__ void dtow_kernel(const int nthreads, const Dtype* const bottom_data,
		const int num, const int channels, const int height, const int width,
		const int channels_out, const int height_out, const int width_out, const int patch_size,
		Dtype* const top_data) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int tw = index%width;
			int th = (index / width) % height;
			int tc = (index / width / height) % channels;
			int	tn = index / width / height / channels;
			int p2size = patch_size*patch_size;
			int pc = tc / p2size;
			int rc = tc % p2size;
			int ph = th*patch_size + rc / patch_size;
			int pw = tw*patch_size + rc % patch_size;
			int pidx = ((tn*channels_out + pc)*height_out + ph)*width_out + pw;
			top_data[pidx] = bottom_data[index];

		}
	}
	template <typename Dtype>
	__global__ void wtod_kernel(const int nthreads, const Dtype* const bottom_data,
		const int num, const int channels, const int height, const int width,
		const int channels_out, const int height_out, const int width_out, const int patch_size,
		Dtype* const top_data) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int tw = index%width;
			int th = (index / width) % height;
			int tc = (index / width / height) % channels;
			int	tn = index / width / height / channels;
			int p2size = patch_size*patch_size;
			int ph = th / patch_size;
			int pw = tw / patch_size;
			int pc = tc * p2size + (th%patch_size)*patch_size + tw%patch_size;
			int pidx = ((tn*channels_out + pc)*height_out + ph)*width_out + pw;
			top_data[pidx] = bottom_data[index];

		}
	}
	template <typename Dtype>
	void DtowLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		Dtype* const top_data = top[0]->mutable_gpu_data();
		const Dtype* bottom_data = bottom[0]->gpu_data();
		int count = bottom[0]->count();
		if (d2w){
			dtow_kernel<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
				(count, bottom_data, num_, ch_in, h_in, w_in, ch_out, h_out, w_out, psize, top_data);
		}
		else{
			wtod_kernel<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
				(count, bottom_data, num_, ch_in, h_in, w_in, ch_out, h_out, w_out, psize, top_data);
		}
		
		CUDA_POST_KERNEL_CHECK;
	}
	template <typename Dtype>
	__global__ void dtow_backward_kernel(const int nthreads, const Dtype* const top_diff,
		const int num, const int channels, const int height, const int width,
		const int channels_out, const int height_out, const int width_out, const int patch_size,
		Dtype* const bottom_diff) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int tw = index%width;
			int th = (index / width) % height;
			int tc = (index / width / height) % channels;
			int	tn = index / width / height / channels;
			int p2size = patch_size*patch_size;
			int pc = tc / p2size;
			int rc = tc % p2size;
			int ph = th*patch_size + rc / patch_size;
			int pw = tw*patch_size + rc % patch_size;
			int pidx = ((tn*channels_out + pc)*height_out + ph)*width_out + pw;
			bottom_diff[index] = top_diff[pidx];
		}
	}
	template <typename Dtype>
	__global__ void wtod_backward_kernel(const int nthreads, const Dtype* const top_diff,
		const int num, const int channels, const int height, const int width,
		const int channels_out, const int height_out, const int width_out, const int patch_size,
		Dtype* const bottom_diff) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int tw = index%width;
			int th = (index / width) % height;
			int tc = (index / width / height) % channels;
			int	tn = index / width / height / channels;
			int p2size = patch_size*patch_size;
			int ph = th / patch_size;
			int pw = tw / patch_size;
			int pc = tc * p2size + (th%patch_size)*patch_size + tw%patch_size;
			int pidx = ((tn*channels_out + pc)*height_out + ph)*width_out + pw;
			bottom_diff[index] = top_diff[pidx];
		}
	}
	template <typename Dtype>
	void DtowLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		const Dtype* const top_diff = top[0]->gpu_diff();
		Dtype* const bottom_diff = bottom[0]->mutable_gpu_diff();
		int count = bottom[0]->count();
		if (d2w){
			dtow_backward_kernel<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
				(count, top_diff, num_, ch_in, h_in, w_in, ch_out, h_out, w_out, psize, bottom_diff);
		}
		else{
			wtod_backward_kernel<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
				(count, top_diff, num_, ch_in, h_in, w_in, ch_out, h_out, w_out, psize, bottom_diff);
		}
		
		//LOG(INFO) << "1";
		CUDA_POST_KERNEL_CHECK;
	}

	INSTANTIATE_LAYER_GPU_FUNCS(DtowLayer);

}  // namespace caffe