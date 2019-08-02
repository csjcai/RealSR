#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/dtow_layer.hpp"

namespace caffe {

template <typename Dtype>
void DtowLayer<Dtype>::Reshape(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	h_in = bottom[0]->height();
	w_in = bottom[0]->width();
	ch_in = bottom[0]->channels();
	num_ = bottom[0]->num();

	DtowParameter rm = this->layer_param_.dtow_param();
	psize = rm.psize();
	d2w = (rm.method() == DtowParameter_DtowMethod_MDTOW);
	if (d2w){
		CHECK_EQ(0, ch_in % (psize*psize)) << "the size of depth must be multiple of the square of the upsampling size";
		ch_out = ch_in / (psize*psize);
		h_out = h_in * psize;
		w_out = w_in * psize;
	}
	else{
		CHECK_EQ(0, w_in % psize) << "the size of width must be multiple of the sampling size";
		CHECK_EQ(0, h_in % psize) << "the size of height must be multiple of the sampling size";
		ch_out = ch_in * psize*psize;
		h_out = h_in / psize;
		w_out = w_in / psize;
	}
	top[0]->Reshape(num_, ch_out, h_out, w_out);
}

template <typename Dtype>
void DtowLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	Dtype * const top_data = top[0]->mutable_cpu_data();
	const Dtype * const bottom_data = bottom[0]->cpu_data();
	int p2size = psize*psize;
	const int stride = ch_in*w_in*h_in;
	int pc, rc, ph, pw, pidx;
	int tc, tw, th, tn;
	if (d2w){
		for (int i = 0; i < num_*stride; i++){
			tw = i%w_in;
			th = (i / w_in) % h_in;
			tc = (i / w_in / h_in) % ch_in;
			tn = i / w_in / h_in / ch_in;
			pc = tc / p2size;
			rc = tc % p2size;
			ph = th*psize + rc / psize;
			pw = tw*psize + rc % psize;
			pidx = tn*stride + (pc*h_out + ph)*w_out + pw;
			top_data[pidx] = bottom_data[i];
		}
	}
	else{
	     for (int i = 0; i < num_*stride; i++){
			tw = i%w_in;
			th = (i / w_in) % h_in;
			tc = (i / w_in / h_in) % ch_in;
			tn = i / w_in / h_in / ch_in;
			ph = th / psize;
			pw = tw / psize;
			pc = tc * p2size+(th%psize)*psize+tw%psize;
			pidx = tn*stride + (pc*h_out + ph)*w_out + pw;
			top_data[pidx] = bottom_data[i];
		}
	}
}

template <typename Dtype>
void DtowLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	const Dtype * const top_diff = top[0]->cpu_diff();
	Dtype * const bottom_diff = bottom[0]->mutable_cpu_diff();
	int p2size = psize*psize;
	const int stride = ch_in*w_in*h_in;
	int pc, rc, ph, pw, pidx;
	int tc, tw, th, tn;
	if (d2w){
		for (int i = 0; i < num_*stride; i++){
			tw = i%w_in;
			th = (i / w_in) % h_in;
			tc = (i / w_in / h_in) % ch_in;
			tn = i / w_in / h_in / ch_in;
			pc = tc / p2size;
			rc = tc % p2size;
			ph = th*psize + rc / psize;
			pw = tw*psize + rc % psize;
			pidx = tn*stride + (pc*h_out + ph)*w_out + pw;
			bottom_diff[i] = top_diff[pidx];
		}
	}
	else{
		for (int i = 0; i < num_*stride; i++){
			tw = i%w_in;
			th = (i / w_in) % h_in;
			tc = (i / w_in / h_in) % ch_in;
			tn = i / w_in / h_in / ch_in;
			ph = th / psize;
			pw = tw / psize;
			pc = tc * p2size + (th%psize)*psize + tw%psize;
			pidx = tn*stride + (pc*h_out + ph)*w_out + pw;
			bottom_diff[i] = top_diff[pidx];
		}
	}
}

#ifdef CPU_ONLY
	STUB_GPU(DtowLayer);
#endif

	INSTANTIATE_CLASS(DtowLayer);
	REGISTER_LAYER_CLASS(Dtow);

}  // namespace caffe