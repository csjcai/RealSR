# [Toward Real-World Single Image Super-Resolution (RealSR)](https://csjcai.github.io/papers/RealSR.pdf)


### Dataset:

Captured device: (Canon 5D3 and Nikon D810) +  (24∼105mm, f/4.0 zoom lens)

A part of this dataset was used in the RealSR challenge in [NTIRE 2019 (in conjunction with CVPR)](http://www.vision.ee.ethz.ch/ntire19/).

#### [Version 1](https://drive.google.com/open?id=1gKnm9BdgyqISCTDAbGbpVitT-QII_unw): 234 scenes, as reported in the original paper (HR have the same resolution as LR).


#### [Version 2](https://drive.google.com/open?id=1dEBRo_1HH6Yk9zrchEg_JTRi-Uhmd-sj): More than 500 scenes, the extended version (HR have the same resolution as LR).


 |Methods|PSNR       |      2      |      3      |      4      |SSIM       |      2      |      3      |      4      |  
 |----------|---------- |:-----------:|:-----------:|:-----------:|---------- |:-----------:|:-----------:|:-----------:|
 |KPN(K=5)|   |    33.41    |    30.47    |    28.80    |           |    0.913    |    0.860    |    0.826    |           
 |KPN(K=7)|   |    33.42    |    30.49    |    28.84    |  |    0.913    |    0.861    |    0.826    |  
 |KPN(K=13)|  |    33.44    |    30.52    |    28.92    |  |    0.913    |    0.863    |    0.829    |  
 |KPN(K=19)|  |    33.45    |    30.57    |    28.99    |  |    0.914    |    0.864    |    0.832    |
 |LP-KPN   |  |    33.49    |    30.60    |    29.05    |    |    0.917    |    0.865    |    0.834    | 
                        

         

 
 
 


#### [Version 3](https://drive.google.com/open?id=17ZMjo-zwFouxnm_aFM6CUHBwgRrLZqIM): More than 500 scenes, the extended version (HR and LR have different resolution).




### Code:
#### Caffe: pretrained-model, training code, and testing code
1. Download the new layers in folder ['Layer'](https://github.com/csjcai/RealSR/tree/master/Layer)
2. Modify the caffe.prototxt
3. Compile Caffe and Matcaffe ([installation](https://caffe.berkeleyvision.org/installation.html))

-- Training --

4. Generate the training data
5. run *solver.prototxt to train the network

-- Testing --

6. run Test.m 



#### Alignment code:
1. Put your own image pairs in the folder and modify the path
2. run [Demo.m](https://github.com/csjcai/RealSR/blob/master/Alignment/Demo.m) in folder ['Alignment'](https://github.com/csjcai/RealSR/tree/master/Alignment)
3. Central region crop



### Citation:
If you find this work useful for your research, please cite:

```
@inproceedings{cai2019toward,
  title={Toward real-world single image super-resolution: A new benchmark and a new model},
  author={Cai, Jianrui and Zeng, Hui and Yong, Hongwei and Cao, Zisheng and Zhang, Lei},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  year={2019}
}
```

```
@inproceedings{cai2019ntire,
  title={Ntire 2019 challenge on real image super-resolution: Methods and results},
  author={Cai, Jianrui and Gu, Shuhang and Timofte, Radu and Zhang, Lei},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  year={2019}
}
```

### Contact:
Please contact me if there is any question (Jianrui CAI: csjcai@comp.polyu.edu.hk).
