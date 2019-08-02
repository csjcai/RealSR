# Toward Real-World Single Image Super-Resolution (RealSR)


### Dataset:

Captured device: (Canon 5D3 and Nikon D810) +  (24∼105mm, f/4.0 zoom lens)

A part of this dataset was used in the RealSR challenge in [NTIRE 2019 (in conjunction with CVPR)](http://www.vision.ee.ethz.ch/ntire19/).

#### [Version 1](https://drive.google.com/open?id=1gKnm9BdgyqISCTDAbGbpVitT-QII_unw): 234 scenes, as reported in the original paper (HR have the same resolution as LR).


#### [Version 2](https://drive.google.com/open?id=1dEBRo_1HH6Yk9zrchEg_JTRi-Uhmd-sj): More than 500 scenes, the extended version (HR have the same resolution as LR).


#### Version 3: More than 500 scenes, the extended version (HR and LR have different resolution).




### Code:
#### Caffe: pretrained-model, training code, and testing code
1. Download the new layers in folder ['Layer'](https://github.com/csjcai/RealSR/tree/master/Layer)
2. Modify the caffe.prototxt
3. Compile Caffe ([installation](https://caffe.berkeleyvision.org/installation.html))
4. Generate the training data
5. run *solver.prototxt to train the network



#### Alignment code:
1. Put your own image pairs in the folder and modify the path
2. run Demo.m in folder ['Alignment'](https://github.com/csjcai/RealSR/tree/master/Alignment)
3. Central region crop



### Citation
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

### Contact
Please contact me if there is any question (Jianrui CAI: csjcai@comp.polyu.edu.hk).
