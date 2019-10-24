### Laplacian pyramid decompositon

#### modify the caffe.proto
```
optional LapPyrParameter laplacianpyramid_param = 182;
```

```
message LapPyrParameter {
  optional bool is_down = 1 [default = true];
}
```

#### Parameter
is_down: whether to decomposite

#### Usage
```
layer {
  name: "LapPyr_Decomposition"
  type: "LapPyr"
  bottom: "data"  <-- input data
  top: "data3"    <-- 3 scale laplacian pyramid (size: data3=1/2*data2=1/4*data3)
  top: "data2"
  top: "data1"
}
```
```
layer {
  name: "LapPyr_Reconstruction"
  type: "LapPyr"
  bottom: "PixelC"
  bottom: "PixelB"
  bottom: "PixelA"
  top: "output"
  laplacianpyramid_param {
      is_down: false
  }
}
```
