### Kernel Prediction Layer

#### Parameter
1. is_pad: whether to pad the image or not
2. is_bpk: whether to cacluate the gradient or update the weight or not
3. is_bpd:whether to cacluate the gradient or update the data or not

#### Usage
```
layer {
  name: "PixelConv"
  type: "PixelConv"
  bottom: "Kernel"   <-- the learned kernel/weight
  bottom: "data"     <-- the input data
  top: "Pixel"
}
