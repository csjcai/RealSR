### Kernel Prediction Layer

#### Usage
```
layer {
  name: "PixelConv"
  type: "PixelConv"
  bottom: "Kernel"   <-- the learned kernel/weight
  bottom: "data"     <-- the input data
  top: "Pixel"
}
