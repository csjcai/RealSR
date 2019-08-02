### Shuffle (sub-pixel)

#### modify the caffe.prototxt
```
  optional DtowParameter dtow_param = 998;
```

```
message DtowParameter {
  enum DtowMethod {
    MDTOW = 1;
    MWTOD = 2;
  }
  optional DtowMethod method = 1 [default = MDTOW];
  optional uint32 psize = 2 [default = 2];
}
```

#### Parameter
1. DtowMethod: Large -> small or small -> Large
2. psize: scale

#### Usage
```
layer {
  name: "Down"
  type: "Dtow"
  bottom: "data"
  top: "Down"
  dtow_param {
    psize: 4
    method: MWTOD
  } 
}
```

```
layer {
  name: "Up"
  type: "Dtow"
  bottom: "sum17"
  top: "Up"
  dtow_param {
    psize: 4
  } 
}
```
