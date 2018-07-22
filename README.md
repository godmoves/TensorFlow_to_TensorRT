# TensorFlow to TensorRT
Demos to show how to convert a TensorFlow model into TensorRT

## Dependencies
- python 3.5  
- matplotlib  
- numpy  
- pycuda  
- tensorflow 1.5+  
- [tensorrt 3.0.4](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#overview)  
- [uff](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/index.html#installing-the-uff-toolkit)  
- cmake 3.1+  
- gcc  

Use `pip` to install what you need. For `tensorrt`, `pycuda` and `uff`, you can
find more info [here](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#overview).   
You need to install `tensorrt` by tar package to get python support, but now only python 2.7 
ans python 3.5 are supported. Find more info about how to [download and install](https://developer.nvidia.com/tensorrt). 

## MNIST demo

### Usage
```
cd ./MNIST/uff2plan
mkdir build && cd build
cmake ..
make
cd ../..
python tf2trt.py
```

### Result
You will get the `uff` and `PLAN` file of the TensorFlow model. You can load them using
python or C++ to speed up you inference. Find how to load the `PLAN` file in python
in the demo code, if you want more info about how to load it in C++, you can find it 
[here](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#c_topics).

## ResNet demo

Find more info in `ResNet` folder. You can get the weight files from [here](https://github.com/godmoves/TensorFlow_to_TensorRT/releases/tag/resnet).

##  Reference
- [TensorRT Integration Speeds Up TensorFlow Inference](https://devblogs.nvidia.com/tensorrt-integration-speeds-tensorflow-inference/)  
- [Generating TensorRT Engines from TensorFlow](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/workflows/tf_to_tensorrt.html)  
- [TensorFlow->TensorRT Image Classification](https://github.com/NVIDIA-Jetson/tf_to_trt_image_classification)  

The demo code is almost the same as what in `Generating TensorRT Engines from TensorFlow`, but
I use the C++ API to convert the `uff` model into `PLAN` due to the [Python API doesn't work well](https://devtalk.nvidia.com/default/topic/1029737/jetson-tx1/tensorrt-saving-models-as-portable-plan-files/).
