# TensorRT 10.x 迁移说明

本文档记录了将 trt_yolov8 项目从旧版本 TensorRT 迁移到 TensorRT 10.10.0 所做的 API 更改。

## 编译环境
- CUDA: 12.9
- TensorRT: 10.10.0.31
- CMake: 3.28.3
- GCC: 11.x

## 主要 API 变更

### 1. 模型构建相关 (src/model.cpp)

#### 1.1 ResizeMode 改为 InterpolationMode
```cpp
// 旧版本
upsample->setResizeMode(nvinfer1::ResizeMode::kNEAREST);

// 新版本 (TensorRT 10+)
upsample->setResizeMode(nvinfer1::InterpolationMode::kNEAREST);
```

#### 1.2 卷积层设置方法名称变更
```cpp
// 旧版本
conv->setStride(nvinfer1::DimsHW{1, 1});
conv->setPadding(nvinfer1::DimsHW{0, 0});

// 新版本 (TensorRT 10+)
conv->setStrideNd(nvinfer1::DimsHW{1, 1});
conv->setPaddingNd(nvinfer1::DimsHW{0, 0});
```

#### 1.3 addConvolution 改为 addConvolutionNd
```cpp
// 旧版本
network->addConvolution(*input, channels, nvinfer1::DimsHW{1, 1}, weights, bias);

// 新版本 (TensorRT 10+)
network->addConvolutionNd(*input, channels, nvinfer1::DimsHW{1, 1}, weights, bias);
```

#### 1.4 Builder 配置方法变更
```cpp
// 旧版本
builder->setMaxBatchSize(kBatchSize);
config->setMaxWorkspaceSize(16 * (1 << 20));

// 新版本 (TensorRT 10+)
// setMaxBatchSize 已废弃，批次大小在运行时设置
// builder->setMaxBatchSize(kBatchSize); // Deprecated
config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 16 * (1 << 20));
```

#### 1.5 全连接层替换为卷积层
TensorRT 10 移除了 `addFullyConnected` API，需要使用 1x1 卷积代替：

```cpp
// 旧版本
nvinfer1::IFullyConnectedLayer* yolo = network->addFullyConnected(
    *pool->getOutput(0), kClsNumClass, 
    weightMap["model.9.linear.weight"], 
    weightMap["model.9.linear.bias"]);

// 新版本 (TensorRT 10+)
// 先 Reshape 再用 1x1 卷积
nvinfer1::IShuffleLayer* flatten = network->addShuffle(*pool->getOutput(0));
flatten->setReshapeDimensions(nvinfer1::Dims3{dims.d[0], 1, 1});

nvinfer1::IConvolutionLayer* yolo = network->addConvolutionNd(
    *flatten->getOutput(0), kClsNumClass, nvinfer1::DimsHW{1, 1},
    weightMap["model.9.linear.weight"], 
    weightMap["model.9.linear.bias"]);
```

### 2. 推理相关 (检测器文件)

#### 2.1 Binding API 替换为 I/O Tensor API
```cpp
// 旧版本
assert(engine->getNbBindings() == 2);
const int inputIndex = engine->getBindingIndex(kInputTensorName);
const int outputIndex = engine->getBindingIndex(kOutputTensorName);
auto out_dims = engine->getBindingDimensions(1);

// 新版本 (TensorRT 10+)
assert(engine->getNbIOTensors() == 2);
// 不再需要 getBindingIndex，直接使用 tensor 名称
auto out_dims = engine->getTensorShape(kOutputTensorName);
```

#### 2.2 推理执行方法变更
```cpp
// 旧版本
context.enqueue(batchsize, buffers, stream, nullptr);

// 新版本 (TensorRT 10+)
context.setTensorAddress(kInputTensorName, buffers[0]);
context.setTensorAddress(kOutputTensorName, buffers[1]);
context.enqueueV3(stream);
```

### 3. CMake 配置 (CMakeLists.txt)

添加 CUDA 库路径以支持 CUDA 12.9：
```cmake
# 添加以下行
link_directories("/usr/local/cuda/targets/x86_64-linux/lib")
```

## 受影响的文件

1. `src/model.cpp` - 模型构建 API 更新
2. `trt_yolov8_detector.cpp` - 目标检测推理 API 更新
3. `trt_yolov8_pose_detector.cpp` - 姿态检测推理 API 更新
4. `trt_yolov8_seg_detector.cpp` - 分割检测推理 API 更新
5. `trt_yolov8_classifier.cpp` - 分类推理 API 更新
6. `CMakeLists.txt` - CUDA 库路径配置

## 向后兼容性注意事项

- 使用 TensorRT 10+ 构建的引擎不能在旧版本 TensorRT 上运行
- 需要重新导出并构建引擎文件
- 建议保留旧版本代码的备份

## 参考资料

- [TensorRT 10.0 Release Notes](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/)
- [TensorRT 10.0 Migration Guide](https://docs.nvidia.com/deeplearning/tensorrt/archives/)
