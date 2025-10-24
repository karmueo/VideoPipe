# TensorRT 10.x 迁移说明

本文档记录了将 trt_vehicle 项目从旧版本 TensorRT 迁移到 TensorRT 10+ 所做的 API 更改。

## 编译环境
- CUDA: 12.9
- TensorRT: 10.10.0+
- CMake: 3.10+
- GCC: 11.x

## 主要 API 变更

### 1. 对象生命周期管理 (models/base_model.cpp)

#### 1.1 destroy() 方法替换为 delete

TensorRT 10 移除了 `destroy()` 方法，使用标准的 C++ `delete` 代替：

```cpp
// 旧版本
if (m_context) m_context->destroy();
if (m_engine) m_engine->destroy();
if (m_runtime) m_runtime->destroy();

// 新版本 (TensorRT 10+)
if (m_context) delete m_context;
if (m_engine) delete m_engine;
if (m_runtime) delete m_runtime;
```

#### 1.2 deserializeCudaEngine 参数变更

```cpp
// 旧版本
m_engine = m_runtime->deserializeCudaEngine(trtModelStream.data(), size, nullptr);

// 新版本 (TensorRT 10+) - 移除了第三个参数
m_engine = m_runtime->deserializeCudaEngine(trtModelStream.data(), size);
```

### 2. 推理相关 API (models/base_model.cpp)

#### 2.1 Binding API 替换为 I/O Tensor API

TensorRT 10 移除了 Binding API，需要使用新的 I/O Tensor API：

```cpp
// 旧版本
auto dims0 = m_engine->getBindingDimensions(0);
auto dims1 = m_engine->getBindingDimensions(1);

// 新版本 (TensorRT 10+)
auto dims0 = m_engine->getTensorShape(m_engine->getIOTensorName(0));
auto dims1 = m_engine->getTensorShape(m_engine->getIOTensorName(1));
```

#### 2.2 推理执行方法变更

```cpp
// 旧版本
bool ok = m_context->execute(batch, buffers.data());

// 新版本 (TensorRT 10+)
// 1. 首先存储所有 tensor 名称（在 prepare() 中）
int nbIOTensors = m_engine->getNbIOTensors();
m_tensorNames.clear();
for (int i = 0; i < nbIOTensors; i++) {
    m_tensorNames.push_back(m_engine->getIOTensorName(i));
}

// 2. 设置每个 tensor 的地址
for (size_t i = 0; i < m_tensorNames.size() && i < buffers.size(); i++) {
    m_context->setTensorAddress(m_tensorNames[i].c_str(), buffers[i]);
}

// 3. 执行推理
bool ok = m_context->enqueueV3(m_stream);
```

#### 2.3 获取 Tensor 数量的方法变更

```cpp
// 旧版本
int nbBindings = engine->getNbBindings();

// 新版本 (TensorRT 10+)
int nbIOTensors = engine->getNbIOTensors();
```

### 3. 数据结构变更 (models/base_model.h)

在 `CudaPredictor` 类中添加了新的成员变量以存储 tensor 名称：

```cpp
class CudaPredictor{
    // ... 其他成员
private:
    // 新增：存储 tensor 名称用于 TensorRT 10+ API
    std::vector<std::string> m_tensorNames;
};
```

### 4. CMake 配置 (CMakeLists.txt)

添加 CUDA 12.9 的库路径以支持 TensorRT 10：

```cmake
# 添加以下行
link_directories("/usr/local/cuda/targets/x86_64-linux/lib")
```

## 受影响的文件

1. `models/base_model.h` - 添加 tensor 名称存储
2. `models/base_model.cpp` - 更新所有 TensorRT API 调用
   - `getSizeYolo()` - 使用 getTensorShape
   - `getSize()` - 使用 getTensorShape
   - `getSizeVehicle()` - 使用 getTensorShape
   - `prepare()` - 初始化 tensor 名称列表
   - `infer()` - 使用 setTensorAddress + enqueueV3
3. `CMakeLists.txt` - 添加 CUDA 12.9 库路径

## 不受影响的文件

以下文件通过 `CudaPredictor` 间接使用 TensorRT，无需修改：
- `models/class_model.cpp`
- `models/feature_model.cpp`
- `models/detect_model.cpp`
- `models/vehicle_*.cpp` (所有具体车辆模型实现)

## 向后兼容性注意事项

- 使用 TensorRT 10+ 构建的引擎不能在旧版本 TensorRT 上运行
- 需要重新导出并构建引擎文件
- 批次大小 (batch size) 在 TensorRT 10 中通过运行时设置，不再在构建时固定
- 建议保留旧版本代码的备份

## 迁移步骤

1. 升级 CUDA 到 12.9+
2. 升级 TensorRT 到 10.10.0+
3. 更新代码（本次迁移已完成）
4. 重新构建项目：
   ```bash
   cd build
   cmake ..
   make -j$(nproc)
   ```
5. 使用相应工具重新生成 TensorRT 引擎文件

## 参考资料

- [TensorRT 10.0 Release Notes](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/)
- [TensorRT 10.0 Migration Guide](https://docs.nvidia.com/deeplearning/tensorrt/archives/)
- trt_yolov8 项目的迁移实现

## 测试建议

1. 测试所有车辆相关模型的加载
2. 验证推理结果的正确性
3. 对比新旧版本的性能差异
4. 检查内存使用情况
