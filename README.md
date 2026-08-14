# MLIR 编译基础设施与方言设计核心知识库

> 本知识库系统收录从现代 C++ 底层系统级机制、MLIR 核心 IR 架构理论到工业级专用加速后端方言（TensorView）端到端实战的高质量技术文档。

---

## 知识体系分层与推荐阅读路线

本知识库采用**“三阶递进、自底向上”**的知识架构。主文档间不设相互引用，默认排在前面的文档为后续文档的前置知识：

```
                    MLIR 核心知识库三阶递进大厦
┌─────────────────────────────────────────────────────────────────────────────┐
│ 第三阶：工业级方言与后端实战 (Industrial Dialects & Acceleration)          │
│ └── tensor-view-core-guide.md: TensorView 硬件动机、ODS 建模、提升与降级   │
├─────────────────────────────────────────────────────────────────────────────┤
│ 第二阶：框架核心理论与经典实现 (MLIR Core Framework & Transformations)       │
│ └── mlir-toy-study-notes.md: Toy 语言全景、AST 到 IR、Dialect 转换与优化     │
├─────────────────────────────────────────────────────────────────────────────┤
│ 第一阶：底层系统基石 (High-Performance C++ Mechanics in Compiler)          │
│ └── cpp-mechanisms/cpp-mechanisms.md ──► 00~04 C++ 惯用法、对象模型等        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 详细模块导航

### 第一阶：底层系统基石（现代 C++ 机制与编译器基础设施）

*导航与关系地图*：**[阅读 MLIR Pass 所需的 C++ 机制总览](cpp-mechanisms/cpp-mechanisms.md)**

| 模块序号 | 文档名称 | 核心机制与认知焦点 |
| :---: | :--- | :--- |
| **00** | [编译器工程中的现代 C++ 高级惯用法](cpp-mechanisms/cpp-idioms.md) | `using Base::Base`、`alignas(8)`、EBCO 空基类优化、`ArrayRef`/`function_ref` 零拷贝视图、`llvm::enumerate` |
| **01** | [C++ 对象模型、LLVM Cast 体系与 MLIR 句柄架构](cpp-mechanisms/object-model-and-casting.md) | 64 位内存排布、Itanium ABI 虚表与 Thunk、LLVM `SubclassID` 标签与 Range Check、MLIR Handle-Body 架构 |
| **02** | [MLIR TypeID 锚点与 LLVM 紧凑指针压缩](cpp-mechanisms/typeid-and-tagged-pointers.md) | `TypeID` 静态内存地址唯一化、四大 Resolver 作用域、`PointerIntPair` / `PointerUnion` 指针低位窃取压缩 |
| **03** | [CRTP 编译期注入与 PassWrapper 基础设施](cpp-mechanisms/crtp-and-pass-wrapper.md) | 不完整类型延迟实例化时序、`PassWrapper` 四重类型注入、`clonePass()` 静态向下强转、TableGen 生成基类 |
| **04** | [MLIR Pass 调度引擎：多线程克隆与执行状态管理](cpp-mechanisms/pass-lifecycle-and-state.md) | `std::unique_ptr<Pass>` 异构类型擦除、多线程两阶段克隆、`PassExecutionState` 就地构造、借用生命周期 |

---

### 第二阶：框架核心理论（MLIR 端到端设计与代码生成）

| 文档名称 | 核心机制与认知焦点 |
| :--- | :--- |
| **[MLIR 核心架构与 Toy 语言全流程学习笔记](mlir-toy-study-notes.md)** | AST 到 MLIR 生成、ODS/TableGen 算子定义、TypeStorage 唯一化、Shape 推导接口、Dialect Conversion 降级流水线与 JIT 运行时执行。 |

---

### 第三阶：工业级方言与专用硬件加速实战（TensorView）

| 文档名称 | 核心机制与认知焦点 |
| :--- | :--- |
| **[Triton TensorView：硬件动机、结构化 IR 与编译提升机制](tensor-view-core-guide.md)** | GPU SIMT Mask vs NPU 2D DMA 硬件动机、`make_tensor_view` 寻址几何与 `load_view` 读写解耦、Four-tier Builder 构造链、`TritonRaiseTensorView` 模式提升 Pass、DMA 描述符生成。 |

---

## 架构原则与阅读约定

1. **严格的自包含性（Self-Containment）**：每个主文档内部专注深挖自身领域，不设置指向其他主文档的悬空或双向跳转，保证阅读时的沉浸感；
2. **前置知识单向继承**：排在前面的文档为后序文档提供天然的基础认知支撑（如学习 MLIR 架构前建议掌握第一阶 C++ 机制；学习 TensorView 前建议掌握第二阶 MLIR 转换理论）。

---

## 开源协议 (License)

本项目采用 [Apache-2.0 License](LICENSE) 开源。欢迎在遵守开源协议的前提下自由阅读、研讨与交流。
