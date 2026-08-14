# MLIR 与编译器基础设施学习笔记

> 本项目收录关于现代 C++ 系统机制、MLIR 核心架构以及专用硬件加速方言（TensorView）的技术笔记与工程剖析。

---

## 知识体系分层与阅读路径

文档按层次组织，排在前面的内容作为后续文档的前置基础：

```
                    MLIR 知识体系分层
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. 专用方言与硬件加速 (Dialects & Acceleration)                              │
│    └── tensor-view-core-guide.md: TensorView 设计动机、ODS 建模、提升与降级 │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. 框架核心理论与实现 (MLIR Core Framework)                                  │
│    └── mlir-toy-study-notes.md: Toy 语言全景、AST 到 IR、Dialect 转换与优化 │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. 编译器 C++ 基础机制 (C++ Mechanics in Compiler)                           │
│    └── cpp-mechanisms/cpp-mechanisms.md ──► 00~04 C++ 惯用法、对象模型等    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 模块导航

### 1. 编译器 C++ 基础机制

*概览与模块关系*：**[阅读 MLIR Pass 所需的 C++ 机制](cpp-mechanisms/cpp-mechanisms.md)**

| 序号 | 文档名称 | 核心主题 |
| :---: | :--- | :--- |
| **00** | [编译器工程中的现代 C++ 惯用法](cpp-mechanisms/cpp-idioms.md) | `using Base::Base`、`alignas(8)`、EBCO 空基类优化、`ArrayRef`/`function_ref` 零拷贝视图、`llvm::enumerate` |
| **01** | [C++ 对象模型、LLVM Cast 体系与 MLIR 句柄架构](cpp-mechanisms/object-model-and-casting.md) | 内存排布、Itanium ABI 虚表与 Thunk、LLVM `SubclassID` 标签判定、MLIR `OpView` 与 `Operation` 解耦 |
| **02** | [MLIR TypeID 机制与 LLVM 紧凑指针](cpp-mechanisms/typeid-and-tagged-pointers.md) | `TypeID` 静态内存地址、Resolver 作用域、`PointerIntPair` / `PointerUnion` 指针低位复用 |
| **03** | [CRTP 模式与 PassWrapper 基础设施](cpp-mechanisms/crtp-and-pass-wrapper.md) | 不完整类型延迟实例化、`PassWrapper` 类型注入、`clonePass()` 静态转换、TableGen 生成基类 |
| **04** | [MLIR Pass 生命周期与执行状态管理](cpp-mechanisms/pass-lifecycle-and-state.md) | `std::unique_ptr<Pass>` 异构管理、多线程两阶段克隆、`PassExecutionState` 就地构造与作用域 |

---

### 2. MLIR 核心框架与实现

| 文档名称 | 核心主题 |
| :--- | :--- |
| **[MLIR 核心架构与 Toy 教程学习笔记](mlir-toy-study-notes.md)** | AST 到 MLIR 生成、ODS/TableGen 算子定义、TypeStorage 唯一化、Shape 推导接口、Dialect 转换流水线与 JIT 运行时执行。 |

---

### 3. 专用方言与硬件加速实战

| 文档名称 | 核心主题 |
| :--- | :--- |
| **[Triton TensorView：硬件动机、结构化 IR 与编译提升机制](tensor-view-core-guide.md)** | SIMT 掩码寻址 vs 2D DMA 硬件动机、`make_tensor_view` 寻址几何与 `load_view` 读写解耦、ODS Builder 构造链、`TritonRaiseTensorView` 模式提升 Pass、DMA 描述符生成。 |

---

## 阅读约定

1. **自包含性（Self-Containment）**：主文档内部专注于本篇主题，不设置跨文档相互跳转；
2. **知识继承**：前序文档为后续文档的基础支撑（阅读 MLIR 核心理论前建议了解 C++ 基础机制；阅读 TensorView 前建议了解 MLIR 转换理论）。

---

## 开源协议 (License)

本项目采用 [Apache-2.0 License](LICENSE) 开源。
