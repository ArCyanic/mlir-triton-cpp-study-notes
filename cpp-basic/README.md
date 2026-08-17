# 现代 C++ 语言基础与核心机制 (CPP-Basic)

> 本目录收拢现代 C++（C++11/14/17/20/23）中**与特定编译器框架解耦的通用核心语言机制**，作为学习 `cpp-mlir` 体系的前置知识底座。内容覆盖类型系统、值类别、智能指针与控制块、闭包与类型擦除、SFINAE 与 Concepts 约束、编译工具链与 ELF 链接模型。

---

## 模块地图

| 序号 | 模块 | 核心问题 | 关键机制与落点 |
| :---: | :--- | :--- | :--- |
| **00** | [C++ 核心机制速查手册](core-mechanics-quick-reference.md) | 如何在 15 分钟内快速通览核心物理模型、关键边界与避坑准则？ | 6 大模块知识点高密度闪卡式索引、物理内存模型、核心不变式与深度文档精准穿透 |
| **01** | [值类别、传参模型与移动语义](value-categories-and-parameter-passing.md) | 如何兼顾左值拷贝与右值移动？重载决议、RVO 与完美转发如何工作？ | 五大值类别（lvalue/xvalue/prvalue）、Sink Parameter 传值+`std::move`、引用折叠、`std::forward`、NRVO |
| **02** | [STL 容器、迭代器与异常安全](iterators-and-stl-containers.md) | `std::vector` 扩容与哈希表 rehash 时迭代器如何失效？`noexcept` 如何影响扩容搬迁？ | 连续内存搬迁、`std::move_if_noexcept` 强异常安全保证、`unordered_map` 桶结构、迭代器失效矩阵 |
| **03** | [智能指针、控制块与 RAII](smart-pointers-and-raii.md) | `make_shared` 与 `new` 的内存排布差异？如何利用 `enable_shared_from_this` 破除图节点环引用？ | 控制块解构（Strong/Weak count）、EBCO 自定义删除器、`shared_from_this` 私有锚点、Use-Def 计算图实战 |
| **04** | [闭包、递归 Lambda 与类型擦除](lambdas-closures-and-type-erasure.md) | Lambda 编译器生成类原理？递归 Lambda 如何零开销实现？`std::function` 如何擦除类型？ | 闭包仿函数生成、广义移动捕获、Y-Combinator / Deducing this、`std::function` 虚表与 SBO 小对象优化 |
| **05** | [模板元编程、SFINAE 与 Concepts](templates-sfinae-and-concepts.md) | 模板两阶段查找与 ODR 规则？SFINAE 如何探测成员？C++20 Concepts 如何优化编译诊断？ | 两阶段查找、`std::enable_if_t`、`std::void_t` / `declval` 探测、`if constexpr` 分支修剪、Concepts 偏序决议 |
| **06** | [编译流水线、链接模型与 ELF](compiler-toolchain-and-elf-linking.md) | 源码到二进制经历哪四阶段？ELF 节区如何分布？Name Mangling 与静态库链接顺序为何敏感？ | 编译四阶段、链接属性（External/Internal）、ELF Section/Segment、`extern "C"`、`.init_array` 构造链、单遍扫描算法 |

---

## 知识分层与关联导航

```text
               【前置基石层: cpp-basic】
   值类别传参 ──► 智能指针与RAII ──► 闭包类型擦除 ──► SFINAE/Concepts ──► ELF 链接模型
                                      │
                                      ▼ 作为前置知识输入
               【进阶架构层: cpp-mlir】
   LLVM 自定义 RTTI ──► TypeID 单例与紧凑指针 ──► CRTP PassWrapper ──► Pass 多线程状态隔离
```

- 若需深入学习 **LLVM / MLIR 体系定制的 C++ 架构机制**，请跳转至：
  👉 **[cpp-mlir 目录](../cpp-mlir/README.md)**
