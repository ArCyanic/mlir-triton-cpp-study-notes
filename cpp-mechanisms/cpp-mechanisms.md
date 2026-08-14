# 阅读 MLIR Pass 所需的 C++ 机制

> **核心问题**：当具体 Pass 被擦除为统一的基类句柄 `Pass *` 后，C++ 如何在无标准 RTTI（`-fno-rtti`）的约束下，完成类型识别、静态派生注入、对象克隆与多线程状态隔离？

这组文档围绕 Triton 与 MLIR 官方源码中的内部 verifier pass 展开：

```cpp
struct VerifyWarpSpecializationPartitions
    : PassWrapper<VerifyWarpSpecializationPartitions,
                  OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      VerifyWarpSpecializationPartitions)

  void runOnOperation() override;
};
```

---

## 模块地图

| 序号 | 模块 | 核心问题 | 关键机制与落点 |
| :---: | :--- | :--- | :--- |
| **00** | [编译器工程中的现代 C++ 惯用法](cpp-idioms.md) | 怎样解析 LLVM/MLIR 源码中反复出现的构造继承、EBCO、变长模板包与零拷贝视图？ | `using Base::Base`、`alignas(8)`、`= delete`、EBCO 空基类优化、`ArrayRef`、`function_ref`、`llvm::enumerate` |
| **01** | [C++ 对象模型、LLVM Cast 体系与 MLIR 句柄架构](object-model-and-casting.md) | 为什么编译器放弃 C++ 虚表多态？LLVM 与 MLIR 如何在零开销与动态扩展下完成类型分派？ | 内存排布、Itanium ABI 虚表、Thunk、`SubclassID`、`classof`、`CastInfo`、`TypeID`、`OpView` |
| **02** | [MLIR TypeID 机制与 LLVM 紧凑指针](typeid-and-tagged-pointers.md) | MLIR 怎样产生稳定类型身份，怎样比较它，为什么锚点要 8 字节对齐并能支持低位复用？ | `TypeID::get<T>()`、Resolver 作用域、显式 TypeID 宏、`PointerLikeTypeTraits`、`PointerIntPair` |
| **03** | [CRTP 模式与 PassWrapper 基础设施](crtp-and-pass-wrapper.md) | 普通基类不知道最终派生类型时，怎样把该类型注入通用样板代码？ | `PassWrapper<PassT, BaseT>` 中依赖 `PassT` 的表达式、向下 `static_cast`、延迟实例化 |
| **04** | [MLIR Pass 生命周期与执行状态管理](pass-lifecycle-and-state.md) | PassManager 怎样拥有和复制异构 Pass？长期配置与单次执行状态怎样分离？ | `unique_ptr<Pass>`、`Pass::clone()`、`PassExecutionState`、`threadingSibling` |

---

## 文档边界

各模块围绕统一的编译器 IR 节点和 Pass 基础设施展开，每段机制只在一个模块中深入剖析：

| 交叉点 | 负责展开的模块 | 其他模块只保留什么 |
| :--- | :--- | :--- |
| 构造继承、EBCO、变长模板包、`ArrayRef`/`function_ref`、结构化绑定 | 现代 C++ 惯用法 | 直接使用语法与惯用法结论 |
| 内存排布、虚表与 Thunk、LLVM 静态标签（`SubclassID`/`classof`/`CastInfo`）、MLIR 句柄（`OpView`/`TypeID`） | C++ 对象模型、LLVM Cast 体系与 MLIR 句柄架构 | 其他模块直接使用结论 |
| `TypeID::get<T>()`、Resolver 作用域、锚点与低位存储（`PointerIntPair`） | TypeID 机制与紧凑指针 | 说明类型身份的输入和使用位置 |
| `Derived : Base<Derived>`、向下 `static_cast`、延迟实例化 | CRTP 模式与 PassWrapper | 把 `PassT` 当作已知的具体类型 |
| `clone()` 两阶段复制、多线程 Sibling、`PassExecutionState` 单次执行安装 | Pass 生命周期与状态管理 | 只使用已有对象模型和 TypeID 结论 |

---

## 建议阅读顺序

第一次阅读编译器源码时，建议依循以下顺序推进：

1. **[00. 现代 C++ 惯用法](cpp-idioms.md)**：掌握 EBCO、变长模板包、`ArrayRef` / `function_ref` 等零拷贝视图工具；
2. **[01. C++ 对象模型与 LLVM/MLIR 类型体系](object-model-and-casting.md)**：建立内存排布、ABI 虚表、LLVM 标签多态与 MLIR Handle-Body 架构心智；
3. **[02. TypeID 机制与紧凑指针](typeid-and-tagged-pointers.md)**：掌握类型身份的跨模块作用域与 `PointerIntPair` 指针位复用技术；
4. **[03. CRTP 模式与 PassWrapper](crtp-and-pass-wrapper.md)**：理解如何利用编译期派生注入消除 Pass 通用样板代码；
5. **[04. Pass 生命周期与状态管理](pass-lifecycle-and-state.md)**：系统掌握异构 Pass 队列、多线程并行克隆与单次执行状态安装。

---

## 机制协作模型

```text
具体 pass 类型
  ├─ C++ 类/继承 ──> 基类子对象与统一接口 ───────────────┐
  ├─ CRTP ────────> PassWrapper<PassT, BaseT> ──────────┤
  └─ 身份协议 ────> TypeID / classof / CastInfo ────────┤
                                                       v
                                                `Pass *` 统一句柄
                                                       |
                                                       v
                                      OpPassManager / unique_ptr<Pass>
                                                       |
                                                   虚函数调用
                                                       v
                                                runOnOperation()
```

- **对象模型** 解释句柄指向的对象内存布局；
- **Casting 体系** 解释通用句柄到具体强类型视图的转换；
- **CRTP** 解释通用模板样板如何自动绑定派生类型；
- **TypeID** 在没有 RTTI 的环境下保存稳定可比的身份；
- **PassManager / ExecutionState** 管理异构生命周期、多线程并行副本与上下文绑定。
