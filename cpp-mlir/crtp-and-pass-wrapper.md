# CRTP 模式与 PassWrapper 基础设施

> 本文系统剖析现代编译器中用于消除重复样板代码的核心设计模式——**CRTP（Curiously Recurring Template Pattern，奇异递归模板模式）**。以 Triton 内部 Verifier Pass 以及 TableGen 自动生成的 Pass 基类为用例，深入 C++ 不完整类型（Incomplete Type）编译期延迟实例化时序；系统拆解 `PassWrapper<PassT, BaseT>` 如何利用静态向下转型（`static_cast`）实现 `TypeID` 注入、动态深拷贝克隆（`clonePass`）与调度根约束。

## 目录

- [1. CRTP 模式与静态分派](#1-crtp-模式与静态分派)
  - [1.1 派生类类型捕获](#11-派生类类型捕获)
  - [1.2 虚函数分派 vs CRTP 静态分派](#12-虚函数分派-vs-crtp-静态分派)
- [2. 不完整类型与延迟实例化](#2-不完整类型与延迟实例化)
  - [2.1 编译期四阶段时序](#21-编译期四阶段时序)
  - [2.2 CRTP 内存布局安全边界](#22-crtp-内存布局安全边界)
- [3. PassWrapper 基础设施实现](#3-passwrapper-基础设施实现)
  - [3.1 PassT 四重信息流向](#31-passt-四重信息流向)
  - [3.2 clonePass 与 static_cast](#32-clonepass-与-static_cast)
  - [3.3 BaseT 调度根约束](#33-baset-调度根约束)
- [4. TableGen 生成基类与 CRTP](#4-tablegen-生成基类与-crtp)
  - [4.1 PassWrapper vs TableGen 生成基类](#41-passwrapper-vs-tablegen-生成基类)
  - [4.2 声明式选项与派生实现](#42-声明式选项与派生实现)

## 1. CRTP 模式与静态分派

### 1.1 派生类类型捕获

在 MLIR 与 Triton 框架中，我们经常看到形如 `class Derived : public Base<Derived>` 的继承定义：

```cpp
// 1. MLIR 算子句柄中的 CRTP
class LoadOp : public Op<LoadOp, OpTrait::MemRead> {};

// 2. MLIR Pass 基础设施中的 CRTP
struct VerifyWarpSpecializationPartitions
    : PassWrapper<VerifyWarpSpecializationPartitions, OperationPass<ModuleOp>> {
  void runOnOperation() override;
};
```

#### 通用基类缺少派生类的具体身份

在面向对象设计中，通用基类（如 `PassWrapper` 或 `Op`）需要提供通用算法实现（如对象克隆、名称查询、RTTI 替代判别等），但这些算法的具体执行**强依赖于最终派生类的具体 C++ 类型**：
- `clonePass()` 必须调用具体派生类的拷贝构造函数 `new PassT(*this)`；
- `getTypeID()` 必须获取派生类的唯一身份 `TypeID::get<PassT>()`；
- `getName()` 必须反射出派生类的类型名称 `llvm::getTypeName<PassT>()`。

传统面向对象范式只能通过声明大量的 `virtual` 虚函数来强迫派生类编写样板代码；而 CRTP 模式通过**将派生类自身作为模板实参传给基类**，让基类在编译期直接捕获派生类的具体类型信息，实现通用样板的自动化静态生成。

### 1.2 虚函数分派 vs CRTP 静态分派

```
           虚函数动态多态 vs CRTP 编译期静态分派模型
【1. 经典虚函数 (Dynamic Polymorphism)】
基类指针 p ──► 读取 vptr ──► 查虚函数表 ──► 执行 Thunk ──► 跳转至具体实现 (无法内联，有间接调用开销)

【2. CRTP 静态多态 (Compile-Time Polymorphism)】
Base<Derived> ──► 编译期 static_cast<Derived*>(this) ──► 直接调用 Derived 成员函数 (完全内联)
```

| 维度 | 经典虚函数（Virtual Function） | CRTP 静态多态（CRTP Pattern） |
| :--- | :--- | :--- |
| **类型绑定时机** | 运行时（Runtime 查虚表） | **编译期（Compile-Time 模板实例化）** |
| **内存开销** | 8 字节 `vptr` + 虚表数据段 | **0 字节额外内存开销** |
| **执行开销** | 间接寻址 + 间接跳转（破坏内联） | **直接内联展开（无额外跳转开销）** |
| **适用场景** | 异构对象必须放入同构容器统一管理 | 抽取通用算法样板、Trait 静态混入 |

## 2. 不完整类型与延迟实例化

### 2.1 编译期四阶段时序

初学 CRTP 时，常见的疑问是：
> “当编译器读到 `struct VerifyPartitions : PassWrapper<VerifyPartitions>` 时，`VerifyPartitions` 还没定义完，为什么能当作模板参数传给基类？”

C++ 标准规定：**类在声明其基类列表时，自身处于不完整类型（Incomplete Type）状态**。类模板在处理不完整类型时，遵循严格的**两阶段名称查找与延迟实例化规则**：

```
                    CRTP 编译期四阶段时序拓扑
 阶段 1: 符号声明 (Symbol Declaration)
   编译器解析到 `struct VerifyPartitions` ──► 符号表中注册类型名 (此时为 Incomplete Type)
          │
          ▼
 阶段 2: 基类实例化 (Base Class Instantiation)
   编译器处理基类列表 `PassWrapper<VerifyPartitions>`:
   - 确定基类内存大小与成员变量排布 (此时 PassWrapper 内部仅需知道 VerifyPartitions 是一个类型名)
          │
          ▼
 阶段 3: 派生类定义完成 (Class Definition Complete)
   编译器解析 `VerifyPartitions` 的花括号 `{ ... }` 内部字段与函数:
   - `VerifyPartitions` 变为完整类型 (Complete Type)
          │
          ▼
 阶段 4: 成员函数体延迟实例化 (Delayed Member Function Instantiation)
   当代码实际调用 `clonePass()` 或 `getName()` 时:
   - 编译器实例化基类模板的函数体
   - 此时 `VerifyPartitions` 已经完整定义，`sizeof(PassT)` 与构造函数均完全可用！
```

### 2.2 CRTP 内存布局安全边界

理解延迟实例化时序，有助于明确 CRTP 基类中哪些操作是合法的：

```cpp
template <typename DerivedT>
class SafeBase {
  // ✅ 合法：指针与引用不需要完整类型
  DerivedT *ptr;

  // ✅ 合法：函数声明不需要完整类型
  void process(const DerivedT &obj);

  // ✅ 合法：函数体延迟到阶段 4 实例化，届时 DerivedT 已经完整
  void doClone() {
    DerivedT copy = *static_cast<DerivedT *>(this); // 合法！
  }

  // ❌ 编译错误：在阶段 2 计算基类大小时，DerivedT 尚未完成定义！
  // DerivedT invalidMember; // 错误：field has incomplete type 'DerivedT'
};
```

## 3. PassWrapper 基础设施实现

### 3.1 PassT 四重信息流向

MLIR 的 `PassWrapper<PassT, BaseT>` 模板骨架通过一个 `PassT` 模板参数，自动生成了 Pass 运行所需的 4 大关键机制：

```cpp
namespace mlir {
template <typename PassT, typename BaseT>
class PassWrapper : public BaseT {
public:
  // 1. 静态 classof 谓词：将 TypeID 比较绑定到 PassT
  static bool classof(const Pass *pass) {
    return pass->getTypeID() == TypeID::get<PassT>();
  }

  ~PassWrapper() override = default;

protected:
  // 2. 构造函数：自动将 PassT 的 TypeID 注入基类 Pass
  PassWrapper() : BaseT(TypeID::get<PassT>()) {}

  // 3. 名称反射：自动从 PassT 获取去修饰的 C++ 类型名
  StringRef getName() const override {
    return llvm::getTypeName<PassT>();
  }

  // 4. 动态克隆：通过 static_cast 派生下转实现完整对象深拷贝
  std::unique_ptr<Pass> clonePass() const override {
    return std::make_unique<PassT>(
        *static_cast<const PassT *>(this));
  }
};
}
```

```
                        PassT 在 PassWrapper 中的四重流向
                         template <typename PassT>
                                     │
       ┌────────────────┬───────────┴───────────┬────────────────┐
       ▼                ▼                       ▼                ▼
【1. 身份注入】   【2. 名称反射】         【3. 谓词判定】   【4. 动态深拷贝】
BaseT(TypeID::   getName() { return      classof(pass) {   clonePass() {
  get<PassT>())    getTypeName<PassT>();}  getTypeID() ==    make_unique<PassT>(
                                           get<PassT>();}      *static_cast<PassT*>(this));}
```

### 3.2 clonePass 与 static_cast

在 `clonePass()` 中，核心转换语句是：

```cpp
const auto *concretePass = static_cast<const PassT *>(this);
return std::make_unique<PassT>(*concretePass);
```

#### 为什么这里使用 `static_cast` 安全且高效？

1. **编译期继承约束**：`PassWrapper<PassT, BaseT>` 是 `PassT` 的公有直接基类，编译器在编译期已掌握完整的继承偏移动态；
2. **0 周期指针转换**：在单继承体系下，基类子对象与派生类完整对象的起始地址严格重合（偏移量 $\Delta = 0$）。因此 `static_cast` 在汇编层面不产生任何加减法指令，直接透传 `this` 指针；
3. **消除动态检查**：不需要调用开销较大的 `dynamic_cast`，因为 CRTP 的实例化语义保证了当前运行的 `this` 对象必定是一个合法的 `PassT` 实例。

### 3.3 BaseT 调度根约束

`PassWrapper` 的第二个模板参数 `BaseT` 承担了 **Pass 调度根约束**：

```cpp
// 该 Pass 只能挂载并运行在 ModuleOp 级别上
struct MyModulePass : PassWrapper<MyModulePass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation(); // 类型安全：直接返回 ModuleOp 句柄
  }
};

// 该 Pass 挂载并运行在具体 Function 级别上
struct MyFuncPass : PassWrapper<MyFuncPass, OperationPass<func::FuncOp>> {
  void runOnOperation() override {
    func::FuncOp func = getOperation(); // 直接返回 func::FuncOp 句柄
  }
};
```

- `OperationPass<OpT>` 在其构造函数中将 `OpT::getOperationName()` 传递给 MLIR Pass 调度管理器，使 PassManager 能够在构建流水线时直接静态校验 Pass 是否被挂载到了合法的 IR 节点上。

## 4. TableGen 生成基类与 CRTP

### 4.1 PassWrapper vs TableGen 生成基类

在大型生产级项目（如 Triton）中，公开的 Pass 通常包含繁多的命令行参数（Options）和统计量（Statistics）。MLIR 采用 **TableGen 声明式元编程 + CRTP** 的组合架构：

```
                    TableGen 与 CRTP 结合的代码生成流向
Passes.td (声明 Options/Dialects) ──► mlir-tblgen ──► 生成 TritonGPU...Base<DerivedT> (CRTP 基类)
                                                                 ▲
                                                                 │ 继承并注入自身
                                                      struct AutomaticWarpSpecialization
```

```cpp
// TableGen 生成的 Pass 基类骨架 (完全遵循 CRTP 模式)
template <typename DerivedT>
class TritonGPUAutomaticWarpSpecializationBase : public OperationPass<ModuleOp> {
public:
  using Base = TritonGPUAutomaticWarpSpecializationBase;

  TritonGPUAutomaticWarpSpecializationBase()
      : OperationPass<ModuleOp>(TypeID::get<DerivedT>()) {}

  // 自动注入 CRTP clonePass
  std::unique_ptr<Pass> clonePass() const override {
    return std::make_unique<DerivedT>(*static_cast<const DerivedT *>(this));
  }

  // 自动生成 Options 成员与解析接口
  Pass::Option<int32_t> numStages{*this, "num-stages", llvm::cl::desc("Pipeline stages"), llvm::cl::init(3)};
};
```

### 4.2 声明式选项与派生实现

在开发者编写的具体 Pass 实现中，代码极其精简，只需关注算法本身：

```cpp
// 开发者编写的具体 Pass
struct AutomaticWarpSpecialization
    : public triton::gpu::impl::TritonGPUAutomaticWarpSpecializationBase<
          AutomaticWarpSpecialization> {
  // 继承 TableGen 基类的构造函数
  using Base::Base;

  void runOnOperation() override {
    // 直接读取 TableGen 自动绑定的 options
    int stages = numStages.getValue();
    // 执行算法 ...
  }
};
```

| 维度 | 手写 `PassWrapper<PassT, BaseT>` | TableGen 生成基类 `...Base<DerivedT>` |
| :--- | :--- | :--- |
| **元数据来源** | C++ 模板参数与原生类型反射 | `Passes.td` 声明式定义文件 |
| **Options/Stats 支持** | 需手写 `Pass::Option` 字段 | **自动生成强类型成员与 CLI 绑定** |
| **适用场景** | 内部测试 Pass、临时 Verifier | **生产级公开 Pass、框架核心变换 Pass** |

> [!TIP]
> **总结**：
> - **CRTP** 在编译期连接基类与派生类，让基类拥有反向感知派生类完整类型的能力；
> - 配合 **`static_cast`** 与 **延迟实例化**，编译器在 **零额外内存与零虚调用开销** 的前提下，生成 Pass 基础设施。
