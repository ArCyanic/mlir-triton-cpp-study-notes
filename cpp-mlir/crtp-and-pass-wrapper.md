# CRTP 模式与 PassWrapper 基础设施

> 本文系统解构现代编译器中用于消除重复样板代码的核心设计模式——**CRTP（Curiously Recurring Template Pattern，奇异递归模板模式）**。以 MLIR 核心框架与 Triton 编译器内部的 Verifier/Pass 基础设施为实战场景，深入剖析 C++ 不完整类型（Incomplete Type）在类模板中的四阶段延迟实例化时序；系统拆解 `PassWrapper<PassT, BaseT>` 基于静态向下转型（`static_cast`）实现的 `TypeID` 注入、隐藏友元运算符、编译期静态纯虚契约检查；详述 MLIR 多线程并发调度下的 `clonePass()` 线程局部隔离拓扑；最后推导 TableGen 声明式代码生成与全局 CLI 管道反射注册的工程闭环。

---

## 1. CRTP 模式与静态多态分派体系

### 1.1 派生类类型反向捕获机理

在 MLIR 与 Triton 框架中，算子句柄、分析工具与编译器 Pass 广泛采用形如 `class Derived : public Base<Derived>` 的继承定义：

```cpp
// 1. MLIR 算子句柄中的 CRTP 静态混入
class LoadOp : public Op<LoadOp, OpTrait::MemRead> {};

// 2. MLIR Pass 基础设施中的 CRTP 包装器
struct VerifyWarpSpecializationPartitions
    : PassWrapper<VerifyWarpSpecializationPartitions, OperationPass<ModuleOp>> {
    void runOnOperation() override;
};
```

在面向对象架构中，通用基类（如 `PassWrapper` 或 `Op`）需要提供通用算法骨架（如对象克隆、全局唯一类型标识提取、类型名称反射等），但这些操作的具体执行**在语义上强依赖于最终派生类的具体 C++ 类型**：

- **对象深拷贝（`clonePass`）**：必须调用具体派生类的拷贝构造函数 `new PassT(*this)`；
- **类型标识生成（`getTypeID`）**：必须获取派生类的唯一地址 `TypeID::get<PassT>()`；
- **类名字符串反射（`getName`）**：必须获取派生类的去修饰 C++ 名称 `llvm::getTypeName<PassT>()`。

传统面向对象范式只能通过在基类中声明大量 `virtual` 虚函数，强迫开发者在每个具体派生类中手动重写样板代码；而 CRTP 模式通过**将派生类自身作为模板实参传递给基类**，使基类在编译期直接捕获了派生类的完整类型信息，实现了通用样板的编译期全自动静态代码生成。

---

### 1.2 虚函数动态多态与 CRTP 静态多态对比

```text
            虚函数动态多态 vs CRTP 编译期静态多态模型
【1. 经典虚函数 (Dynamic Polymorphism)】
基类指针 p ──► 读取 vptr ──► 查虚函数表 (vtable) ──► 间接跳转执行 (无法内联，存在内存与周期开销)

【2. CRTP 静态多态 (Compile-Time Polymorphism)】
Base<Derived> ──► static_cast<Derived*>(this) ──► 编译期直接调用 Derived 成员 (100% 深度内联展开)
```

| 架构对比维度 | 经典虚函数体系（Virtual Functions） | CRTP 静态多态体系（CRTP Pattern） |
| :--- | :--- | :--- |
| **类型绑定时机** | **运行期（Runtime）**：通过虚函数表指针动态决议 | **编译期（Compile-Time）**：通过模板实例化静态下转 |
| **内存空间开销** | 每个对象额外承担 8 字节 `vptr` + 全局只读虚表段 | **严格为 0 字节**（无任何额外内存开销） |
| **执行性能与内联** | 间接寻址 + 间接跳转（破坏 CPU 分支预测，阻碍内联） | **直接内联展开（Inlining）**，与手写原生函数性能完全一致 |
| **容器同构性需求** | 支持将不同派生类对象放入 `std::vector<Base*>` 统一管理 | 派生类属于完全独立的 C++ 类型，无法直接放入非泛型同构容器 |
| **编译器适用场景** | 跨模块动态插件扩展、顶层 PassManager 统一调度链 | **算子句柄（OpView）、Trait 静态混入、PassWrapper 样板生成** |

---

### 1.3 隐藏友元机制与基类非虚析构防泄漏

#### 隐藏友元运算符注入

在设计编译器轻量句柄（如 `OpView` / `Value`）时，通常需要支持对称的比较运算符（如 `==` 与 `!=`）。利用 CRTP 结合 **隐藏友元（Hidden Friends / Barton-Nackman 技巧）**，可以在基类内部自动为派生类生成对称的比较函数，同时将实参依赖查找（ADL, Argument-Dependent Lookup）严格限制在当前类作用域内，避免全局命名空间污染与不必要的隐式类型转换：

```cpp
template <typename Derived>
class EqualityComparable {
public:
    // 隐藏友元：在基类内部直接内联定义友元函数
    friend bool operator==(const Derived &lhs, const Derived &rhs) {
        return lhs.isEqual(rhs); // 静态调用派生类的具体比对方法
    }

    friend bool operator!=(const Derived &lhs, const Derived &rhs) {
        return !lhs.isEqual(rhs);
    }
};

// 派生类继承即可获得全套对称比较操作符：
class IntAttr : public EqualityComparable<IntAttr> {
    int value_;
public:
    explicit IntAttr(int v) : value_(v) {}
    bool isEqual(const IntAttr &other) const { return value_ == other.value_; }
};
```

#### 基类受保护析构函数法则

在 CRTP 体系中，基类通常**不包含虚析构函数**以避免引入 `vptr` 开销。若外部代码误将派生类对象赋给基类指针并执行 `delete base_ptr`，将导致派生类的析构函数无法执行，引发未定义行为（Undefined Behavior）与内存泄漏。

> [!IMPORTANT]
> **CRTP 防泄漏黄金法则**：CRTP 基类的析构函数必须声明为 **`protected` 且非虚（Non-virtual Protected Destructor）**。这样既能允许派生类在自身析构时正常调用基类析构，又能强行在编译期拦截任何通过基类裸指针执行 `delete` 的非法操作。

---

## 2. 不完整类型约束与延迟实例化时序

### 2.1 编译期四阶段生命周期时序拓扑

C++ 标准严格规定：**当一个类正在声明其基类列表时，该类自身处于不完整类型（Incomplete Type）状态**。例如在解析 `struct VerifyPartitions : PassWrapper<VerifyPartitions, ...>` 时，编译器在处理 `PassWrapper<...>` 时尚未读取 `VerifyPartitions` 的类主体花括号。

CRTP 能够成功编译的核心在于 C++ 类模板的**两阶段名字查找与成员函数延迟实例化机制（Delayed Instantiation）**：

```text
                     CRTP 编译期四阶段时序拓扑
 阶段 1: 符号前向声明 (Symbol Forward Declaration)
   编译器读取到 `struct VerifyPartitions` ──► 在符号表中注册该类型名 (此时为 Incomplete Type)
          │
          ▼
 阶段 2: 基类模板实例化 (Base Class Template Instantiation)
   编译器处理基类列表 `PassWrapper<VerifyPartitions, ...>`:
   - 确定基类的成员变量与物理内存排布 (此时 PassWrapper 内部仅需知道 VerifyPartitions 是一个合法类型名)
   - 绝不计算 `sizeof(VerifyPartitions)`，也不展开基类的成员函数代码体！
          │
          ▼
 阶段 3: 派生类主体解析完成 (Derived Class Definition Complete)
   编译器解析 `VerifyPartitions` 的 `{ ... }` 内部字段与函数:
   - `VerifyPartitions` 正式转化为完整类型 (Complete Type)
          │
          ▼
 阶段 4: 基类成员函数延迟实例化 (Delayed Member Function Instantiation)
   当代码在外部实际调用 `pass.clonePass()` 或 `pass.getName()` 时:
   - 编译器才真正展开并编译基类内部的成员函数代码体
   - 此时 `VerifyPartitions` 已经完整定义，`sizeof(PassT)` 与拷贝构造函数均完全就绪！
```

---

### 2.2 CRTP 内存排布合法性与物理安全边界

理解四阶段延迟实例化时序，能够严密界定 CRTP 基类中哪些操作是合法的：

```cpp
template <typename DerivedT>
class CrtpBase {
    // ✅ 允许：定义指向不完整类型的指针或引用 (阶段 2 仅需 8 字节指针大小)
    DerivedT *cachedDerivedPtr_;

    // ✅ 允许：声明接受或返回不完整类型的函数原型
    void process(const DerivedT &obj);

    // ✅ 允许：在成员函数体内部使用 DerivedT 的方法与成员 (延迟到阶段 4 编译)
    void execute() {
        auto *derived = static_cast<DerivedT *>(this);
        derived->run(); // 合法！在阶段 4 展开时 DerivedT 已拥有 run() 定义
    }

    // ❌ 严禁：在基类内部直接声明不完整类型的普通成员变量！
    // DerivedT invalidMember_; // 编译报错：field has incomplete type 'DerivedT'
};
```

---

### 2.3 静态纯虚契约与编译期强制重写校验

传统虚函数体系使用 `= 0` 声明纯虚接口，强迫派生类必须重写；而在 CRTP 静态多态中，若派生类遗漏了关键方法的实现，默认情况下编译器可能会静默回退到基类的缺省实现或产生晦涩的链接错误。

在现代编译器开发中，可以通过在 CRTP 基类中注入 **`static_assert` 静态断言**，在编译期强制校验派生类是否重写了特定接口：

```cpp
template <typename Derived>
class PassContract {
public:
    void executePass() {
        // 编译期静态校验：确保派生类必须显式提供 runOnOperation 成员函数
        static_assert(&Derived::runOnOperation != &PassContract::runOnOperationFallback,
                      "CRTP Violation: Derived Pass MUST implement 'runOnOperation()'!");
        
        static_cast<Derived *>(this)->runOnOperation();
    }

private:
    void runOnOperationFallback() {}
};
```

若派生类未能重写 `runOnOperation`，编译器在阶段 4 实例化时会直接抛出精准、高可读性的静态断言报错，使 CRTP 静态多态兼具与纯虚函数完全等效的接口约束力。

---

## 3. PassWrapper 基础设施与多线程隔离架构

### 3.1 PassT 四重元数据与流向闭环

MLIR 的 `PassWrapper<PassT, BaseT>` 模板骨架通过引入唯一的 `PassT` 模板形参，在基类中全自动闭环生成了 Pass 运行所需的 4 大核心机制：

```cpp
namespace mlir {
template <typename PassT, typename BaseT>
class PassWrapper : public BaseT {
public:
    // 1. 静态 classof 谓词：用于 LLVM RTTI 体系中的 llvm::dyn_cast<PassT>
    static bool classof(const Pass *pass) {
        return pass->getTypeID() == TypeID::get<PassT>();
    }

    ~PassWrapper() override = default;

protected:
    // 2. 构造函数：自动将 PassT 的全局唯一 TypeID 静态注入底层 BaseT
    PassWrapper() : BaseT(TypeID::get<PassT>()) {}

    // 3. 名称反射：自动从 PassT 类型中提取人类可读的 C++ 类型名
    StringRef getName() const override {
        return llvm::getTypeName<PassT>();
    }

    // 4. 动态克隆：通过 static_cast 派生下转实现对象的精准深拷贝
    std::unique_ptr<Pass> clonePass() const override {
        return std::make_unique<PassT>(*static_cast<const PassT *>(this));
    }
};
}
```

```text
                        PassT 在 PassWrapper 中的四重元数据流向
                         template <typename PassT>
                                     │
       ┌────────────────┬───────────┴───────────┬────────────────┐
       ▼                ▼                       ▼                ▼
【1. 身份注入】   【2. 名称反射】         【3. 谓词判定】   【4. 动态深拷贝】
BaseT(TypeID::   getName() { return      classof(pass) {   clonePass() {
  get<PassT>())    getTypeName<PassT>();}  getTypeID() ==    make_unique<PassT>(
                                           get<PassT>();}      *static_cast<PassT*>(this));}
```

---

### 3.2 static_cast 零开销派生下转与汇编透传

在 `clonePass()` 的实现中，核心代码是 `*static_cast<const PassT *>(this)`。该转换在 C++ 底层具有极高的执行效率与完备的安全性：

- **单继承偏移归零（$\Delta = 0$）**：在典型的单继承物理模型下，`PassWrapper` 基类子对象的起始地址与外部 `PassT` 派生类完整对象的内存起始地址完全重合。因此在 x86-64 汇编指令层面，`static_cast` **不产生任何加减法寻址指令，直接在寄存器中透传 `this` 指针**；
- **类型安全保障**：由于 `PassWrapper` 是通过 CRTP 专门为 `PassT` 实例化的基类，其运行时的 `this` 指针所指向的实体在类型系统层面 100% 确定是 `PassT`，因而**彻底免除了昂贵的 `dynamic_cast` 运行时虚表遍历与 RTTI 检查**。

---

### 3.3 多线程并行 Pass 调度与线程局部隔离

在现代高性能编译器（如 MLIR / LLVM / Triton）中，PassManager 在处理包含数千个函数的 Module 时，会启动多线程工作池（Worker Thread Pool）并发执行变换 Pass。

```text
                  MLIR 多线程并发 Pass 调度与 clonePass 隔离架构
                               PassManager 调度中心
                                        │
             ┌──────────────────────────┼──────────────────────────┐
             ▼                          ▼                          ▼
     [Worker Thread 0]          [Worker Thread 1]          [Worker Thread 2]
             │                          │                          │
             │ 调用 clonePass()         │ 调用 clonePass()         │ 调用 clonePass()
             ▼                          ▼                          ▼
   ┌───────────────────┐      ┌───────────────────┐      ┌───────────────────┐
   │ Pass 实例 A (副本) │      │ Pass 实例 B (副本) │      │ Pass 实例 C (副本) │
   │ 独立成员与分析状态 │      │ 独立成员与分析状态 │      │ 独立成员与分析状态 │
   └─────────┬─────────┘      └─────────┬─────────┘      └─────────┬─────────┘
             │                          │                          │
             ▼                          ▼                          ▼
       func @kernel_0             func @kernel_1             func @kernel_2
```

Pass 类往往包含内部成员变量（如统计计数器、临时分析缓存、DominanceTree 句柄）。如果多个线程共享同一个 Pass 单例实例，并发读写成员变量将直接引发毁灭性的**数据竞争（Data Race）**。

`PassManager` 在将 Pass 分发给各个工作线程前，必须强制调用虚接口 `clonePass()` 为每个线程克隆出一个独立的 Pass 副本。`PassWrapper` 借助 CRTP 自动生成了强类型的深拷贝逻辑，从而在多线程并发架构下提供了坚如磐石的**线程局部状态隔离（Thread-Local Isolation）**。

---

### 3.4 BaseT 调度根约束与类型安全校验

`PassWrapper` 的第二个模板参数 `BaseT` 承担了 **Pass 调度根约束** 的职责：

```cpp
// 该 Pass 只能挂载并调度在 ModuleOp 级别上
struct TritonModulePass : PassWrapper<TritonModulePass, OperationPass<ModuleOp>> {
    void runOnOperation() override {
        ModuleOp module = getOperation(); // 类型安全：直接返回 ModuleOp 句柄
    }
};

// 该 Pass 只能挂载并调度在具体 Function 级别上
struct TritonFunctionPass : PassWrapper<TritonFunctionPass, OperationPass<func::FuncOp>> {
    void runOnOperation() override {
        func::FuncOp func = getOperation(); // 直接返回 func::FuncOp 句柄
    }
};
```

`OperationPass<OpT>` 在其构造函数中将目标算子的名称 `OpT::getOperationName()` 传递给底层调度器。当开发者试图将一个 `OperationPass<func::FuncOp>` 错误地挂载到 Module 流水线上时，PassManager 会在流水线构建阶段直接拦截并报告静态约束错误。

---

## 4. TableGen 声明式生成基类与工程闭环

### 4.1 TableGen 声明式代码生成与 CRTP 协同

在大型生产级编译器工程（如 Triton / MLIR Dialect）中，公开的 Pass 通常包含复杂的命令行参数（CLI Options）与运行时统计指标（Statistics）。MLIR 采用了 **TableGen 声明式定义 + CRTP 骨架生成** 的协同架构：

```text
                     TableGen 声明式 Pass 生成闭环
Passes.td 声明式配置 ──► mlir-tblgen ──► 生成 TritonGPU...Base<DerivedT> (CRTP 基类)
 (定义 CLI Options 等)                           ▲
                                                 │ 继承并注入具体实现类型
                                     struct AutomaticWarpSpecialization
```

```cpp
// TableGen 自动生成的 Pass 基类骨架 (完全遵循 CRTP 设计范式)
template <typename DerivedT>
class TritonGPUAutomaticWarpSpecializationBase : public OperationPass<ModuleOp> {
public:
    using Base = TritonGPUAutomaticWarpSpecializationBase;

    TritonGPUAutomaticWarpSpecializationBase()
        : OperationPass<ModuleOp>(TypeID::get<DerivedT>()) {}

    // 自动生成的 CRTP 深拷贝方法
    std::unique_ptr<Pass> clonePass() const override {
        return std::make_unique<DerivedT>(*static_cast<const DerivedT *>(this));
    }

    // 自动生成的强类型 CLI Option 成员
    Pass::Option<int32_t> numStages{
        *this, "num-stages", 
        llvm::cl::desc("Number of pipeline stages"), 
        llvm::cl::init(3)
    };
};
```

---

### 4.2 全局 Pass 注册中心与 CLI 管道反射解析

在 TableGen 生成的代码末尾，会自动生成全局注册函数：

```cpp
// TableGen 自动生成的全局 Pass 注册函数
inline void registerTritonGPUAutomaticWarpSpecializationPass() {
    ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
        return std::make_unique<AutomaticWarpSpecialization>();
    });
}
```

该机制使编译器驱动工具（如 `mlir-opt` 或 `triton-opt`）能够通过文本格式的流水线管道描述字符串直接反射解析并动态构造 Pass：

```bash
# 命令行通过文本直接反射构造带有参数配置的 Pass Pipeline：
mlir-opt --pass-pipeline='builtin.module(triton-gpu-automatic-warp-specialization{num-stages=4})' input.mlir
```

---

### 4.3 手写 PassWrapper 与 TableGen 生成基类全景对比

| 架构对比维度 | 手写 `PassWrapper<PassT, BaseT>` | TableGen 生成基类 `...Base<DerivedT>` |
| :--- | :--- | :--- |
| **元数据与配置来源** | 纯 C++ 模板参数与原生类型反射 | `Passes.td` 声明式 DSL 描述文件 |
| **CLI 命令行参数绑定** | 需手动在类内声明 `Pass::Option<T>` 字段 | **TableGen 自动生成参数解析、默认值与帮助文档** |
| **流水线注册支持** | 需手写注册函数代码 | **自动生成 `registerPass` 与管道反射解析适配器** |
| **工程维护成本** | 随着 Pass 增多容易遗漏样板代码 | **单处 DSL 修改，全工程自动同步生成** |
| **编译器系统典型适用场景** | 内部单元测试 Pass、私有验证器（Verifier Pass） | **生产级公开 Pass、框架核心优化与降级 Pass** |
