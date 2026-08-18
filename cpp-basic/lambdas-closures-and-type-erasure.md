# 闭包、递归 Lambda 与类型擦除

> 本文深入剖析 C++ Lambda 表达式的**编译器闭包类（Closure Class）生成原理**、引用捕获生命周期悬空陷阱、**递归 Lambda 的四种设计方案（从 `std::function` 到 C++23 Deducing this 与 Y-Combinator）**，以及 `std::function` 的**类型擦除（Type Erasure）**与小对象优化（SBO）底层实现。

## 目录

- [1. 闭包类生成机制](#1-闭包类生成机制)
  - [1.1 仿函数下沉（Functor Lowering）](#11-仿函数下沉functor-lowering)
  - [1.2 捕获模式与内存排布](#12-捕获模式与内存排布)
  - [1.3 `mutable` 语义与 const 修饰](#13-mutable-语义与-const-修饰)
- [2. 捕获机制与生命周期](#2-捕获机制与生命周期)
  - [2.1 引用捕获与生命周期悬空](#21-引用捕获与生命周期悬空)
  - [2.2 广义捕获与移动捕获](#22-广义捕获与移动捕获)
- [3. 递归 Lambda 实现方案](#3-递归-lambda-实现方案)
  - [3.1 类型推导死锁问题](#31-类型推导死锁问题)
  - [3.2 `std::function` 包装方案](#32-stdfunction-包装方案)
  - [3.3 泛型自传递方案（C++14）](#33-泛型自传递方案c14)
  - [3.4 Y-Combinator 包装方案（C++17）](#34-y-combinator-包装方案c17)
  - [3.5 显式对象形参方案（C++23 Deducing this）](#35-显式对象形参方案c23-deducing-this)
- [4. 类型擦除与 SBO 机制](#4-类型擦除与-sbo-机制)
  - [4.1 类型擦除与分派架构](#41-类型擦除与分派架构)
  - [4.2 小对象优化（SBO）](#42-小对象优化sbo)
- [5. 可调用对象选型对比](#5-可调用对象选型对比)

## 1. 闭包类生成机制

### 1.1 仿函数下沉（Functor Lowering）

在 C++ 语言标准中，Lambda 表达式并不是特殊的“函数”，它实质上是**编译器在编译期自动合成的一个匿名类（Closure Class / 闭包类型）的语法糖**。

```cpp
// 源码中的 Lambda:
int bias = 10;
auto add_bias = [bias](int x) {
    return x + bias;
};
```

#### 编译器前端（如 Clang AST）生成的等价结构：
```cpp
// 编译器自动生成的匿名类 (Closure Type)
class __lambda_unique_id {
public:
    // 捕获的变量成为类的成员变量
    explicit __lambda_unique_id(int bias) : bias_(bias) {}

    // 函数调用操作符默认修饰为 const
    int operator()(int x) const {
        return x + bias_;
    }

private:
    int bias_;
};

// 调用点的实例化
auto add_bias = __lambda_unique_id{bias};
```

> [!IMPORTANT]
> **每一个 Lambda 表达式都拥有唯一且独立的具体类型**。即使两个 Lambda 的参数和返回值完全一致、代码一模一样，它们的类型在类型系统中也是完全不兼容的独立类型。

### 1.2 捕获模式与内存排布

| 捕获方式 | 语法示例 | 闭包类内部成员类型 | 物理内存影响 |
| :--- | :--- | :--- | :--- |
| **按值捕获** | `[bias]` | `int bias_;` (独立副本) | 闭包对象占用 `sizeof(int)` 字节 |
| **按引用捕获** | `[&bias]` | `int& bias_ref_;` (内部指针) | 闭包对象占用 8 字节（指针大小） |
| **无捕获** | `[]` | 无任何成员变量 | 大小为 1 字节（C++ 空类大小），**可隐式转换为普通函数指针** |

```cpp
// 无捕获 Lambda 可以退化为裸函数指针
int (*func_ptr)(int) = [](int x) { return x * 2; };
```

### 1.3 `mutable` 语义与 const 修饰

默认情况下，编译器生成的闭包类的 `operator()` 带有 `const` 修饰符：
```cpp
int count = 0;
// auto inc = [count]() { count++; }; // ❌ 编译报错：不能在 const 成员函数中修改成员变量 count_
```

加上 `mutable` 关键字后：
```cpp
auto inc = [count]() mutable { count++; }; // ✔️ 编译通过
```
* **本质**：`mutable` 仅仅指示编译器**在生成 `operator()` 时不要附加 `const` 修饰符**，允许修改闭包内部由值捕获产生的本地成员副本。

## 2. 捕获机制与生命周期

### 2.1 引用捕获与生命周期悬空

```cpp
// ❌ 极度危险的野引用代码：
std::function<void()> createCallback() {
    std::string local_str = "Compilation Task";
    return [&local_str]() { // 💥 捕获了栈上局部变量的引用！
        std::cout << local_str << "\n";
    };
}

int main() {
    auto cb = createCallback();
    cb(); // 💥 访问已销毁的栈内存，引发 Undefined Behavior (段错误 / 垃圾数据)
}
```

> [!WARNING]
> **生命周期法则**：若闭包的生命周期可能超过当前函数作用域（如作为回调函数传递、推入任务队列、跨线程异步分发），**严禁按引用捕获局部变量**！必须使用值捕获或移动捕获。

### 2.2 广义捕获与移动捕获

在 C++11 中，无法将独占所有权的对象（如 `std::unique_ptr`）移入 Lambda。C++14 引入了**初始化捕获（Init-Capture）**：

```cpp
auto buffer = std::make_unique<TensorBuffer>(1024);

// C++14 移动捕获：在捕获列表中直接执行所有权转移
auto task = [buf = std::move(buffer)]() {
    buf->process();
};

// 此时原 buffer 已经被掏空，所有权安全转移至闭包类成员 buf 中
```

## 3. 递归 Lambda 实现方案

在计算图遍历（DFS/BFS）或 AST 表达式求值时，经常需要书写递归遍历逻辑。

### 3.1 类型推导死锁问题

```cpp
// ❌ 编译失败写法：
auto dfs = [&](int u) {
    if (u == 0) return;
    dfs(u - 1); // 💥 error: variable 'dfs' declared with deduced type 'auto' cannot appear in its own initializer
};
```
* **类型推导死锁**：在编译器解析 Lambda 函数体中的 `dfs(u - 1)` 时，变量 `dfs` 的类型推导尚未完成（`auto` 还不知道自己是什么类型）。编译器无法在类型尚未成型的对象上进行调用决议。

### 3.2 `std::function` 包装方案

显式声明类型，打破 `auto` 推导依赖：

```cpp
std::function<void(int)> dfs = [&](int u) {
    if (u == 0) return;
    dfs(u - 1); // ✔️ 编译通过：dfs 类型已明确为 std::function<void(int)>
};
```
* **缺点**：引入了 `std::function` 的动态多态（间接函数指针调用）以及潜在的堆内存分配开销，无法被编译器内联展开。

### 3.3 泛型自传递方案（C++14）

将“自身”作为第一个泛型参数传入：

```cpp
auto dfs = [](auto&& self, int u) -> void {
    if (u == 0) return;
    self(self, u - 1); // ✔️ 零额外开销！编译器可完全内联展开！
};

// 启动递归
dfs(dfs, 10);
```
* **优点**：**完全零开销（Zero Overhead）**，纯模板参数推导，编译器可以像普通函数一样对其进行深层内联优化。

### 3.4 Y-Combinator 包装方案（C++17）

如果觉得每次调用都要写 `dfs(dfs, x)` 过于繁琐，可以通过一个通用的 Y-Combinator 包装器消除自传递冗余：

```cpp
template <typename F>
struct YCombinator {
    F func;

    template <typename... Args>
    decltype(auto) operator()(Args&&... args) const {
        return func(*this, std::forward<Args>(args)...);
    }
};

// CTAD (C++17 类模板参数推导辅助)
template <typename F>
YCombinator(F) -> YCombinator<F>;

// 使用示例：
auto dfs = YCombinator([&](auto&& self, int u) -> void {
    if (u == 0) return;
    self(u - 1); // ✔️ 自然调用，无需写 self(self, ...)
});

dfs(10); // 干净优雅
```

### 3.5 显式对象形参方案（C++23 Deducing this）

在 C++23 中，标准引入了显式对象形参（Deducing this），为递归 Lambda 提供了终极原生支持：

```cpp
// C++23 原生语法：
auto dfs = [](this auto&& self, int u) -> void {
    if (u == 0) return;
    self(u - 1); // ✔️ 原生自引用，0 包装器，0 开销！
};

dfs(10);
```

## 4. 类型擦除与 SBO 机制

### 4.1 类型擦除与分派架构

`std::function<R(Args...)>` 能够容纳任何具有匹配签名的可调用实体（普通函数、函数指针、成员函数指针、各类唯一的 Lambda 闭包）。

其底层通过**桥接模式（Bridge Pattern）与虚函数/函数指针表**完成类型擦除：

```text
std::function<void(int)> 外部句柄 (32 字节)
┌───────────────────────────────────────────────────────────┐
│ void (*invoker_)(void* storage, int)   (8 字节调用分派指针) │
│ void (*manager_)(void* dest, void* src)(8 字节生命周期管理器)│
│ char storage_[16]                      (16 字节内联存储缓冲区)│
└───────────────────────────────────────────────────────────┘
```

#### 概念实现简化模型：
```cpp
template <typename FunctionType>
class SimpleFunction;

template <typename R, typename... Args>
class SimpleFunction<R(Args...)> {
public:
    template <typename Callable>
    SimpleFunction(Callable c) {
        // 1. 编译期捕获具体 Callable 类型并生成专用的静态分派函数
        invoker_ = [](void* obj, Args... args) -> R {
            return (*static_cast<Callable*>(obj))(std::forward<Args>(args)...);
        };
        // 2. 存储对象
        storage_ = new Callable(std::move(c));
    }

    R operator()(Args... args) const {
        // 3. 通过函数指针进行间接调用 (类型被成功擦除！)
        return invoker_(storage_, std::forward<Args>(args)...);
    }

private:
    void* storage_;
    R (*invoker_)(void*, Args...);
};
```

### 4.2 小对象优化（SBO）

为了避免频繁在堆上执行 `new` 分配闭包内存，现代标准库（如 libstdc++ / libc++）在 `std::function` 内部预留了一块 **16 ~ 24 字节的栈上内联缓冲区（`storage_`）**：

* **当闭包大小 $\le$ 缓冲区大小**（且为 `noexcept` 移动时）：闭包对象直接构造在 `std::function` 内部，**0 堆内存分配**！
* **当闭包捕获了大量数据（超出缓冲区）**：退化为在堆上 `malloc` 开辟内存。

## 5. 可调用对象选型对比

在编译器系统与高性能开发中，选择合适的可调用对象接口至关重要：

| 方案 | 内存所有权 | 堆内存分配 | 间接调用开销 | 典型适用场景 |
| :--- | :--- | :---: | :---: | :--- |
| **具体泛型 Lambda (`template<class F>`)** | 拥有闭包 | **0** | **0（完全内联）** | 局部 STL 算法（`std::sort`）、高频紧密循环 |
| **`std::function<Sig>`** | **独占拥有闭包** | 小对象 0 (SBO) / 大对象触发 heap | 存在函数指针间接调用 | 长期持有的跨模块回调、异步任务队列 |
| **`llvm::function_ref<Sig>`** | **非拥有（纯借用只读视图）** | **严格为 0** | 单次函数指针间接调用 | **编译器 Pass 遍历（如 `Operation::walk`）** |

> [!TIP]
> **与 `cpp-mlir` 的接口契约**：  
> 在 MLIR / LLVM 内部源码中，对于短期调用的回调（如遍历 IR 节点的 `func.walk([&](Operation *op) { ... })`），**一律使用 `llvm::function_ref` 而非 `std::function`**。`function_ref` 仅占用 16 字节（包含一个 `void*` 对象指针与一个 `R(*)(void*, Args...)` 函数指针），既拥有统一接口，又彻底消除了堆内存分配！
