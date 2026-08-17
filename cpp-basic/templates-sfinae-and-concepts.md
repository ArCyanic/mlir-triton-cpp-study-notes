# 模板元编程、SFINAE 与 Concepts

> 本文系统性解析 C++ 编译期多态与代码生成机制，包括**模板的两阶段名字查找（Two-phase Lookup）与 ODR 规则**、**SFINAE 替换失败非错原则**、`std::enable_if` 与 `std::void_t` 探测惯用法，以及现代 **C++20 Concepts 约束体系**的语法演进与编译诊断优化。

---

## 目录

- [1. 模板编译模型与代码生成](#1-模板编译模型与代码生成)
  - [1.1 两阶段名字查找（Two-phase Lookup）](#11-两阶段名字查找two-phase-lookup)
  - [1.2 模板单一定义规则（Template ODR）](#12-模板单一定义规则template-odr)
- [2. SFINAE 替换失败非错原则](#2-sfinae-替换失败非错原则)
  - [2.1 SFINAE 发生阶段与适用边界](#21-sfinae-发生阶段与适用边界)
  - [2.2 `std::enable_if` 偏特化原理](#22-stdenable_if-偏特化原理)
  - [2.3 `enable_if` 注入位置对比](#23-enable_if-注入位置对比)
- [3. 类型萃取与成员探测](#3-类型萃取与成员探测)
  - [3.1 `std::declval` 表达式推演](#31-stddeclval-表达式推演)
  - [3.2 `std::void_t` 成员探测](#32-stdvoid_t-成员探测)
- [4. 编译期分支修剪（`if constexpr`）](#4-编译期分支修剪if-constexpr)
- [5. Concepts 与约束体系](#5-concepts-与约束体系)
  - [5.1 Concepts 设计动机](#51-concepts-设计动机)
  - [5.2 `concept` 定义与 `requires` 语法](#52-concept-定义与-requires-语法)
  - [5.3 偏序重载决议（Subsumption）](#53-偏序重载决议subsumption)
- [6. 模板约束演进速查表](#6-模板约束演进速查表)

---

## 1. 模板编译模型与代码生成

C++ 模板不是运行时的多态，而是**编译期的代码生成器（Code Generation Blueprint）**。

### 1.1 两阶段名字查找（Two-phase Lookup）

编译器在解析模板时，将其严格分为两个阶段：

```text
源码中的模板定义
      │
      ├─► 【第 1 阶段：解析期 (Parsing Time)】
      │    • 检查不依赖模板参数的名字 (Non-dependent names) 语法错误
      │    • 此时模板参数 T 未知，依赖 T 的名字不会被绑定
      │
      └─► 【第 2 阶段：实例化期 (Instantiation Time)】
           • 给定具体的具体类型 (如 T = int)
           • 解析所有依赖模板参数的名字 (Dependent names) 并生成具体机器码
```

#### 为什么需要 `typename` 和 `template` 关键字？
当编译器在第 1 阶段看到 `T::iterator * p;` 时，存在语法二义性：
* 解释 A：声明一个指向类型 `T::iterator` 的指针变量 `p`；
* 解释 B：将静态成员变量 `T::iterator` 乘以全局变量 `p`。

C++ 标准规定：**在第 1 阶段，编译器默认将 `T::xxx` 视为静态成员变量**。若要指示其为类型，必须显式前缀 `typename`：

```cpp
template <typename Container>
void iterate(const Container& c) {
    typename Container::const_iterator it = c.begin(); // ✔️ 必须加 typename 指明是嵌套类型
}
```

---

### 1.2 模板单一定义规则（Template ODR）

* **普通函数的 ODR**：一个实体在整个程序中只能有一处定义，否则引发链接期 `duplicate symbol` 冲突。
* **模板的 ODR**：
  * 模板定义可以被包含在多个翻译单元（`.cpp`）中；
  * 编译器会在每个使用该模板的 `.o` 目标文件中各自生成一份弱符号（Weak Symbol）代码；
  * **链接器（Linker）在最终合并时，会自动保留其中一份，并安全地丢弃其他所有重复副本**。

---

## 2. SFINAE 替换失败非错原则

### 2.1 SFINAE 发生阶段与适用边界

**SFINAE 原则**：在函数模板重载决议的**实参推导与形参替换阶段**，如果尝试用实参替换模板形参时产生了不合法的类型或表达式，**编译器不会将其视为编译错误，而是直接将该模板从候选函数集合（Overload Candidate Set）中静默剔除**。

> [!IMPORTANT]
> SFINAE 仅发生在**函数签名（模板参数列表、返回值类型、函数形参列表）的替换期**。如果替换成功进入了函数体内部，而函数体内的代码存在语法错误，则**依然会直接引发致命编译报错**！

---

### 2.2 `std::enable_if` 偏特化原理

查看 `<type_traits>` 中 `std::enable_if` 的标准实现：

```cpp
// 1. 主模板：默认未定义内部 ::type
template <bool Condition, typename T = void>
struct enable_if {};

// 2. 偏特化版本：当 Condition 为 true 时，定义 ::type 为 T
template <typename T>
struct enable_if<true, T> {
    using type = T;
};

// C++14 别名模板
template <bool Condition, typename T = void>
using enable_if_t = typename enable_if<Condition, T>::type;
```

* 当 `Condition` 为 `true` 时：`enable_if_t<true, int>` 成功展开为 `int`。
* 当 `Condition` 为 `false` 时：`enable_if<false, int>` 内部没有 `type` 成员，尝试访问 `::type` 触发替换失败，依据 SFINAE 原则静默剔除该候选。

---

### 2.3 `enable_if` 注入位置对比

```cpp
// 场景：我们希望编写一个仅接受“整数类型”的函数 process()

// 方式 1：注入到返回值类型 (最直观，但无法用于构造函数)
template <typename T>
std::enable_if_t<std::is_integral_v<T>, void> process(T val) {}

// 方式 2：注入到匿名模板参数 (最推荐的现代写法，适用于普通函数与构造函数)
template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
void process(T val) {}

// 方式 3：注入到函数参数默认值 (较少使用)
template <typename T>
void process(T val, std::enable_if_t<std::is_integral_v<T>>* = nullptr) {}
```

---

## 3. 类型萃取与成员探测

### 3.1 `std::declval` 表达式推演

在编译期，我们经常需要推导表达式 `t.foo()` 的返回类型，但类型 `T` 可能没有默认构造函数，或者构造函数是私有的。

```cpp
template <typename T>
std::add_rvalue_reference_t<T> declval() noexcept;
```
* **机制**：`std::declval<T>()` 仅能出现在 `decltype`、`sizeof` 等**不求值操作数（Unevaluated Operand）**上下文里。它在不实际创建任何对象实例的前提下，假想并返回一个 `T&&` 右值引用，供编译器推导其成员函数的调用结果类型。

---

### 3.2 `std::void_t` 成员探测

C++17 引入了极简的元编程神器 `std::void_t`：
```cpp
template <typename...>
using void_t = void;
```

#### 探测一个类是否拥有 `.begin()` 成员函数：
```cpp
#include <type_traits>
#include <vector>

// 1. 主模板：默认为 false
template <typename T, typename = void>
struct has_begin : std::false_type {};

// 2. 偏特化版本：当 declval<T>().begin() 存在且合法时匹配成功
template <typename T>
struct has_begin<T, std::void_t<decltype(std::declval<T>().begin())>> : std::true_type {};

// 验证：
static_assert(has_begin<std::vector<int>>::value == true);
static_assert(has_begin<int>::value == false);
```

---

## 4. 编译期分支修剪（`if constexpr`）

在 C++17 之前，根据类型执行不同逻辑必须写多个重载模板配合 SFINAE。C++17 引入了 `if constexpr`：

```cpp
template <typename T>
void processValue(T val) {
    if constexpr (std::is_pointer_v<T>) {
        if (val) std::cout << *val << "\n"; // 仅在 T 是指针类型时编译此分支
    } else {
        std::cout << val << "\n";           // 仅在 T 是非指针类型时编译此分支
    }
}
```

> [!NOTE]
> `if constexpr` 会在编译期直接**丢弃不满足条件的分支（Discarded Statement）**，被丢弃分支内的代码不会被实例化，因此即使被丢弃分支在当前类型下语义不合法（例如对 `int` 执行解引用 `*val`），也不会导致编译失败。

---

## 5. Concepts 与约束体系

### 5.1 Concepts 设计动机

SFINAE 虽然功能强大，但其本质是“利用语法替换规则的漏洞做条件编译”，带来的最大痛点是：**一旦匹配失败，编译器会吐出长达几百行的错误调用栈，极难排查**。

C++20 引入了**一等公民级别的约束体系——Concepts**，直接在语言层面提供类型约束接口。

---

### 5.2 `concept` 定义与 `requires` 语法

```cpp
#include <concepts>

// 1. 定义一个 Concept: 要求类型 T 必须支持 .begin() 与 .end()
template <typename T>
concept Iterable = requires(T t) {
    // A. 简单约束 (Simple Requirement): 表达式必须合法
    t.begin();
    t.end();
    
    // B. 复合约束 (Compound Requirement): 表达式合法且返回值满足特定约束
    { *t.begin() } -> std::convertible_to<int>;
};

// 2. 四种使用 Concept 约束函数的方式：

// 方式 1: 经典 requires 子句
template <typename T>
requires Iterable<T>
void printAll(const T& c) {}

// 方式 2: 紧凑模板参数约束
template <Iterable T>
void printAll(const T& c) {}

// 方式 3: 后置 requires 约束
template <typename T>
void printAll(const T& c) requires Iterable<T> {}

// 方式 4: 极简 auto 简写函数模板 (Terse Syntax)
void printAll(const Iterable auto& c) {}
```

---

### 5.3 偏序重载决议（Subsumption）

Concepts 原生支持继承与蕴含关系（Subsumption），编译器会自动选择**约束更为严格（More Constrained）**的重载版本，无需像 SFINAE 那样写复杂的互斥条件：

```cpp
template <typename T>
concept Number = std::is_arithmetic_v<T>;

template <typename T>
concept FloatingPoint = Number<T> && std::floating_point<T>; // 包含 Number 且更严格

void compute(Number auto x) {
    std::cout << "通用数值计算\n";
}

void compute(FloatingPoint auto x) {
    std::cout << "浮点专用高精度计算 (自动命中更严格的 Concept！)\n";
}

int main() {
    compute(10);   // 命中 compute(Number)
    compute(3.14); // 自动优先命中 compute(FloatingPoint)
}
```

---

## 6. 模板约束演进速查表

| 技术维度 | C++98 / 03 | C++11 / 14 (SFINAE) | C++17 | C++20 (Concepts) |
| :--- | :--- | :--- | :--- | :--- |
| **条件约束** | 宏与模板全特化 | `std::enable_if` / `enable_if_t` | `std::void_t` 探测 | `concept` + `requires` |
| **分支分派** | 多重函数重载 | 结构体偏特化标签分派（Tag Dispatch） | `if constexpr` | `if constexpr` + Concepts |
| **语法侵入性** | 极高 | 破坏函数签名可读性 | 中等 | **原生一流语法，极其整洁** |
| **编译器报错** | 晦涩冗长 | 极其晦涩（模板替换回退深渊） | 稍有改善 | **精准提示“不满足某具体约束”** |
| **与 `cpp-mlir` 关系** | - | MLIR `llvm::isa`/`cast` 类型萃取基石 | PassWrapper 条件匹配 | 现代 C++ 编译器 Dialect 接口约束 |
