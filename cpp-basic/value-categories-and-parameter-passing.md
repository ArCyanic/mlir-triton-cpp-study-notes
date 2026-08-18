# 值类别、传参模型与移动语义

> 本文以**编译器 IR 节点（`Operation` 与 `Value`）的设计与构建**为单一贯穿场景，系统性拆解 C++ 中的**值类别体系（Value Categories）**、**形参正交空间与 5×6 绑定矩阵**、**重载决议规则**、**移动语义底层实现**以及**完美转发与返回值优化（RVO/NRVO）**。

## 目录

- [1. 场景与问题建模](#1-场景与问题建模)
- [2. 表达式与值类别体系](#2-表达式与值类别体系)
  - [2.1 值类别分类体系](#21-值类别分类体系)
  - [2.2 身份与可移动性（Identity / Movability）](#22-身份与可移动性identity--movability)
  - [2.3 表达式值类别推导](#23-表达式值类别推导)
- [3. 形参空间与绑定矩阵](#3-形参空间与绑定矩阵)
  - [3.1 形参类型的正交分解](#31-形参类型的正交分解)
  - [3.2 5×6 实参绑定矩阵](#32-56-实参绑定矩阵)
  - [3.3 引用绑定规则](#33-引用绑定规则)
  - [3.4 `const T&&` 语义分析](#34-const-t-语义分析)
  - [3.5 传参范式与 Sink 模式](#35-传参范式与-sink-模式)
  - [3.6 重载决议优先级](#36-重载决议优先级)
- [4. 移动语义与类型退化](#4-移动语义与类型退化)
  - [4.1 `std::move` 静态转型机制](#41-stdmove-静态转型机制)
  - [4.2 右值引用变量的左值属性](#42-右值引用变量的左值属性)
  - [4.3 指针所有权转移机制](#43-指针所有权转移机制)
  - [4.4 `const` 对象的移动退化](#44-const-对象的移动退化)
- [5. 拷贝消除与返回值优化](#5-拷贝消除与返回值优化)
  - [5.1 保证拷贝消除与 Prvalue](#51-保证拷贝消除与-prvalue)
  - [5.2 `return std::move` 反优化](#52-return-stdmove-反优化)
- [6. 万能引用与完美转发](#6-万能引用与完美转发)
  - [6.1 万能引用与引用折叠](#61-万能引用与引用折叠)
  - [6.2 `std::forward` 转发机制](#62-stdforward-转发机制)
- [7. 传参选型矩阵与决策流](#7-传参选型矩阵与决策流)
  - [7.1 参数设计决策流](#71-参数设计决策流)
  - [7.2 传参方式特性对比](#72-传参方式特性对比)

## 1. 场景与问题建模

在编译器中间表示（IR）的构建过程中，我们经常需要实现一个 `Operation` 类，用来保存算子的名字、输入操作数 ID 列表以及输出结果 ID 列表：

```cpp
class Operation {
public:
    // 问题：这里的参数应该如何设计？
    // 是传 const 引用、传值 + std::move、传右值引用，还是使用模板完美转发？
    Operation(int id, 
              /* ??? */ std::string name, 
              /* ??? */ std::vector<int> operands, 
              /* ??? */ std::vector<int> results);

private:
    int id_;
    std::string name_;
    std::vector<int> operands_;
    std::vector<int> results_;
};
```

用户在创建算子时，会有以下几种典型调用场景：

1. **场景 A（传入命名变量/左值）**：外部维护了一个 `std::vector<int> op_list`，后续还要继续读取。
2. **场景 B（传入临时字面量/右值）**：直接传入 `{1, 2}` 或 `std::vector<int>{v_in, t_in}`。
3. **场景 C（主动移交所有权/移后亡值）**：外部调用 `std::move(temp_op_list)`。

选择哪种参数形式，决定了编译器在底层是触发**昂贵的堆内存深拷贝（Deep Copy）**，还是**纳秒级的指针所有权转移（Move）**。

## 2. 表达式与值类别体系

要搞懂传参和移动语义，必须首先厘清 C++ 类型系统中最为基础的分类法则——**值类别（Value Categories）**。

### 2.1 值类别分类体系

在 C 语言时代，划分标准极其朴素：
- 能放在赋值号 `=` 左边的叫**左值（Lvalue）**。
- 只能放在赋值号 `=` 右边的叫**右值（Rvalue）**。

但到了 C++11 引入移动语义后，这种二分法彻底失效了。C++ 标准委员会（由 Bjarne Stroustrup 等人）重新确立了**五大值类别分类树**：

```text
                  表达式 Expression
                 /                 \
     泛左值 (glvalue)             右值 (rvalue)
        /          \             /          \
  左值 (lvalue)    亡值 (xvalue)    纯右值 (prvalue)
```

### 2.2 身份与可移动性（Identity / Movability）

现代 C++ 通过两个独立的正交维度来判定任意表达式的值类别：

1. **是否有身份标识（Has Identity）**：
   - 该表达式是否占据明确的内存地址？能否对其取地址（`&expr`）？
2. **是否可被移动（Can be Moved from）**：
   - 该表达式所绑定的资源生命周期是否即将结束？我们是否能安全地“偷”走它的内部资源？

根据这两个维度的组合，正好划分出 3 种基础类别与 2 种复合类别：

| 基础值类别 | 具有身份标识 (Identity)？ | 可以被安全移动 (Movable)？ | 核心定义与通俗解释 |
| :--- | :---: | :---: | :--- |
| **`lvalue` (左值)** | **✔️ 有** | ❌ 否 | 拥有明确内存名字的对象，生命周期跨越当前表达式。 |
| **`xvalue` (亡值 / eXpiring value)** | **✔️ 有** | **✔️ 是** | 拥有明确地址，但生命周期即将终结，明确允许被掏空（如 `std::move(x)`）。 |
| **`prvalue` (纯右值 / Pure rvalue)** | ❌ 无 | **✔️ 是** | 计算的中间临时结果、字面量（在 C++17 中代表“对象的初始化蓝图”）。 |

- **泛左值（`glvalue = lvalue + xvalue`）**：所有拥有物理内存身份的表达式总称。
- **右值（`rvalue = xvalue + prvalue`）**：所有允许被移动转移资源的表达式总称。

### 2.3 表达式值类别推导

```cpp
int a = 10;
int b = 20;

a;                      // lvalue (有名字，能取地址 &a)
a + b;                  // prvalue (临时算术结果，无持久地址)
100;                    // prvalue (字面量)
std::string("conv");    // prvalue (临时构造的对象)
std::move(a);           // xvalue (有地址，但被强制标记为即将死亡，可移动)
static_cast<int&&>(a);  // xvalue (静态类型转换产生的右值引用)
```

## 3. 形参空间与绑定矩阵

在掌握了值类别之后，我们从类型系统第一性原理出发，剖析函数形参是如何与实参进行合法性绑定与重载决议的。

### 3.1 形参类型的正交分解

对于任意给定的底层基类型 `T`，形参的完整类型空间由两个正交维度笛卡尔积生成：
1. **维度一：是否具备 `const` 限定符**（`non-const` vs `const`）；
2. **维度二：引用修饰符（Ref-qualifier）**（无引用 `值` vs 左值引用 `&` vs 右值引用 `&&`）。

```text
               【形参类型空间的 6 种完整正交组合】

                 维度二：引用修饰 (Ref-qualifier)
                 ┌──────────────┬──────────────┬──────────────┐
                 │ 无引用 (值)  │ 左值引用 (&) │ 右值引用 (&&)│
  ┌──────────────┼──────────────┼──────────────┼──────────────┤
维│ 非常量 (non) │     T        │     T&       │     T&&      │
度├──────────────┼──────────────┼──────────────┼──────────────┤
一│ 常量 (const) │   const T    │   const T&   │   const T&&  │
  └──────────────┴──────────────┴──────────────┴──────────────┘
```

### 3.2 5×6 实参绑定矩阵

实参（表达式）同样由 **是否 `const`** 和 **值类别（左值 / 纯右值 / 亡值）** 组合出 5 种实际输入状态。以下是标准 C++ 编译器的全量合法性决议网格：

| 实参形态（调用端传入） | 1. `T`<br>(传值) | 2. `const T`<br>(只读传值) | 3. `T&`<br>(可写左值引) | 4. `const T&`<br>(只读左值引) | 5. `T&&`<br>(可写右值引) | 6. `const T&&`<br>(只读右值引) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **① 非常量左值**<br>`T a;` | ✔️ (拷贝) | ✔️ (拷贝) | **✔️ 精确匹配** | ✔️ (只读绑定) | ❌ 拒绝左值 | ❌ 拒绝左值 |
| **② 常量左值**<br>`const T ca;` | ✔️ (拷贝) | ✔️ (拷贝) | ❌ **拒绝 (权限放大)** | **✔️ 精确匹配** | ❌ 拒绝左值 | ❌ 拒绝左值 |
| **③ 纯右值**<br>`T()` 或 字面量 | ✔️ (移动) | ✔️ (移动) | ❌ **拒绝 (临死不可写)**| ✔️ (延长寿命) | **✔️ 精确匹配** | ✔️ (只读绑定) |
| **④ 非常量亡值**<br>`std::move(a)` | ✔️ (移动) | ✔️ (移动) | ❌ **拒绝 (临死不可写)**| ✔️ (只读绑定) | **✔️ 精确匹配** | ✔️ (只读绑定) |
| **⑤ 常量亡值**<br>`std::move(ca)` | ✔️ (退化拷贝) | ✔️ (退化拷贝) | ❌ **拒绝 (权限放大)** | ✔️ (只读绑定) | ❌ **拒绝 (权限放大)** | **✔️ 精确匹配** |

### 3.3 引用绑定规则

1. **规则一：权限只允许收紧，绝不允许放大（Const-correctness）**：
   - `const` 实参**绝对不能**绑定到非 `const` 引用（`T&` 和 `T&&`），防止调用方通过别名意外修改只读内存，击穿类型安全。
2. **规则二：引用符号表达调用方的“意图契约”**：
   - **`&`（左值引用）** 宣称：*“我要的是活着的、有明确持久名字的对象，我可能会就地修改它。”* $\implies$ 拒绝纯右值和亡值（修改一个马上析构的临时对象无意义且危险）。
   - **`&&`（右值引用）** 宣称：*“我要的是快要销毁的对象（右值），准备掏空并窃取它的内部堆资源。”* $\implies$ 拒绝普通左值（防止调用方后续还要使用的具名变量被悄悄掏空）。
3. **规则三：C++ 标准唯一的救生圈特例 —— `const T&`**：
   - 为了让通用只读接口能够无差别接收临时对象，C++ 规定 `const T&` 允许绑定右值，并在编译期**将该右值临时对象的生命周期延长至当前引用的作用域结束**。

### 3.4 `const T&&` 语义分析

在实际工程中，几乎没有人书写 `void foo(const T&& x)`，其原因在于它语义自相矛盾：
* **输入限制**：它声明了 `&&`，要求必须传入将亡的右值，拒绝一切左值；
* **操作受限**：它同时声明了 `const`，禁止修改，**导致无法调用移动构造函数，无法执行指针窃取**；
* **尴尬后果**：它要求调用方传入一个本可被移动掏空的临时对象，自己却只能对它进行**全量深拷贝**。
* **唯一现实场景**：在极少数库中用于显式 `= delete` 禁用右值入参（如 `void push(const T&&) = delete;`）。

### 3.5 传参范式与 Sink 模式

在实际工程设计中，形参设计主要收敛为以下 4 种经典模式：

```text
                                实参传递 (Argument Binding)
                                            │
        ┌───────────────────┬───────────────┴───────────────┬───────────────────┐
        ▼                   ▼                               ▼                   ▼
【1. 传值 T】       【2. 常量左值引用 const T&】      【3. 非常量左值引用 T&】   【4. 右值引用 T&&】
 内部 Sink 所有权转移   万能只读绑定 (只读观察)           只绑定具名左值 (就地修改)  只绑定右值 (专属性移动)
```

#### 1. 传值 + `std::move`（Sink Parameter 模式）
```cpp
void setOperands(std::vector<int> operands) { // operands 是私有中转站
    operands_ = std::move(operands);
}
```
* **实参传左值**：形参触发 1 次拷贝构造，函数内 `move` 移入成员 $\implies$ 共 1 Copy + 1 Move；
* **实参传右值**：形参触发 1 次移动构造，函数内 `move` 移入成员 $\implies$ 共 0 Copy + 2 Move（全生命周期 0 堆分配）。
* **适用场景**：函数确定要**无条件接管该对象的所有权**并保存在成员变量中。

#### 2. 常量左值引用（`const T&`）
```cpp
void inspectOperands(const std::vector<int>& operands) {
    // 纯只读观察，不触发任何构造与拷贝
}
```
* **适用场景**：只读检查、属性计算、短生命周期遍历。

#### 3. 非常量左值引用（`T&`）
```cpp
void appendOperands(std::vector<int>& operands) {
    operands.push_back(42); // 就地修改外部实参
}
```
* **适用场景**：In-Out 出入参就地修改。

#### 4. 右值引用（`T&&`）
```cpp
void transferOperands(std::vector<int>&& operands) {
    operands_ = std::move(operands); // 显式要求调用方放弃所有权
}
```
* **适用场景**：强制调用方在入参处必须写 `std::move(...)` 或传入临时对象。

### 3.6 重载决议优先级

当多个重载函数同时出现时，编译器依据**“最精准匹配优先（Best Match）”**裁决：

```cpp
void process(const std::vector<int>& v); // 重载 A (const&)
void process(std::vector<int>&& v);      // 重载 B (&&)
```

| 传入实参形态 | 决议命中 | 底层决议原理 |
| :--- | :---: | :--- |
| **`std::vector<int> x; process(x);` (左值)** | **重载 A** | 左值无法绑定到右值引用 `&&`，只有 `const&` 是合法候选。 |
| **`process(std::vector<int>{1, 2});` (纯右值)** | **重载 B** | 右值引用 `&&` 对右值是**精确匹配（Exact Match）**，优于需附加 const 转换的 `const&`。 |
| **`process(std::move(x));` (亡值)** | **重载 B** | `std::move(x)` 产出 `xvalue`，精确匹配右值引用 `&&`。 |

## 4. 移动语义与类型退化

### 4.1 `std::move` 静态转型机制

很多人误以为 `std::move` 会在运行时搬运内存。**完全错误！**

查看 `std::move` 的标准库实现（位于 `<utility>`）：

```cpp
template <typename T>
constexpr std::remove_reference_t<T>&& move(T&& t) noexcept {
    // 强制转换为无引用的右值引用类型 (xvalue)
    return static_cast<std::remove_reference_t<T>&&>(t);
}
```

- **机制**：`std::move` 在底层**没有任何 CPU 汇编指令产生**！
- 它唯一的职责就是欺骗类型系统：**将一个左值强制转为亡值（`xvalue`），从而允许下游的重载决议去命中移动构造函数或移动赋值运算符**。

### 4.2 右值引用变量的左值属性

这是一个极高频的面试与工程认知陷阱：

```cpp
void foo(std::vector<int>&& rvalue_ref) {
    // 陷阱提问：在 foo 函数体内，rvalue_ref 表达式是左值还是右值？
    
    // 错误写法：
    // operands_ = rvalue_ref;            // ❌ 触发拷贝构造！
    
    // 正确写法：
    operands_ = std::move(rvalue_ref); // ✔️ 触发移动构造！
}
```

#### 表达式左值属性成因
- **类型（Type） vs 值类别（Value Category）的区别**：
  - `rvalue_ref` 的**类型**是 `std::vector<int>&&`（右值引用类型）。
  - 但 `rvalue_ref` 作为一个表达式，它**拥有明确的名字和内存地址**（在当前函数的栈帧上）。
- **核心黄金法则**：
  > **“If it has a name, it is an lvalue!”（凡是有名字的表达式，全都是左值！）**

因为 `rvalue_ref` 是具名左值，如果编译器自动把它当右值处理，那么你在第一行读了它之后，它可能就失效了，第二行就无法再次使用。因此在函数体内如果想继续转移它，**必须显式再次使用 `std::move(rvalue_ref)`**。

### 4.3 指针所有权转移机制

以 `std::vector` 的移动构造函数为例，底层执行过程仅涉及 3 个指针的赋值：

```cpp
// std::vector 移动构造伪代码
vector(vector&& other) noexcept {
    // 1. 窃取对方的堆指针与大小
    this->data_start_ = other.data_start_;
    this->data_end_   = other.data_end_;
    this->capacity_end_ = other.capacity_end_;

    // 2. 将对方重置为空指针，防止其析构时释放这块内存
    other.data_start_ = nullptr;
    other.data_end_   = nullptr;
    other.capacity_end_ = nullptr;
}
```

```text
移动前：
[other]   ─── ptr ───► Heap Buffer [0x1000 ... 0x2000] (100 MB 数据)
[this]    ─── ptr ───► nullptr

移动后 (耗时 < 5 纳秒)：
[other]   ─── ptr ───► nullptr (变为空容器，安全析构)
[this]    ─── ptr ───► Heap Buffer [0x1000 ... 0x2000] (100 MB 数据完好无损)
```

### 4.4 `const` 对象的移动退化

如果在声明变量时加了 `const`，随后对其执行 `std::move`：

```cpp
const std::vector<int> locked_vec = {1, 2, 3};
Operation op(1, "Conv", std::move(locked_vec), {}); // 传入 std::move(const)
```

#### 拷贝退化执行过程

1. `std::move(locked_vec)` 产生的类型是 `const std::vector<int>&&`（常量右值引用）。
2. `std::vector` 的移动构造函数签名是 `vector(vector&& other)`（非常量右值引用）。
3. 编译器发现：`const std::vector<int>&&` **无法绑定到非常量 `vector&&`**！
4. 编译器退而求其次，去匹配拷贝构造函数 `vector(const vector& other)`。
5. 👉 **结果**：**编译器静默执行了深拷贝，没有任何告警，但移动优化完全失效！**

> ⚠️ **编译器工程准则**：**永远不要对准备 `std::move` 的对象添加 `const` 修饰符！**

## 5. 拷贝消除与返回值优化

### 5.1 保证拷贝消除与 Prvalue

在 C++17 之前，通过函数返回一个对象或用临时变量初始化对象，理论上会触发移动构造。

而在 C++17 中，标准重新定义了 **纯右值（Prvalue）** 的语义：
- **Prvalue 不再是一个临时对象，它仅仅是一个“初始化处方（Recipe for initialization）”**。
- 只有当 Prvalue 真正用来初始化一个具名对象时，内存才会在该目标位置就地构建（Materialization）。

```cpp
std::vector<int> createOperands() {
    return std::vector<int>{1, 2, 3, 4}; // Prvalue
}

// C++17 下：0 次拷贝，0 次移动！直接在 main 函数的 op 栈内存上就地构造！
std::vector<int> op = createOperands();
```

### 5.2 `return std::move` 反优化

看下面这道极高频的经典反模式代码：

```cpp
// ❌ 极度糟糕的写法：
std::vector<int> buildVector() {
    std::vector<int> local_vec = {1, 2, 3};
    return std::move(local_vec); // 💥 反优化！强行摧毁了 NRVO 优化！
}

// ✔️ 最佳工业级写法：
std::vector<int> buildVector() {
    std::vector<int> local_vec = {1, 2, 3};
    return local_vec; // 触发具名返回值优化 (NRVO)，直接在外部调用者栈上就地构造！
}
```

#### 深度机制解析

1. **直接 `return local_vec;`**：编译器应用 **NRVO（Named Return Value Optimization）**。编译器在编译期将 `local_vec` 的内存地址直接与外部接收变量合二为一，实现 **0 拷贝、0 移动**。
2. **写了 `return std::move(local_vec);`**：
   - 你显式将一个具名局部变量强转为了右值引用（`xvalue`）。
   - 编译器看到你返回的是一个引用表达式，**被迫放弃 NRVO 优化**，转而退化去执行一次移动构造函数！
3. **结论**：在局部变量返回时显式写 `std::move`，不仅没有变快，反而硬生生增加了一次移动构造的开销。

## 6. 万能引用与完美转发

当我们在编写通用工厂函数（如 `std::make_unique` 或 MLIR 的 `OpBuilder::create<OpType>`）时，如何将任意数量、任意值类别的参数原封不动地传递到底层构造函数？

### 6.1 万能引用与引用折叠

当 `&&` 出现在**需要类型推导的模板参数**中时，它不再是普通的右值引用，而是**万能引用（Forwarding Reference）**：

```cpp
template <typename T>
void wrapper(T&& arg); // T&& 是万能引用，可以接收一切左值和右值！
```

#### 引用折叠规则（Reference Collapsing Rules）

C++ 严禁“引用的引用”（如 `int& &`），但在模板实例化时，会按照以下规则折叠：

| 实例化出现的组合 | 最终折叠结果 | 记忆口诀 |
| :---: | :---: | :--- |
| `&  +  &` | **`&`** (左值引用) | **只要有左值引用 `&` 参与，一律折叠为左值引用 `&`** |
| `&  +  &&` | **`&`** (左值引用) | |
| `&& +  &` | **`&`** (左值引用) | |
| `&& +  &&` | **`&&`** (右值引用) | **当且仅当两边都是 `&&` 时，才折叠为右值引用 `&&`** |

### 6.2 `std::forward` 转发机制

如果我们在 `wrapper` 里直接调用 `target(arg)`，根据“凡是有名字的都是左值”，`arg` 又变回了左值。为了恢复其最初的值类别，必须使用 `std::forward<T>`：

```cpp
template <typename T>
void wrapper(T&& arg) {
    target(std::forward<T>(arg)); // 完美还原 arg 最初的左值/右值属性
}
```

#### `std::forward` 实现机制

```cpp
template <typename T>
constexpr T&& forward(std::remove_reference_t<T>& t) noexcept {
    return static_cast<T&&>(t); // 依靠 T 中的引用信息触发引用折叠！
}
```

- **如果外部传入左值**：`T` 被推导为 `std::string&`，`static_cast<std::string& &&>` 折叠为 `std::string&`（返回左值引用，触发深拷贝）。
- **如果外部传入右值**：`T` 被推导为 `std::string`，`static_cast<std::string&&>`（返回右值引用，触发零拷贝移动）。

## 7. 传参选型矩阵与决策流

### 7.1 参数设计决策流

针对编译器与高性能 C++ 开发中的接口设计，遵循以下参数选型决策树：

```text
                               参数设计决策树 (Parameter Design Flowchart)
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
     【函数内部只是只读读取】                                          【函数需要存储/获取该对象所有权】
                 │                                                             │
         ┌───────┴───────┐                                             ┌───────┴───────┐
         ▼               ▼                                             ▼               ▼
   [小标量/基础类型]    [复杂对象/容器]                               [普通业务类/算子节点]   [通用模板/工厂函数]
   (int, float, etc.)  (std::vector, etc.)                           (如 Operation 构造)    (如 make_unique)
         │               │                                             │               │
         ▼               ▼                                             ▼               ▼
     【直接传值】     【传 const T&】                              【传值 + std::move】    【万能引用 T&& + forward】
                      (或 std::string_view / ArrayRef)             (Sink Parameter 模式)  (Perfect Forwarding)
```

### 7.2 传参方式特性对比

| 传参方式 | 签名示例 | 左值实参开销 | 右值实参开销 | 核心适用场景 |
| :--- | :--- | :---: | :---: | :--- |
| **小对象传值** | `void fn(int x)` | 寄存器传参 | 寄存器传参 | 基础算术类型、指针、轻量句柄 |
| **只读引用** | `void fn(const std::vector<int>& v)` | 0 拷贝（传指针） | 0 拷贝（传指针） | 只读查询、遍历计算、不存储所有权 |
| **连续只读视图** | `void fn(llvm::ArrayRef<int> v)` | 0 拷贝（仅传指针+长度） | 0 拷贝（仅传指针+长度） | LLVM/MLIR 中接收连续数组、切片视图 |
| **Sink 参数模式** | `Operation(std::vector<int> v)` | 1 Copy + 1 Move | **2 Move (0 堆分配)** | **构造函数、Setter、明确转交所有权的实体类** |
| **右值引用专属** | `void fn(std::vector<int>&& v)` | ❌ 拒绝左值 | **1 Move (极限)** | 极少数强制要求掏空原对象的特定内部 API |
| **完美转发模板** | `template<class T> void fn(T&& v)` | 1 Copy | **1 Move (极限)** | 通用工厂函数、多参数模板容器插入（`emplace`） |
