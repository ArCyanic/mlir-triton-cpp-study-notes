# C++ 核心机制速查手册

> 本文档作为 `cpp-basic` 知识底座的**高信息密度极速查阅索引**。剔除所有冗余叙述，直击各语言特性的**核心机制、物理内存模型、关键代码/图解、关键设计边界与典型避坑准则**，并附带深度解析文档的精准章节跳转。

---

## 目录

- [1. 值类别、传参机制与生命周期](#1-值类别传参机制与生命周期)
  - [1.1 五大值类别划分（Identity vs Movability）](#11-五大值类别划分identity-vs-movability)
  - [1.2 形参正交分解与 5×6 绑定矩阵](#12-形参正交分解与-56-绑定矩阵)
  - [1.3 Sink Parameter 传值 + `std::move` 模式](#13-sink-parameter-传值--stdmove-模式)
  - [1.4 `std::move` 本质与 `const` 移动退化](#14-stdmove-本质与-const-移动退化)
  - [1.5 返回值优化（RVO/NRVO）与 `return std::move` 反优化](#15-返回值优化rvonrvo与-return-stdmove-反优化)
  - [1.6 万能引用、引用折叠规则与 `std::forward`](#16-万能引用引用折叠规则与-stdforward)
- [2. STL 容器内部原理、迭代器与异常安全](#2-stl-容器内部原理迭代器与异常安全)
  - [2.1 迭代器能力层级链与 `contiguous_iterator`](#21-迭代器能力层级链与-contiguous_iterator)
  - [2.2 `std::vector` 三指针内存模型与扩容失效规则](#22-stdvector-三指针内存模型与扩容失效规则)
  - [2.3 `std::deque` 中控映射表与分段缓冲区模型](#23-stddeque-中控映射表与分段缓冲区模型)
  - [2.4 哈希表 `unordered_map` Rehash 后的指针与迭代器分歧](#24-哈希表-unordered_map-rehash-后的指针与迭代器分歧)
  - [2.5 `noexcept` 移动构造与 `std::move_if_noexcept` 强异常安全](#25-noexcept-移动构造与-stdmove_if_noexcept-强异常安全)
- [3. 智能指针、控制块与 RAII 资源管理](#3-智能指针控制块与-raii-资源管理)
  - [3.1 `std::unique_ptr` 零开销抽象与 EBCO 自定义删除器](#31-stdunique_ptr-零开销抽象与-ebco-自定义删除器)
  - [3.2 `std::shared_ptr` 控制块双计数结构与内存排布](#32-stdshared_ptr-控制块双计数结构与内存排布)
  - [3.3 `make_shared` 单次分配 vs `shared_ptr(new T)` 弱引用滞留权衡](#33-make_shared-单次分配-vs-shared_ptrnew-t-弱引用滞留权衡)
  - [3.4 `std::weak_ptr` 非拥有观察与原子提升 `.lock()`](#34-stdweak_ptr-非拥有观察与原子提升-lock)
  - [3.5 `std::enable_shared_from_this` 机制与 `bad_weak_ptr` 异常](#35-stdenable_shared_from_this-机制与-bad_weak_ptr-异常)
- [4. 闭包原理、递归 Lambda 与类型擦除](#4-闭包原理递归-lambda-与类型擦除)
  - [4.1 编译器闭包类生成（Functor Lowering）与捕获内存排布](#41-编译器闭包类生成functor-lowering与捕获内存排布)
  - [4.2 异步/延迟执行中的引用捕获悬空与广义移动捕获](#42-异步延迟执行中的引用捕获悬空与广义移动捕获)
  - [4.3 递归 Lambda 的四种解法对比](#43-递归-lambda-的四种解法对比)
  - [4.4 `std::function` 类型擦除原理与 SBO 小对象优化](#44-stdfunction-类型擦除原理与-sbo-小对象优化)
  - [4.5 `std::function` vs `llvm::function_ref` 零分配视图](#45-stdfunction-vs-llvmfunction_ref-零分配视图)
- [5. 模板元编程、SFINAE 与 C++20 Concepts](#5-模板元编程sfinae-与-c20-concepts)
  - [5.1 模板两阶段名字查找与 `typename` 消歧义](#51-模板两阶段名字查找与-typename-消歧义)
  - [5.2 模板单一定义规则（Template ODR）与弱符号合并](#52-模板单一定义规则template-odr与弱符号合并)
  - [5.3 SFINAE 原则与 `std::enable_if_t` 偏特化实现](#53-sfinae-原则与-stdenable_if_t-偏特化实现)
  - [5.4 编译期成员探测：`std::void_t` 与 `std::declval`](#54-编译期成员探测stdvoid_t-与-stddeclval)
  - [5.5 C++17 `if constexpr` 编译期分支修剪](#55-c17-if-constexpr-编译期分支修剪)
  - [5.6 C++20 Concepts、`requires` 表达式与偏序重载](#56-c20-conceptsrequires-表达式与偏序重载)
- [6. 编译工具链、链接模型与 ELF 格式](#6-编译工具链链接模型与-elf-格式)
  - [6.1 编译四阶段调用链（预处理/编译/汇编/链接）](#61-编译四阶段调用链预处理编译汇编链接)
  - [6.2 作用域、存储期与链接属性（Linkage）](#62-作用域存储期与链接属性linkage)
  - [6.3 ELF Section 链接视图 vs Segment 装载视图与 `.bss` 机制](#63-elf-section-链接视图-vs-segment-装载视图与-bss-机制)
  - [6.4 C++ Name Mangling 编码规则与 `extern "C"` 桥接](#64-c-name-mangling-编码规则与-extern-c-桥接)
  - [6.5 全局对象 `.init_array` 构造链与 Meyers' Singleton](#65-全局对象-init_array-构造链与-meyers-singleton)
  - [6.6 静态库单遍扫描算法与命令行链接顺序敏感性](#66-静态库单遍扫描算法与命令行链接顺序敏感性)

---

## 1. 值类别、传参机制与生命周期

### 1.1 五大值类别划分（Identity vs Movability）

```text
                  表达式 Expression
                 /                 \
     泛左值 (glvalue)             右值 (rvalue)
        /          \             /          \
  左值 (lvalue)    亡值 (xvalue)    纯右值 (prvalue)
```

* **判定维度**：
  * **`lvalue`**：有身份（能取地址 `&e`），不可移动（生命周期跨越当前表达式，如具名变量 `int x`）；
  * **`xvalue`**（亡值）：有身份，**允许移动**（如 `std::move(x)` 产生的移后状态，或返回右值引用的函数调用）；
  * **`prvalue`**（纯右值）：无身份，允许移动（计算字面量 `10`、临时对象 `std::string("a")`，在 C++17 中作为对象的初始化蓝图）。
* **核心不变式**：
  * **“If it has a name, it is an lvalue!”（凡是有名字的表达式，一律为左值）**。
  * `decltype(x)` 产生变量的声明类型；`decltype((x))`（带双括号）产生表达式的值类别引用类型（左值产生 `T&`）。
* **🔗 深度解析**：[value-categories-and-parameter-passing.md 第 2 章](value-categories-and-parameter-passing.md#2-表达式与值类别体系)

### 1.2 形参正交分解与 5×6 绑定矩阵

```text
                 维度二：引用修饰 (Ref-qualifier)
                 ┌──────────────┬──────────────┬──────────────┐
                 │ 无引用 (值)  │ 左值引用 (&) │ 右值引用 (&&)│
  ┌──────────────┼──────────────┼──────────────┼──────────────┤
维│ 非常量 (non) │     T        │     T&       │     T&&      │
度├──────────────┼──────────────┼──────────────┼──────────────┤
一│ 常量 (const) │   const T    │   const T&   │   const T&&  │
  └──────────────┴──────────────┴──────────────┴──────────────┘
```

| 实参形态 / 形参签名 | `T` (传值) | `const T` (只读值) | `T&` (可写左值引) | `const T&` (只读左值引) | `T&&` (可写右值引) | `const T&&` (只读右值引) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **非常量左值 `T a`** | ✔️ 拷贝 | ✔️ 拷贝 | **✔️ 精确匹配** | ✔️ 只读绑定 | ❌ 拒绝左值 | ❌ 拒绝左值 |
| **常量左值 `const T ca`** | ✔️ 拷贝 | ✔️ 拷贝 | ❌ 权限放大 | **✔️ 精确匹配** | ❌ 拒绝左值 | ❌ 拒绝左值 |
| **纯右值 `T()`** | ✔️ 移动 | ✔️ 移动 | ❌ 临死不可写 | ✔️ 延长寿命 | **✔️ 精确匹配** | ✔️ 只读绑定 |
| **非常量亡值 `move(a)`** | ✔️ 移动 | ✔️ 移动 | ❌ 临死不可写 | ✔️ 只读绑定 | **✔️ 精确匹配** | ✔️ 只读绑定 |
| **常量亡值 `move(ca)`** | ✔️ 拷贝 | ✔️ 拷贝 | ❌ 权限放大 | ✔️ 只读绑定 | ❌ 权限放大 | **✔️ 精确匹配** |

* **三大底层不变式**：
  1. **权限不放大**：`const` 实参绝对无法绑定到非 `const` 引用（`T&` / `T&&`）；
  2. **意图契约**：`&` 专绑活体左值（可写）；`&&` 专绑将亡右值（窃取所有权）；
  3. **标准救生圈**：`const T&` 允许绑定右值并延长生命周期；
  4. **反模式**：`const T&&` 既要右值又不准修改，导致退化深拷贝，现实中仅用于 `= delete` 禁用右值入参。
* **🔗 深度解析**：[value-categories-and-parameter-passing.md 第 3 章](value-categories-and-parameter-passing.md#3-形参空间与绑定矩阵)

---

### 1.3 Sink Parameter 传值 + `std::move` 模式

* **模式代码**：
  ```cpp
  class Node {
      std::string name_;
      std::vector<int> operands_;
  public:
      // 传值形参作为私有中转站
      Node(std::string name, std::vector<int> operands)
          : name_(std::move(name)), operands_(std::move(operands)) {}
  };
  ```
* **分流路径与开销分析**：
  * **实参传左值**（`Node(name_var, op_var)`）：形参触发 1 次深拷贝构造，初始化列表触发 1 次移动构造 $\implies$ **共 1 Copy + 1 Move**（保全调用方原数据）；
  * **实参传右值/临时对象**（`Node("Add", {1, 2})`）：形参触发 1 次移动构造，初始化列表触发 1 次移动构造 $\implies$ **共 0 Copy + 2 Move**（全生命周期 0 堆内存分配）。
* **🔗 深度解析**：[value-categories-and-parameter-passing.md 第 3.5 节](value-categories-and-parameter-passing.md#35-传参范式与-sink-模式)

---

### 1.4 `std::move` 本质与 `const` 移动退化

* **底层实现本质**：
  ```cpp
  template <typename T>
  constexpr std::remove_reference_t<T>&& move(T&& t) noexcept {
      return static_cast<std::remove_reference_t<T>&&>(t); // 纯编译期静态类型转换
  }
  ```
* **物理执行**：`std::move` **不产生任何运行时 CPU 汇编指令**，它仅通知编译器类型系统将表达式值类别标记为 `xvalue`，使下游重载决议能够命中移动构造函数 `T(T&&)`。
* **`const T` 移动退化陷阱**：
  ```cpp
  const std::vector<int> v = {1, 2, 3};
  std::vector<int> v2 = std::move(v); // 💥 静默退化为全量深拷贝！
  ```
  * **机理**：`std::move(const)` 产生 `const std::vector<int>&&`，因带 `const` 无法绑定非常量 `vector(vector&&)`，编译器自动退回调用拷贝构造 `vector(const vector&)`。
* **🔗 深度解析**：[value-categories-and-parameter-passing.md 第 4 章](value-categories-and-parameter-passing.md#4-移动语义与类型退化)

---

### 1.5 返回值优化（RVO/NRVO）与 `return std::move` 反优化

* **C++17 保证拷贝消除（RVO）**：返回同类型纯右值（`return std::vector<int>{1, 2};`）由标准强制保证直接在调用方存储位置就地构造，**0 拷贝、0 移动**。
* **具名返回值优化（NRVO）**：返回局部具名变量（`return local_v;`）由编译器将局部变量与外部返回值栈地址合二为一。
* **反模式陷阱**：
  ```cpp
  std::vector<int> makeBuffer() {
      std::vector<int> buf = {1, 2, 3};
      return std::move(buf); // 💥 严重反优化！强行摧毁 NRVO！
  }
  ```
  * **机理**：显式写 `std::move` 将具名左值强转为右值引用表达式，编译器被迫放弃 0 开销的 NRVO 栈地址合并，退化去执行 1 次移动构造函数。
* **🔗 深度解析**：[value-categories-and-parameter-passing.md 第 5 章](value-categories-and-parameter-passing.md#5-拷贝消除与返回值优化)

---

### 1.6 万能引用、引用折叠规则与 `std::forward`

* **万能引用（Forwarding Reference）**：必须同时满足 **`T&&` 形式** 与 **`T` 处于推导上下文**（如函数模板 `template<class T> void f(T&&)` 或 `auto&&`）。
* **引用折叠规则（Reference Collapsing）**：
  | 模板实参推导出的 `T` | 签名 `T&&` 折叠后的真实形参 | 最终效果 |
  | :--- | :--- | :--- |
  | `U&`（实参为左值） | `U&  &&` $\to$ **`U&`** | 成为左值引用 |
  | `U`（实参为右值） | `U   &&` $\to$ **`U&&`** | 成为右值引用 |
* **完美转发实现**：
  ```cpp
  template <typename T>
  constexpr T&& forward(std::remove_reference_t<T>& t) noexcept {
      return static_cast<T&&>(t); // 依靠 T 中的引用信息触发折叠，还原原始值类别
  }
  ```
* **🔗 深度解析**：[value-categories-and-parameter-passing.md 第 6 章](value-categories-and-parameter-passing.md#6-万能引用与完美转发)

---

## 2. STL 容器内部原理、迭代器与异常安全

### 2.1 迭代器能力层级链与 `contiguous_iterator`

```text
               Input Iterator (输入，单向单遍读取 single-pass)
                     │
                     ▼
              Forward Iterator (前向，多遍读取与可重入遍历 multi-pass)
                     │
                     ▼
           Bidirectional Iterator (双向，支持 operator--)
                     │
                     ▼
           Random Access Iterator (随机访问，O(1) 常数时间 it + n / it - n)
                     │
                     ▼
            Contiguous Iterator (连续迭代器，逻辑连续严格对应物理连续内存)
```

* **能力层级**：`Input -> Forward -> Bidirectional -> RandomAccess -> Contiguous`。
* **核心判定**：
  * `Input` 为单遍（single-pass）不可重入流读取（如 `std::istream_iterator`）；
  * `Forward` 保证多次保存副本并重入安全遍历（如 `std::forward_list`）；
  * `Contiguous` 严格保证物理地址连续性：`&(*(it + n)) == (&*it) + n`（如 `std::vector`、原生指针 `T*`）。
* **🔗 深度解析**：[iterators-and-stl-containers.md 第 1 章](iterators-and-stl-containers.md#1-stl-四层抽象架构与设计哲学)

---

### 2.2 `std::vector` 三指针内存模型与扩容失效规则

```text
std::vector<T> 控制句柄 (栈上 24 字节)
┌──────────────────────┬──────────────────────┬──────────────────────┐
│  T* _M_start (8B)    │  T* _M_finish (8B)   │ T* _M_end_storage(8B)│
└──────────┬───────────┴──────────┬───────────┴──────────┬───────────┘
           │                      │                      │
           ▼                      ▼                      ▼
堆内存:    [ Elem 0 ][ Elem 1 ]...[ Elem N-1 ]           [ 空闲未构造槽位 ... ]
```

* **物理排布**：64 位下固定占用 **24 字节**（`size() = finish - start`，`capacity() = end_of_storage - start`）。
* **失效判定规则**：
  * **扩容插入**（`size == capacity`）：申请新堆内存并搬迁旧元素，旧内存 `free`，**所有迭代器、指针、引用彻底失效**；
  * **非扩容中间插入**：插入点之前的迭代器有效，**插入点及其之后的所有迭代器/引用因元素后移全部失效**；
  * **删除操作（`erase`）**：被删元素及后续元素的迭代器/引用失效。
* **🔗 深度解析**：[iterators-and-stl-containers.md 第 3.1 与 4.1 节](iterators-and-stl-containers.md#31-stdvector连续堆内存与三指针模型)

---

### 2.3 `std::deque` 中控映射表与分段缓冲区模型

```text
std::deque 控制中枢
┌────────────────────────────────────────┐
│  T** _M_map ──► 指针数组 (连续中控表)    │
└──────────────────┬───┬───┬─────────────┘
                   │   │   │
                   ▼   ▼   ▼
Buffer 0:        [   |   |   |   ] (定长固定大小物理页，如 512 字节)
Buffer 1:        [ Elem 0 | Elem 1 | Elem 2 | Elem 3 ]
Buffer 2:        [ Elem 4 | Elem 5 |   |   ]
```

* **物理架构**：连续指针数组 `_M_map` 充当中控索引，指向多块固定大小的独立物理缓冲区（Chunks）。
* **特性**：
  * 头尾插入/删除 $O(1)$，**绝不触发已有缓冲区块的物理搬迁**；
  * 头尾插入可能引发中控表重分配导致迭代器失效，但**指向已有元素的指针和引用保持物理有效**。
* **🔗 深度解析**：[iterators-and-stl-containers.md 第 3.2 节](iterators-and-stl-containers.md#32-stddeque中控索引数组与分段连续缓冲区map-of-chunks)

---

### 2.4 哈希表 `unordered_map` Rehash 后的指针与迭代器分歧

```text
Bucket Array (连续指针数组): [ ptr 0 | ptr 1 | ptr 2 | ... ]
                                │
                                ▼
                       Node (Key-Value 节点)
                       ┌──────────────────────┐
                       │ - next : Node* (8B)  │ ──► 碰撞链表下一个节点
                       │ - hash : size_t (8B) │
                       │ - value: pair<K, V>  │
                       └──────────────────────┘
```

* **Rehash 行为**：当 `size > bucket_count * max_load_factor` 时，重新分配更大的桶数组并重构链表。
* **失效分歧**：
  * **迭代器（Iterators）全部失效**（因为桶数组重建，遍历游标重置）；
  * **指针与引用（Pointers & References）依然有效**（节点实体仍在原独立堆内存上，仅链表指针重挂）。
* **🔗 深度解析**：[iterators-and-stl-containers.md 第 4.3 节](iterators-and-stl-containers.md#43-哈希容器-rehash-后的指针与迭代器分歧)

---

### 2.5 `noexcept` 移动构造与 `std::move_if_noexcept` 强异常安全

* **强异常安全（Strong Guarantee）契约**：操作具备事务性，若抛出异常，容器必须完整回滚到调用前的初始状态。
* **标准库内部判定**：
  ```cpp
  template <typename T>
  using move_if_noexcept_t = std::conditional_t<
      !std::is_nothrow_move_constructible_v<T> && std::is_copy_constructible_v<T>,
      const T&, // 强制走拷贝构造 (深拷贝)
      T&&       // 走移动构造
  >;
  ```
* **性能影响**：若自定义类的移动构造未显式标注 `noexcept`，`vector` 在扩容搬迁时为防止中途抛异常破坏旧数据，会**强制放弃移动，退化为全量深拷贝**。
* **🔗 深度解析**：[iterators-and-stl-containers.md 第 5 章](iterators-and-stl-containers.md#5-异常安全保证与-stdmove_if_noexcept-扩容机制)

---

## 3. 智能指针、控制块与 RAII 资源管理

### 3.1 `std::unique_ptr` 零开销抽象与 EBCO 自定义删除器

* **物理排布**：默认占用 **8 字节**（单个裸指针大小），汇编层与原生指针 100% 同构。
* **自定义删除器优化**：
  ```cpp
  struct CudaDeleter { void operator()(void* p) const { cudaFree(p); } };
  // 空基类优化 (EBCO) 生效，大小仍为 8 字节
  std::unique_ptr<float, CudaDeleter> d_buf;

  // 函数指针删除器大小变为 16 字节 (8B 裸指针 + 8B 函数指针)
  std::unique_ptr<FILE, decltype(&fclose)> f_buf(fopen(...), &fclose);
  ```
* **🔗 深度解析**：[smart-pointers-and-raii.md 第 2 章](smart-pointers-and-raii.md#2-stdunique_ptr零开销独占所有权)

---

### 3.2 `std::shared_ptr` 控制块双计数结构与内存排布

```text
std::shared_ptr<T> 句柄 (16 字节)
┌──────────────────────────────┬──────────────────────────────┐
│        T* ptr (8 字节)       │   ControlBlock* cb (8 字节)   │
└──────────────┬───────────────┴──────────────┬───────────────┘
               │                              │
               ▼                              ▼
      ┌─────────────────┐           ┌────────────────────────────────┐
      │  T 实际对象实体   │           │      控制块 (Control Block)     │
      └─────────────────┘           │────────────────────────────────│
                                    │ - strong_ref_count: 引用计数    │
                                    │ - weak_ref_count:   弱引用计数  │
                                    │ - custom_deleter:   自定义删除器│
                                    │ - custom_allocator: 分配器元数据│
                                    └────────────────────────────────┘
```

* **大小**：占用 **16 字节**（8B 对象指针 + 8B 控制块指针）。
* **生命周期分解**：
  * 当 `strong_ref_count` 原子减至 0 时：**立即调用对象 `T` 的析构函数**；
  * 当 `weak_ref_count` 也减至 0 时：**彻底 `free` 释放控制块自身占用的堆内存**。
* **🔗 深度解析**：[smart-pointers-and-raii.md 第 3.1 与 3.2 节](smart-pointers-and-raii.md#3-stdshared_ptr共享所有权与控制块解密)

---

### 3.3 `make_shared` 单次分配 vs `shared_ptr(new T)` 弱引用滞留权衡

```text
1. std::make_shared<T>() 内存排布 (单次 malloc，紧凑连续)：
┌─────────────────────────────────────────────────────────────┐
│                 一块连续堆内存 (Single Allocation Chunk)        │
│  [ Control Block (strong/weak count) ] [   T Object Data   ]  │
└─────────────────────────────────────────────────────────────┘

2. std::shared_ptr<T>(new T()) 内存排布 (两次独立的 malloc)：
┌─────────────────────────────┐         ┌─────────────────────────────┐
│    控制块 (Control Block)     │ ──ptr──►│       T 目标对象实体        │
│ [ Strong Ref ] [ Weak Ref ] │         │       [ Object Data ]       │
└─────────────────────────────┘         └─────────────────────────────┘
```

* **`std::make_shared<T>()`**：单次 `malloc` 同时开辟控制块与对象，CPU 缓存局部性极佳。
  * ⚠️ **弱引用滞留陷阱**：控制块与对象紧挨在同一物理内存块中，若有长寿命 `weak_ptr` 存活，即使 `T` 已析构，**整块连续物理内存也无法归还 OS**。
* **`std::shared_ptr<T>(new T())`**：两次独立 `malloc`，`strong_count == 0` 时立刻单独释放 `T` 的内存。
* **🔗 深度解析**：[smart-pointers-and-raii.md 第 3.3 节](smart-pointers-and-raii.md#33-make_shared-vs-shared_ptrnew-t-深度权衡)

---

### 3.4 `std::weak_ptr` 非拥有观察与原子提升 `.lock()`

* **机制**：不增加 `strong_ref_count`，用于打破有向图环引用。
* **访问规范**：严禁直接解引用，必须调用 `.lock()` 在**原子操作下尝试提升为 `std::shared_ptr`**：
  ```cpp
  if (std::shared_ptr<Node> locked = weak_node.lock()) {
      locked->process(); // 保证在 locked 作用域内对象绝不会被并发析构
  }
  ```
* **🔗 深度解析**：[smart-pointers-and-raii.md 第 4 章](smart-pointers-and-raii.md#4-stdweak_ptr非拥有观察与循环引用破除)

---

### 3.5 `std::enable_shared_from_this` 机制与 `bad_weak_ptr` 异常

* **实现原理**：基类内部持有一个私有的 `std::weak_ptr<T> __weak_this_`，在对象首次被 `shared_ptr` 接管时由标准库构造函数自动初始化。
* **避坑要点**：
  * 严禁直接 `shared_ptr<T>(this)`（会产生两个独立的控制块，导致同一物理地址被 `delete` 两次引发 Double Free 崩溃）；
  * 在对象未交由 `shared_ptr` 管理前调用 `shared_from_this()` 会直接抛出 **`std::bad_weak_ptr` 异常**。
* **🔗 深度解析**：[smart-pointers-and-raii.md 第 5 章](smart-pointers-and-raii.md#5-stdenable_shared_from_this安全获取自身的-shared_ptr)

---

## 4. 闭包原理、递归 Lambda 与类型擦除

### 4.1 编译器闭包类生成（Functor Lowering）与捕获内存排布

```cpp
// 源码:
int bias = 10;
auto add = [bias](int x) { return x + bias; };

// 编译器生成等价仿函数:
class __lambda_id {
    int bias_; // 捕获变量变为成员变量
public:
    explicit __lambda_id(int b) : bias_(b) {}
    int operator()(int x) const { return x + bias_; } // 默认带 const
};
```

* **排布规则**：
  * 值捕获 `[x]` $\to$ 独立成员 `T x_`；
  * 引用捕获 `[&x]` $\to$ 引用成员 `T& x_`（物理占用 8 字节指针空间）；
  * 无捕获 `[]` $\to$ 空类（1 字节），**可隐式退化为普通函数指针 `int (*)(int)`**。
* **🔗 深度解析**：[lambdas-closures-and-type-erasure.md 第 1 章](lambdas-closures-and-type-erasure.md#1-lambda-表达式与闭包类closure-class底层生成机制)

---

### 4.2 异步/延迟执行中的引用捕获悬空与广义移动捕获

* **悬空崩溃**：异步分发或跨作用域回调若采用引用捕获 `[&]` 栈变量，原函数栈帧销毁后执行必导致段错误。
* **C++14 广义移动捕获**：
  ```cpp
  auto task = [buf = std::move(heap_buffer)]() { buf->process(); };
  ```
* **🔗 深度解析**：[lambdas-closures-and-type-erasure.md 第 2 章](lambdas-closures-and-type-erasure.md#2-捕获机制与生命周期陷阱)

---

### 4.3 递归 Lambda 的四种解法对比

```cpp
// 方案 1: std::function (有类型擦除与虚调用开销)
std::function<void(int)> dfs = [&](int u) { if (u > 0) dfs(u - 1); };

// 方案 2: C++14 泛型 Lambda 自传递 (0 开销完全内联)
auto dfs = [](auto&& self, int u) -> void { if (u > 0) self(self, u - 1); };
dfs(dfs, 10);

// 方案 3: Y-Combinator 结构体包装 (消除调用端自传递冗余)
auto dfs = YCombinator([&](auto&& self, int u) { if (u > 0) self(u - 1); });

// 方案 4: C++23 显式对象形参 Deducing This (原生最佳语法)
auto dfs = [](this auto&& self, int u) { if (u > 0) self(u - 1); };
```
* **🔗 深度解析**：[lambdas-closures-and-type-erasure.md 第 3 章](lambdas-closures-and-type-erasure.md#3-递归-lambdarecursive-lambda的困境与四种解法)

---

### 4.4 `std::function` 类型擦除原理与 SBO 小对象优化

```text
std::function<void(int)> 外部句柄 (32 字节)
┌───────────────────────────────────────────────────────────┐
│ void (*invoker_)(void* storage, int)   (8 字节调用分派指针) │
│ void (*manager_)(void* dest, void* src)(8 字节生命周期管理器)│
│ char storage_[16]                      (16 字节内联存储缓冲区)│
└───────────────────────────────────────────────────────────┘
```

* **类型擦除模型**：包含对象存储区指针 + 静态分派函数指针（Invoker）+ 生命周期管理函数指针（Manager）。
* **SBO 机制**：若闭包体积 $\le 16 \sim 24$ 字节且支持 `noexcept` 移动，直接在栈上内联构造（**0 堆分配**）；超出体积则退化为在堆上 `malloc` 开辟存储。
* **🔗 深度解析**：[lambdas-closures-and-type-erasure.md 第 4 章](lambdas-closures-and-type-erasure.md#4-stdfunction-底层解密类型擦除与小对象优化sbo)

---

### 4.5 `std::function` vs `llvm::function_ref` 零分配视图

* **`std::function`**：独占拥有闭包对象，具备长生命周期拷贝/移动管理，开销较重。
* **`llvm::function_ref`**：仅占用 **16 字节**（`void*` 对象指针 + `R(*)(void*, Args...)` 函数指针），**非拥有、纯下向借用**，**严格 0 堆内存分配**，专为编译器 Pass 内部的同步遍历优化（如 `Operation::walk`）。
* **🔗 深度解析**：[lambdas-closures-and-type-erasure.md 第 5 章](lambdas-closures-and-type-erasure.md#5-可调用对象横向对比concrete-lambda-vs-stdfunction-vs-llvmfunction_ref)

---

## 5. 模板元编程、SFINAE 与 C++20 Concepts

### 5.1 模板两阶段名字查找与 `typename` 消歧义

* **时序分流**：
  * **第 1 阶段（解析期）**：检查非依赖性名字语法错误；
  * **第 2 阶段（实例化期）**：绑定依赖性名字，为具体特化类型生成机器码。
* **消歧义原则**：编译器在第 1 阶段默认将 `T::nested_name` 视为静态成员变量。若要指示其为类型，**必须显式添加 `typename T::nested_name`**；若调用成员模板，必须添加 `obj.template method<U>()`。
* **🔗 深度解析**：[templates-sfinae-and-concepts.md 第 1.1 节](templates-sfinae-and-concepts.md#11-两阶段名字查找two-phase-lookup与-typenametemplate-消歧义)

---

### 5.2 模板单一定义规则（Template ODR）与弱符号合并

* 模板定义与隐式实例化可合法出现在多个 `.cpp` 编译单元中。编译器为每个实例生成弱符号（Weak Symbol），**由链接器在合并阶段自动去重保留唯一物理副本**。
* **🔗 深度解析**：[templates-sfinae-and-concepts.md 第 1.2 节](templates-sfinae-and-concepts.md#12-模板单一定义规则template-odr与头文件包含模式)

---

### 5.3 SFINAE 原则与 `std::enable_if_t` 偏特化实现

* **SFINAE 契约**：在函数重载候选集推导替换期间产生的类型非法错误，不中断编译，静默剔除该候选。
* **标准实现模型**：
  ```cpp
  template <bool B, typename T = void> struct enable_if {};
  template <typename T> struct enable_if<true, T> { using type = T; };
  ```
* **最佳注入位置**：匿名模板参数 `template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>`。
* **🔗 深度解析**：[templates-sfinae-and-concepts.md 第 2 章](templates-sfinae-and-concepts.md#2-sfinae-核心原则substitution-failure-is-not-an-error)

---

### 5.4 编译期成员探测：`std::void_t` 与 `std::declval`

* **`std::declval<T>()`**：在不求值（Unevaluated）上下文假想返回 `T&&`，无需构造实例即可推演表达式类型。
* **`std::void_t` 成员探测模板**：
  ```cpp
  template <typename T, typename = void>
  struct has_begin : std::false_type {};

  template <typename T>
  struct has_begin<T, std::void_t<decltype(std::declval<T>().begin())>> : std::true_type {};
  ```
* **🔗 深度解析**：[templates-sfinae-and-concepts.md 第 3 章](templates-sfinae-and-concepts.md#3-编译期类型萃取与成员探测type-traits--detection-idiom)

---

### 5.5 C++17 `if constexpr` 编译期分支修剪

* **机制**：条件为 false 的分支为**废弃语句（Discarded Statement）**，编译器**完全不对其进行模板实例化**，允许在分支内书写针对特定类型才合法的代码（如指针解引用）。
* **🔗 深度解析**：[templates-sfinae-and-concepts.md 第 4 章](templates-sfinae-and-concepts.md#4-c17-编译期分支修剪if-constexpr)

---

### 5.6 C++20 Concepts、`requires` 表达式与偏序重载

```cpp
template <typename T>
concept Iterable = requires(T t) {
    t.begin();
    t.end();
    { *t.begin() } -> std::same_as<int>;
};

// 极简使用:
void print(const Iterable auto& container);
```
* **优势**：
  1. 消除 SFINAE 产生的数百行深层回退报错栈，精准提示缺失的约束；
  2. **偏序重载决议（Subsumption）**：编译器自动优先命中约束更为严格（More Constrained）的 Concept 分支。
* **🔗 深度解析**：[templates-sfinae-and-concepts.md 第 5 章](templates-sfinae-and-concepts.md#5-c20-concepts-与-requires-约束体系终极替代-sfinae)

---

## 6. 编译工具链、链接模型与 ELF 格式

### 6.1 编译四阶段调用链（预处理/编译/汇编/链接）

```text
源码 (.cpp) ──[gcc -E]──► 预处理文件 (.i) ──[gcc -S]──► 汇编文件 (.s)
                                                              │
                                                      [gcc -c (as)]
                                                              │
                                                              ▼
可执行文件/库 ◄──[gcc -o (ld)]── 符号决议与重定位 ◄── 目标文件 (.o, ELF)
```

1. **预处理（`-E`）**：宏替换、头文件文本拷贝、`#ifdef` 展开；
2. **编译（`-S`）**：语法语义分析、LLVM IR 优化、输出目标汇编 `.s`；
3. **汇编（`-c`）**：汇编器将助记符翻译为机器码，生成 ELF 重定位目标文件 `.o`；
4. **链接（`-o`）**：链接器执行符号决议与地址重定位，合并为最终 ELF 可执行文件或 `.so`。
* **🔗 深度解析**：[compiler-toolchain-and-elf-linking.md 第 1 章](compiler-toolchain-and-elf-linking.md#1-c-编译与工具链四大阶段流水线)

---

### 6.2 作用域、存储期与链接属性（Linkage）

| 链接属性 | 符号表可见性 | 声明语法 |
| :--- | :--- | :--- |
| **外部链接 (External)** | 全局可见（`.symtab` 标记 `GLOBAL`） | 普通全局函数/变量、`extern` 声明 |
| **内部链接 (Internal)** | 仅本 `.cpp` 目标文件可见（`.symtab` 标记 `LOCAL`） | 全局 `static` 变量/函数、匿名命名空间 `namespace { ... }` |
| **无链接 (No Linkage)** | 不进入链接符号表 | 函数局部变量、局部 `using` |

* **🔗 深度解析**：[compiler-toolchain-and-elf-linking.md 第 2 章](compiler-toolchain-and-elf-linking.md#2-c-实体的四大管理维度作用域存储期与链接属性)

---

### 6.3 ELF Section 链接视图 vs Segment 装载视图与 `.bss` 机制

```text
      【链接器视角：Section 节区】                         【装载器视角：Segment 内存段】
┌─────────────────────────────────────────┐       ┌─────────────────────────────────────────┐
│ .text       : 机器指令代码 (可执行)      │ ────► │ Segment 1: PT_LOAD [R-X] (只读代码内存页)│
│ .rodata     : 只读常量、虚表 (vtable)   │       └─────────────────────────────────────────┘
├─────────────────────────────────────────┤       ┌─────────────────────────────────────────┐
│ .data       : 已初始化的全局 / static 变量│ ────► │ Segment 2: PT_LOAD [RW-] (可读写数据页)  │
│ .bss        : 未初始化/初值为0的全局变量 │       └─────────────────────────────────────────┘
│ .symtab     : 符号表                     │
│ .rela.text  : 指令地址重定位表           │
│ .init_array : 全局对象构造函数数组       │
└─────────────────────────────────────────┘
```

* **`.bss` 零磁盘原理**：磁盘文件中**仅记录所需字节大小，不占实际磁盘文件体积**；操作系统 `execve` 装载时直接分配已清零的物理内存页。
* **🔗 深度解析**：[compiler-toolchain-and-elf-linking.md 第 3 章](compiler-toolchain-and-elf-linking.md#3-elf-二进制格式section-与-segment-双重视图)

---

### 6.4 C++ Name Mangling 编码规则与 `extern "C"` 桥接

* **编码机理**：C++ 为了支持重载与命名空间，遵循 Itanium ABI 将符号编码（如 `mlir::process(int, float)` $\to$ `_ZN4mlir7processEif`）。
* **`extern "C"` 核心作用**：
  1. 关闭 C++ Name Mangling，生成裸符号名 `process`；
  2. 建立与纯 C、Python（`ctypes`/`cffi`）或 Rust 的跨语言 ABI 桥接。
* **🔗 深度解析**：[compiler-toolchain-and-elf-linking.md 第 4 章](compiler-toolchain-and-elf-linking.md#4-c-名字改编name-mangling与-abi-互操作)

---

### 6.5 全局对象 `.init_array` 构造链与 Meyers' Singleton

* **main 前执行链路**：操作系统内核装载后，`__libc_start_main` 在进入 `main()` 前遍历 ELF 的 `.init_array` 节依次调用全局对象构造函数。
* **静态初始化死锁规避（Meyers' Singleton）**：不同 `.cpp` 间的全局对象构造顺序未定义；将全局变量改为局部静态变量（`static Context instance;`），C++11 保证其在**首次使用时（On-first-use）线程安全地懒加载初始化**。
* **🔗 深度解析**：[compiler-toolchain-and-elf-linking.md 第 5 章](compiler-toolchain-and-elf-linking.md#5-全局对象的生命周期init_array-与-main-前执行机制)

---

### 6.6 静态库单遍扫描算法与命令行链接顺序敏感性

* **链接器单遍扫描模型**：维护已选目标集合 $E$、未决符号集合 $U$、已定义符号集合 $D$。
* **命令行顺序准则**：若静态库放在调用者左侧（`g++ -lA -lB app.o`），扫描到库时 $U$ 为空，静态库中的 `.o` 直接被丢弃导致后续报 `undefined reference`。**被依赖的静态库必须严格放置在命令行右侧（`g++ app.o -lA -lB`）**。
* **🔗 深度解析**：[compiler-toolchain-and-elf-linking.md 第 6 章](compiler-toolchain-and-elf-linking.md#6-静态库与动态库链接机制)
