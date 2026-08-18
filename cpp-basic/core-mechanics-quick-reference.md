# C++ 核心机制速查手册

> 本文档作为 `cpp-basic` 知识底座的**高信息密度极速查阅索引**。剔除所有冗余叙述，直击各语言特性的**核心机制、物理内存模型、关键代码/图解、关键设计边界与典型避坑准则**，并附带深度解析文档的精准章节跳转。

---

## 1. 值类别、传参机制与生命周期

### 1.1 五大值类别划分

```text
                  表达式 Expression
                 /                 \
     泛左值 (glvalue)             右值 (rvalue)
        /          \             /          \
   左值 (lvalue)    亡值 (xvalue)    纯右值 (prvalue)
```

- **`lvalue`**：有身份（能取地址 `&e`），不可移动（生命周期跨越当前表达式，如具名变量 `int x`）；
- **`xvalue`**（亡值）：有身份，**允许移动**（如 `std::move(x)` 产生的移后状态，或返回右值引用的函数调用）；
- **`prvalue`**（纯右值）：无身份，允许移动（计算字面量 `10`、临时对象 `std::string("a")`，在 C++17 中作为对象的初始化处方）。

**核心不变式**：
- **“If it has a name, it is an lvalue!”（凡是有名字的表达式，在当前作用域内皆为左值）**；
- `decltype(x)` 产生变量的声明类型；`decltype((x))`（带双括号）产生表达式的值类别引用类型（左值产生 `T&`）。

> [!TIP]
> 🔗 **深度解析**：[value-categories-and-parameter-passing.md 第 2 章](value-categories-and-parameter-passing.md#2-表达式五大值类别体系)

---

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

**四大底层不变式**：
1. **权限不放大**：`const` 实参绝对无法绑定到非 `const` 引用（`T&` / `T&&`）；
2. **意图契约**：`&` 专绑活体左值（可写）；`&&` 专绑将亡右值（窃取所有权）；
3. **标准救生圈**：`const T&` 允许绑定右值并延长生命周期；
4. **反模式**：`const T&&` 既要右值又不准修改，导致退化深拷贝，现实中仅用于 `= delete` 禁用右值入参。

> [!TIP]
> 🔗 **深度解析**：[value-categories-and-parameter-passing.md 第 5.1 节](value-categories-and-parameter-passing.md#51-六乘五形参实参绑定决议矩阵)

---

### 1.3 Sink Parameter 传值与 std::move 模式

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

**分流路径与开销分析**：
- **实参传左值**（`Node(name_var, op_var)`）：形参触发 1 次深拷贝构造，初始化列表触发 1 次移动构造 $\implies$ **共 1 Copy + 1 Move**（保全调用方原数据）；
- **实参传右值/临时对象**（`Node("Add", {1, 2})`）：形参触发 1 次移动构造，初始化列表触发 1 次移动构造 $\implies$ **共 0 Copy + 2 Move**（全生命周期 0 堆内存分配）。

> [!TIP]
> 🔗 **深度解析**：[value-categories-and-parameter-passing.md 第 5.2 节](value-categories-and-parameter-passing.md#52-经典传参范式与重载决议优先级)

---

### 1.4 std::move 本质与 const 移动退化

```cpp
template <typename T>
constexpr std::remove_reference_t<T>&& move(T&& t) noexcept {
    return static_cast<std::remove_reference_t<T>&&>(t); // 纯编译期静态类型转换
}
```

- **物理执行**：`std::move` **不产生任何运行时 CPU 汇编指令**，它仅通知编译器类型系统将表达式值类别标记为 `xvalue`，使下游重载决议能够命中移动构造函数 `T(T&&)`。

**const T 移动退化陷阱**：

```cpp
const std::vector<int> v = {1, 2, 3};
auto v2 = std::move(v); // 💥 陷阱：移动构造函数接受 vector&& (非 const)，无法绑定 const rvalue
                        // 编译器退而求其次选择拷贝构造 vector(const vector&)，静默执行全量深拷贝！
```

> [!TIP]
> 🔗 **深度解析**：[value-categories-and-parameter-passing.md 第 3.2 节](value-categories-and-parameter-passing.md#32-容器指针所有权转移与移后状态契约)

---

### 1.5 NRVO 拷贝消除与反模式

```cpp
// ✔️ 正确：触发编译器具名返回值优化 (NRVO)，在调用方栈上就地构造，0 Copy 0 Move
std::vector<int> buildVec() {
    std::vector<int> res;
    // ... 填充数据
    return res; 
}

// ❌ 错误反模式：强行 std::move 破坏 NRVO，退化为 1 次 Move 构造
std::vector<int> badBuildVec() {
    std::vector<int> res;
    return std::move(res); // 💥 反优化！
}
```

> [!TIP]
> 🔗 **深度解析**：[value-categories-and-parameter-passing.md 第 3.3 节](value-categories-and-parameter-passing.md#33-保证拷贝消除与返回值优化)

---

### 1.6 万能引用、折叠规则与 std::forward

- **万能引用（Forwarding Reference）**：当且仅当发生模板类型推导时的 `T&&`（或 `auto&&`）；
- **引用折叠黄金定律**：**只要有左值引用 `&` 参与折叠，结果一律坍缩为左值引用 `&`**；只有两边皆为 `&&` 时才折叠为 `&&`。

```cpp
template <typename T>
void wrapper(T&& arg) {
    // std::forward 依靠显式传递的模板类型实参 T 还原其原始左/右值属性
    target(std::forward<T>(arg)); 
}
```

**贪婪构造函数陷阱防御**：

```cpp
template <typename T>
    requires (!std::same_as<std::remove_cvref_t<T>, Node>) // C++20 约束排斥自身拷贝
explicit Node(T&& name) : name_(std::forward<T>(name)) {}
```

> [!TIP]
> 🔗 **深度解析**：[value-categories-and-parameter-passing.md 第 4 章](value-categories-and-parameter-passing.md#4-万能引用与完美转发机制)

---

## 2. 容器架构与迭代器模型

### 2.1 STL 四层正交解耦模型

```text
┌──────────────┐         ┌───────────────┐         ┌──────────────┐
│  算法层      │ ──遍历─► │  迭代器层     │ ──访问─► │  容器层      │
│ (Algorithms) │         │  (Iterators)  │         │ (Containers) │
└──────────────┘         └───────────────┘         └──────┬───────┘
                                                          │ 申请内存
                                                          ▼
                                                   ┌──────────────┐
                                                   │ 内存分配层   │
                                                   │ (Allocators) │
                                                   └──────────────┘
```

- **正交解耦价值**：将 $M$ 个算法与 $N$ 个容器的依赖复杂度从 $M 	imes N$ 骤降至 $M + N$。

> [!TIP]
> 🔗 **深度解析**：[iterators-and-stl-containers.md 第 1.1 节](iterators-and-stl-containers.md#11-算法容器四层正交解耦模型)

---

### 2.2 顺序容器物理内存模型

#### std::vector

- **内存拓扑**：单块连续物理堆内存，对象句柄占用 **24 字节**（3 个连续指针：`start`, `finish`, `end_of_storage`）；
- **几何级数扩容**：$k=1.5$（MSVC，内存碎片可复用） vs $k=2.0$（GCC/Clang，快速分配），均摊插入复杂度严格证明为 $O(1)$；
- **小对象优化**：`llvm::SmallVector<T, N>` 内部预留 $N$ 个元素栈上存储，小规模数据 0 堆内存分配。

#### std::deque

- **内存拓扑**：**中控指针数组（Map of Buffers）**管理多块离散连续缓冲区（通常 512 字节/页）；
- **核心优势**：头尾插入/删除均为严格 $O(1)$，且**绝不触发已有数据物理搬迁**；
- **寻址开销**：下标随机访问 `deque[i]` 经历两次解引用（先定位中控 Buffer，再定位 Buffer 内偏移）。

#### std::list

- **内存拓扑**：环形双向链表，每个堆节点承载 16 字节指针开销（`prev` + `next`）；
- **失效特征**：插入/删除绝对不导致其他节点的迭代器与引用失效，但 CPU Cache 局部性极差。

> [!TIP]
> 🔗 **深度解析**：[iterators-and-stl-containers.md 第 2.1 节](iterators-and-stl-containers.md#21-顺序容器内存排布)

---

### 2.3 扩容与删除引发的迭代器失效边界

| 容器类型 | 插入触发扩容 | 插入未触发扩容 | 删除操作 (`erase`) |
| :--- | :---: | :---: | :---: |
| **`std::vector`** | **全部失效** | 插入点前有效，**插入点后全失效** | 被删点前有效，**被删点后全失效** |
| **`std::deque`** | — | **头尾插入**：迭代器失效，**指针引用仍有效**！<br>**中间插入**：全失效 | **头尾删除**：仅被删失效；<br>**中间删除**：全失效 |
| **`std::list`** | — | **全量永久有效** | **仅被删除元素迭代器失效** |
| **`std::map / set`** | — | **全量永久有效** | **仅被删除元素迭代器失效** |
| **`std::unordered_map`** | **Rehash 触发**：**迭代器全失效，指针引用仍有效**！ | 全量有效 | **仅被删除元素迭代器失效** |

**Erase-Remove 惯用法**：

```cpp
// C++11 经典：
vec.erase(std::remove_if(vec.begin(), vec.end(), predicate), vec.end());
// C++20 统一接口：
std::erase_if(vec, predicate);
```

> [!TIP]
> 🔗 **深度解析**：[iterators-and-stl-containers.md 第 3.1 节](iterators-and-stl-containers.md#31-扩容与删除迭代器失效全景)

---

### 2.4 std::move_if_noexcept 强异常安全回滚

```cpp
// vector 扩容时搬迁元素的静态决议：
// 若移动构造标记 noexcept -> 使用 move 零拷贝转移；
// 若未标记 noexcept -> 退化为深拷贝！若中途抛异常，销毁新内存，原数据完好，达成强异常安全！
template <typename T>
void reallocate_and_move(T* new_mem, T* old_mem, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        std::construct_at(new_mem + i, std::move_if_noexcept(old_mem[i]));
    }
}
```

> [!TIP]
> 🔗 **深度解析**：[iterators-and-stl-containers.md 第 3.3 节](iterators-and-stl-containers.md#33-异常安全三级契约与-move_if_noexcept-事务性回滚)

---

## 3. 智能指针与 RAII 资源管理

### 3.1 智能指针物理内存模型对比

```text
std::unique_ptr<T> (8 字节，EBCO 空基类优化下与裸指针零额外开销)
┌─────────────────────────────────────────────────────────────┐
│ T* ptr_ (8 字节裸指针)                                      │
└─────────────────────────────────────────────────────────────┘

std::shared_ptr<T> (16 字节双指针)
┌──────────────────────────────┬──────────────────────────────┐
│ T* ptr_ (指向托管数据对象)    │ ControlBlock* ref_cnt_ (8B)  │
└──────────────────────────────┴──────────────┬───────────────┘
                                              │
                                              ▼
                             ┌────────────────────────────────┐
                             │ atomic<int> strong_ref_count   │ (4 字节)
                             │ atomic<int> weak_ref_count     │ (4 字节)
                             │ Deleter / Allocator (类型擦除) │
                             └────────────────────────────────┘
```

> [!TIP]
> 🔗 **深度解析**：[smart-pointers-and-raii.md 第 2 与 3 章](smart-pointers-and-raii.md#2-unique_ptr-独占所有权模型)

---

### 3.2 std::make_shared 的物理权衡

```text
make_shared 内存布局 (单块连续内存分配):
┌───────────────────────────────────────┬─────────────────────┐
│ 控制块 (Control Block: strong/weak cnt)│ 托管对象 T 数据空间 │
└───────────────────────────────────────┴─────────────────────┘
```

- **核心优势**：将 2 次独立堆内存分配（对象 + 控制块）缩减为 **1 次连续分配**，提升 CPU 缓存局部性；
- **潜在代价**：只要 `weak_ptr` 计数未清零，**整块连续物理内存（包含已析构的 T）均无法释放还给操作系统**。

> [!TIP]
> 🔗 **深度解析**：[smart-pointers-and-raii.md 第 3.4 节](smart-pointers-and-raii.md#34-make_shared-内存排布与权衡)

---

### 3.3 weak_ptr 与无锁 CAS 提升

`weak_ptr` 观察生命周期但不增加强引用计数。将其提升为 `shared_ptr` 通过无锁原子 CAS 实现：

```cpp
// weak_ptr::lock() 底层原子自增机理
long count = strong_count.load(std::memory_order_relaxed);
do {
    if (count == 0) return nullptr; // 对象已被析构，安全判定死亡
} while (!strong_count.compare_exchange_weak(count, count + 1,
                                            std::memory_order_acquire,
                                            std::memory_order_relaxed));
```

> [!TIP]
> 🔗 **深度解析**：[smart-pointers-and-raii.md 第 4.2 节](smart-pointers-and-raii.md#42-无锁-cas-原子提升机制)

---

### 3.4 enable_shared_from_this 机制

```cpp
class Node : public std::enable_shared_from_this<Node> {
public:
    std::shared_ptr<Node> getSelf() {
        return shared_from_this(); // ✔️ 安全复用已有控制块
    }
};
```

- **底层原理**：基类内部持有一个 `weak_ptr<T> weak_this_`，在 `shared_ptr<Node>` 构造期通过模板 SFINAE 自动将控制块地址注入其中；
- **致命陷阱**：严禁在对象构造函数执行期间或在栈对象上调用 `shared_from_this()`（此时控制块尚未建立，直接抛出 `std::bad_weak_ptr`）。

> [!TIP]
> 🔗 **深度解析**：[smart-pointers-and-raii.md 第 5 章](smart-pointers-and-raii.md#5-enable_shared_from_this-安全共享)

---

## 4. 闭包、递归 Lambda 与类型擦除

### 4.1 闭包类的编译器合成机理

```cpp
int bias = 10;
auto add_bias = [bias](int x) { return x + bias; };
```

编译器前端下沉展开为等价仿函数类：

```cpp
class __lambda_unique_id {
    int bias_; // 捕获变量成为类成员
public:
    explicit __lambda_unique_id(int b) : bias_(b) {}
    int operator()(int x) const { return x + bias_; } // 默认 const
};
```

- **无捕获 Lambda**：大小为 1 字节（空类），**可隐式退化为原生 C 函数指针**；
- **`mutable` 说明符**：指示编译器在生成 `operator()` 时不要附加 `const` 限定符。

> [!TIP]
> 🔗 **深度解析**：[lambdas-closures-and-type-erasure.md 第 1 章](lambdas-closures-and-type-erasure.md#1-闭包类生成机制与泛型-lambda-展开)

---

### 4.2 递归 Lambda 的四代演进方案

```cpp
// 1. C++11 std::function (有间接调用与堆分配开销，无法内联)
std::function<void(int)> dfs1 = [&](int u) { if (u) dfs1(u-1); };

// 2. C++14 泛型自传递 (完全零开销，支持深层内联)
auto dfs2 = [](auto&& self, int u) -> void { if (u) self(self, u-1); };
dfs2(dfs2, 10);

// 3. C++17 Y-Combinator (利用 CTAD 包装自传递样板)
auto dfs3 = YCombinator([&](auto&& self, int u) -> void { if (u) self(u-1); });

// 4. C++23 显式对象形参 Deducing this (终极原生语法，零开销)
auto dfs4 = [](this auto&& self, int u) -> void { if (u) self(u-1); };
dfs4(10);
```

> [!TIP]
> 🔗 **深度解析**：[lambdas-closures-and-type-erasure.md 第 3 章](lambdas-closures-and-type-erasure.md#3-递归-lambda-的演进与四大约束解法)

---

### 4.3 std::function 与 llvm::function_ref 物理实现对比

| 特性 | `std::function<Sig>` | `llvm::function_ref<Sig>` |
| :--- | :--- | :--- |
| **句柄尺寸** | **32 字节**（双函数指针 + 16B SBO 缓冲区） | **16 字节**（`void* callable_` + `callback_`） |
| **内存所有权** | **独占拥有闭包生命周期** | **非拥有（纯只读借用视图）** |
| **堆内存分配** | 小闭包 0 次 (SBO) / 大闭包动态分配 | **严格为 0** |
| **生命周期约束** | 支持长期保存、异步任务队列 | **严禁异步持久化**，仅用于同步函数传参 |

> [!TIP]
> 🔗 **深度解析**：[lambdas-closures-and-type-erasure.md 第 4 与 5 章](lambdas-closures-and-type-erasure.md#4-类型擦除与无虚表分派架构)

---

## 5. 模板元编程与类型约束体系

### 5.1 两阶段名字查找与消歧义关键字

- **Phase 1（模板定义期）**：检查非依赖名字（Non-dependent names）与基本语法；
- **Phase 2（模板实例化期）**：检查依赖模板参数的依赖名字（Dependent names）。

**两大消歧义关键字黄金准则**：
1. **`typename`**：依赖名字代表**类型**时必须前缀 `typename T::iterator`，防止被误判为静态变量；
2. **`template`**：依赖名字代表**成员模板**时必须插入 `obj.template method<U>()`，防止 `<` 被误判为小于号。

> [!TIP]
> 🔗 **深度解析**：[templates-sfinae-and-concepts.md 第 1.1 节](templates-sfinae-and-concepts.md#11-两阶段名字查找与消歧义关键字)

---

### 5.2 SFINAE 与 void_t 成员探测

- **SFINAE 定律**：**替换失败并非错误（Substitution Failure Is Not An Error）**。类型推导失败时仅将该重载从候选集剔除，不触发编译报错。

```cpp
// 基于 void_t 探测类型是否具有 serialize() 成员函数
template <typename T, typename = void>
struct has_serialize : std::false_type {};

template <typename T>
struct has_serialize<T, std::void_t<decltype(std::declval<T>().serialize())>> : std::true_type {};
```

> [!TIP]
> 🔗 **深度解析**：[templates-sfinae-and-concepts.md 第 2 与 3 章](templates-sfinae-and-concepts.md#2-sfinae-特化路由与表达式推导)

---

### 5.3 C++20 Concepts 与 requires 四大约束

```cpp
template <typename T>
concept ComplexContainer = requires(T a, typename T::value_type v) {
    // 1. 简单约束：表达式必须合法
    a.clear();
    // 2. 类型约束：必须存在该嵌套类型
    typename T::iterator;
    // 3. 复合约束：返回值必须满足 Concept 且不抛异常
    { a.size() } noexcept -> std::same_as<size_t>;
    // 4. 嵌套约束：对类型属性进一步断言
    requires sizeof(v) <= 64;
};
```

> [!TIP]
> 🔗 **深度解析**：[templates-sfinae-and-concepts.md 第 4.2 节](templates-sfinae-and-concepts.md#42-concepts-语法与-requires-四大约束)

---

## 6. 编译器工具链与 ELF 链接模型

### 6.1 ELF 重定位计算公式

| 重定位类型 | 适用寻址模式 | 核心计算公式 | 物理计算含义 |
| :--- | :--- | :--- | :--- |
| **`R_X86_64_64`** | 64 位绝对寻址 | **$S + A$** | 符号最终绝对虚拟地址 + 加数 |
| **`R_X86_64_32S`** | 32 位符号扩展绝对寻址 | **$S + A$** | 截断为 32 位符号扩展地址 |
| **`R_X86_64_PC32`** | 32 位相对 PC 偏移寻址 | **$S + A - P$** | 目标符号地址 - 重定位项自身物理位置 $P$ |
| **`R_X86_64_PLT32`** | 动态链接延迟绑定跳板 | **$L + A - P$** | PLT 跳板入口地址 $L$ - 当前指令位置 $P$ |

> [!TIP]
> 🔗 **深度解析**：[compiler-toolchain-and-elf-linking.md 第 3.3 节](compiler-toolchain-and-elf-linking.md#33-重定位表与符号地址修补机理)

---

### 6.2 Meyers' Singleton 汇编级原子锁

```cpp
MeyersSingleton& getInstance() {
    static MeyersSingleton instance; // C++11 保证线程安全的惰性初始化
    return instance;
}
```

编译器为局部静态变量生成基于原子守卫变量的内部状态机：

```cpp
if (!__atomic_load(&guard, std::memory_order_acquire)) {
    if (__cxa_guard_acquire(&guard)) { // 内部加互斥锁
        try {
            new (&instance) MeyersSingleton();
            __cxa_guard_release(&guard); // 解锁并置标志位
        } catch (...) {
            __cxa_guard_abort(&guard);
            throw;
        }
    }
}
```

> [!TIP]
> 🔗 **深度解析**：[compiler-toolchain-and-elf-linking.md 第 5.3 节](compiler-toolchain-and-elf-linking.md#53-meyers-单例惰性初始化与并发原子锁)

---

### 6.3 PLT/GOT 动态延迟绑定两阶段执行链

```text
第一次调用 (延迟绑定阶段)：
call foo@plt ──► JMP *foo@got (跳转回 PLT[i]+6) ──► PUSH ID ──► JMP _dl_runtime_resolve
                                                                        │
                                                                        ▼ (解析真实符号地址)
                                                               写入 foo@got (覆盖为真实地址)

后续调用 (零开销直达阶段)：
call foo@plt ──► JMP *foo@got (此时 GOT 已直存目标绝对地址) ──────────► 执行动态库内真正的 foo()！
```

> [!TIP]
> 🔗 **深度解析**：[compiler-toolchain-and-elf-linking.md 第 6.2 节](compiler-toolchain-and-elf-linking.md#62-位置无关代码与-pltgot-延迟绑定)
