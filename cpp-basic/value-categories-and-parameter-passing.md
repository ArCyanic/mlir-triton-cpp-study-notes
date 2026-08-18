# 值类别与参数传递模型

> 本文以**编译器 IR 节点（`Operation` 与 `Value`）的设计与构建**为单一贯穿场景，系统性拆解现代 C++ 的**五大值类别分类体系（Value Categories）**、**移动语义底层机理与移后状态契约**、**C++17 纯右值物化与 NRVO 栈帧 ABI 优化**、**万能引用折叠与完美转发贪婪构造陷阱**，并系统推导**六乘五形参实参绑定决议矩阵、零拷贝非拥有视图（`std::string_view` / `llvm::ArrayRef`）与工程传参选型决策全景**。

---

## 1. 算子实体传参场景建模与核心设计考量

在编译器中间表示（IR）的基础设施开发中，核心任务之一是构建高效、健壮的 `Operation`（算子节点）类。该类负责管理算子的唯一标识 ID、算子操作名称、输入操作数 ID 列表以及输出结果 ID 列表：

```cpp
class Operation {
public:
    // 核心设计决策：构造函数的参数应当如何声明与传递？
    // 是传 const 引用、传值 + std::move、传右值引用，还是采用模板万能引用完美转发？
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

调用端在生成与构建 IR 节点时，存在三种典型的物理上下文：

- **场景 A（传入具名变量 / 活跃左值）**：外部作用域维护了一个 `std::vector<int> op_list`，在构造当前 `Operation` 后，后续控制流中仍需继续读取该列表；
- **场景 B（传入临时字面量 / 纯右值）**：调用方直接传入 `{1, 2}` 列表初始化字面量，或传入即时计算生成的临时对象 `std::vector<int>{v_in, t_in}`；
- **场景 C（主动移交所有权 / 亡值）**：调用方持有一个局部容器 `temp_op_list`，在构造算子后该局部变量不再使用，显式调用 `std::move(temp_op_list)` 移交资源。

参数传递模型的设计优劣，直接决定了编译器在构建数以百万计的 IR 节点时，触发的是极其沉重的**系统堆内存深拷贝（Deep Copy）**，还是仅需几个 CPU 周期的**常数时间指针所有权移交（Move）或只读借用（View）**。

---

## 2. 表达式五大值类别体系

### 2.1 五大值类别分类体系

在传统 C 语言中，值类别仅仅依据表达式能否出现在赋值号（`=`）左侧粗略划分为左值（lvalue）与右值（rvalue）。C++11 引入移动语义后，标准委员会基于**表达式的物理属性**确立了严密的**五大值类别分类体系树**：

```text
                  表达式 (Expressions)
                  /                  \
      泛左值 (glvalue)              右值 (rvalue)
         /          \              /          \
    左值 (lvalue)    亡值 (xvalue)    纯右值 (prvalue)
```

---

### 2.2 身份与可移动性正交判定准则

现代 C++ 依据两个正交的物理属性维度精确界定任意表达式的值类别：

1. **是否具备身份标识（Has Identity）**：表达式是否占据明确且持久的物理内存地址？能否通过取地址运算符（`&expr`）合法获取其物理指针？
2. **是否允许被安全移动（Can be Moved from）**：表达式所绑定的资源生命周期是否即将终结？是否允许在保证析构安全的前提下直接窃取其内部堆资源？

依据这两个正交维度的判定组合，划分出 3 种独立基础类别与 2 种复合类别：

| 基础值类别 | 具有身份标识 (Identity)？ | 允许被安全移动 (Movable)？ | 核心语义契约与物理判定标准 |
| :--- | :---: | :---: | :--- |
| **`lvalue` (左值)** | **✔️ 有** | ❌ 否 | 拥有明确具名内存地址的对象，生命周期跨越当前单条语句，不可被随意篡改或窃取。 |
| **`xvalue` (亡值 / eXpiring value)** | **✔️ 有** | **✔️ 是** | 拥有明确内存地址，但生命周期即将终结，显式允许被转移（如 `std::move(x)`）。 |
| **`prvalue` (纯右值 / Pure rvalue)** | ❌ 无 | **✔️ 是** | 运算产生的临时中间结果、字面量（在 C++17 中被定义为对象的初始化处方）。 |

- **泛左值（`glvalue = lvalue + xvalue`）**：所有具备物理内存身份标识的表达式总称，其求值结果确定了某个对象的物理位置；
- **右值（`rvalue = xvalue + prvalue`）**：所有允许被窃取其底层资源的表达式总称，可被移动构造函数或右值引用绑定。

---

### 2.3 典型表达式值类别判定与推导

```cpp
int a = 10;
int b = 20;

a;                      // lvalue (具名变量，拥有持久内存地址，可执行 &a)
a + b;                  // prvalue (临时算术运算结果，无持久地址)
100;                    // prvalue (整型字面量)
std::string("conv");    // prvalue (临时构造的对象处方)
std::move(a);           // xvalue (具有地址 &a，但类型被静态转换为右值引用，显式标记为将亡)
static_cast<int&&>(a);  // xvalue (静态类型转换为右值引用)
```

---

## 3. 移动语义底层机理与拷贝消除

### 3.1 std::move 零开销静态转型与具名左值法则

#### 零开销静态类型转换

`std::move` 在运行时**绝不搬运任何内存数据，也不产生任何一条 CPU 机器指令**。其在 `<utility>` 中的标准库实现本质上是一个编译期静态类型转换器：

```cpp
template <typename T>
constexpr std::remove_reference_t<T>&& move(T&& t) noexcept {
    // 强制静态转换为无引用的右值引用类型 (xvalue)
    return static_cast<std::remove_reference_t<T>&&>(t);
}
```

`std::move` 的唯一职责是**在编译器类型系统中将一个左值强行转为亡值（`xvalue`）**，从而在重载决议中驱动编译器优先命中移动构造函数或移动赋值运算符。

#### 具名左值黄金法则

在 C++ 中，**凡是拥有明确名称的变量表达式，在当前作用域内一律被严格判定为左值**：

```cpp
void consumeBuffer(std::vector<int>&& rvalue_ref) {
    // 陷阱：在函数体内，形参名 rvalue_ref 本身是左值还是右值？
    // operands_ = rvalue_ref;            // ❌ 错误：具名变量是左值，触发全量深拷贝！
    operands_ = std::move(rvalue_ref);    // ✔️ 正确：显式再次转为亡值，触发指针所有权移动！
}
```

形参 `rvalue_ref` 的类型是右值引用 `std::vector<int>&&`，但其表达式本身是一个**具名变量**，拥有确定的栈内存地址。为了防止后续代码读取该变量时发生非法访问，编译器默认将其作为左值对待。若要在函数体内将其资源继续向下传递，必须显式调用 `std::move`。

---

### 3.2 容器指针所有权转移与移后状态契约

以 `std::vector` 的移动构造函数为例，底层仅涉及 3 个 8 字节指针的轻量赋值，完全免除了堆内存分配与数据拷贝：

```cpp
vector(vector&& other) noexcept {
    // 1. 窃取源对象的堆内存首尾与容量指针 (3 条 mov 寄存器指令，< 5 纳秒)
    this->data_start_ = other.data_start_;
    this->data_end_   = other.data_end_;
    this->capacity_end_ = other.capacity_end_;

    // 2. 将源对象指针清空，防止其析构时触发 double-free
    other.data_start_ = nullptr;
    other.data_end_   = nullptr;
    other.capacity_end_ = nullptr;
}
```

```text
移动前：
[other]   ─── ptr ───► Heap Buffer [0x1000 ... 0x2000] (包含 100 万个浮点数)
[this]    ─── ptr ───► nullptr

移动后 (常数时间 O(1) 指针窃取)：
[other]   ─── ptr ───► nullptr (变为空容器，析构时无任何物理内存释放)
[this]    ─── ptr ───► Heap Buffer [0x1000 ... 0x2000] (完整接管堆内存)
```

#### 移后对象的标准规范契约

C++ 标准明确规定：**被移动后的对象必须处于“有效但未指定的状态（Valid but Unspecified State）”**：
1. **析构安全性**：移后对象在离开作用域调用析构函数时，绝不能引发崩溃或重复释放；
2. **可重用性**：移后对象必须能够安全地接收新的赋值（如 `other = std::vector<int>{1, 2}`）或调用无前置条件限制的成员函数（如 `other.clear()` / `other.empty()`）；
3. **自移动安全防御**：移动赋值运算符必须具备自赋值防御（`if (this != &other)`），防止 `a = std::move(a)` 意外销毁自身持有的物理堆内存。

#### 常量对象移动退化陷阱

若在具名对象前修饰了 `const`，随后对其执行 `std::move`：

```cpp
const std::vector<int> frozen_operands = {1, 2, 3};
Operation op(1, "Conv", std::move(frozen_operands), {}); // 传入 std::move(const)
```

1. `std::move(frozen_operands)` 产生的静态类型是 `const std::vector<int>&&`（常量右值引用）；
2. `std::vector` 的移动构造函数签名是 `vector(vector&& other)`（非常量右值引用），参数类型不兼容，拒绝绑定；
3. 重载决议退而求其次，命中拷贝构造函数 `vector(const vector& other)`，**静默触发昂贵的全量深拷贝**。

> [!WARNING]
> 准备通过 `std::move` 转移所有权的对象，**严禁修饰 `const`**。任何对 `const` 对象的 `std::move` 都会无声无息地退化为全量内存深拷贝。

---

### 3.3 保证拷贝消除与返回值优化

#### C++17 纯右值物化机制

在 C++17 标准中，**纯右值（Prvalue）被重新定义为对象的“初始化处方”，而非实体对象本身**。只有当 Prvalue 被用来初始化一个具备物理位置的变量时，对象才会在该目标地址上**就地物化构造（Materialization）**：

```cpp
std::vector<int> createOperands() {
    return std::vector<int>{1, 2, 3, 4}; // Prvalue 初始化处方
}

// C++17 保证拷贝消除：0 次拷贝，0 次移动！直接在外部变量 op 的物理内存位置就地构造！
std::vector<int> op = createOperands();
```

#### NRVO 具名返回值优化与 ABI 隐藏指针

当函数返回一个局部具名变量时，现代编译器通过 **NRVO（Named Return Value Optimization）** 消除对象的中间拷贝与移动。

在底层 System V AMD64 ABI 规范中，当函数返回大型对象（如 `sizeof > 16` 字节）时，调用方会在自己的栈帧上预留这块内存，并将其地址作为一个**隐藏的首参数（Hidden Return Buffer Pointer，通过 `rdi` 寄存器）**隐式传递给被调函数。被调函数直接在该指针指向的内存区域就地执行构造，实现真正的 **0 拷贝、0 移动**。

```cpp
// ❌ 极度糟糕的破坏优化写法：
std::vector<int> buildNodeList() {
    std::vector<int> local_list = {1, 2, 3};
    return std::move(local_list); // 💥 强行破坏了 NRVO 优化！
}

// ✔️ 标准现代写法：
std::vector<int> buildNodeList() {
    std::vector<int> local_list = {1, 2, 3};
    return local_list; // 触发 NRVO，编译器直接在调用方提供的栈缓冲区中就地构造！
}
```

在 `return` 语句处显式添加 `std::move(local_list)` 会强制将局部变量转换为 `xvalue`，迫使编译器放弃 NRVO 隐藏指针就地构造，退化为执行一次移动构造函数，反而增加了寄存器传递与指针赋值开销。

---

## 4. 万能引用与完美转发机制

### 4.1 模板万能引用与引用折叠规则

当 `&&` 出现在**需要发生类型推导的模板参数上下文**中时，该引用被赋予了特殊的语义——**万能引用（Universal Reference / Forwarding Reference）**：

```cpp
template <typename T>
void makeNode(T&& arg); // T&& 是万能引用，能够同等兼容接收左值与右值实参
```

在模板实例化过程中，若实参类型带有引用，编译器依据标准的**引用折叠规则（Reference Collapsing Rules）**合成最终形参类型：

| 实参传入形态 | 推导出的模板形参 `T` | 引用折叠推导过程 | 最终形参绑定类型 |
| :---: | :---: | :---: | :---: |
| **非常量左值** `var` | `Type&` | `Type& + &&` $\to$ **`Type&`** | **左值引用**（精确接收左值） |
| **常量左值** `const_var` | `const Type&` | `const Type& + &&` $\to$ **`const Type&`** | **常量左值引用**（只读接收） |
| **纯右值** `Type{}` | `Type` | `Type + &&` $\to$ **`Type&&`** | **右值引用**（精确接收右值） |
| **亡值** `std::move(var)`| `Type` | `Type + &&` $\to$ **`Type&&`** | **右值引用**（精确接收右值） |

**引用折叠核心定理**：
> **只要有左值引用 `&` 参与折叠，结果一律坍缩为左值引用 `&`**；当且仅当两侧均为右值引用 `&&` 时，结果才折叠为右值引用 `&&`。

---

### 4.2 std::forward 静态类型还原机制

在万能引用函数体内，由于形参 `arg` 具备名字，其表达式直接退化为左值。若要将其按最初传入时的真实值类别向下游函数传递，必须借助 `std::forward<T>`：

```cpp
template <typename T>
void makeNode(T&& arg) {
    // 依赖推导出的模板实参 T 触发类型还原
    constructInternal(std::forward<T>(arg));
}
```

```cpp
// std::forward 标准库实现
template <typename T>
constexpr T&& forward(std::remove_reference_t<T>& t) noexcept {
    return static_cast<T&&>(t);
}
```

- **当外部传入左值时**：`T` 被推导为 `std::string&`，`static_cast<std::string& &&>(t)` 折叠为 `std::string&`，向下游传递左值引用（下游触发深拷贝）；
- **当外部传入右值时**：`T` 被推导为 `std::string`，`static_cast<std::string&&>(t)`，向下游传递右值引用（下游触发零拷贝移动）。

---

### 4.3 构造函数贪婪捕获陷阱与约束防御

#### 万能引用贪婪劫持成因

在设计包含完美转发构造函数的类时，存在一个极其危险的 C++ 陷阱——**万能引用构造函数贪婪捕获拷贝构造**：

```cpp
class OperationNode {
public:
    // 万能引用构造函数
    template <typename T>
    explicit OperationNode(T&& name) : name_(std::forward<T>(name)) {}

    // 默认生成的拷贝构造函数
    OperationNode(const OperationNode& other) = default;

private:
    std::string name_;
};

int main() {
    OperationNode node1("AddOp");
    
    // ❌ 编译报错：无法将 OperationNode 转换为 std::string！
    // OperationNode node2(node1); 
}
```

**陷阱根因**：
当执行 `OperationNode node2(node1);` 时，传入的 `node1` 是一个非常量左值（类型为 `OperationNode&`）。
- 候选 A（拷贝构造）：`OperationNode(const OperationNode&)` 需要附加一次 `const` 权限修饰转换；
- 候选 B（万能引用模板）：`T` 被直接推导为 `OperationNode&`，生成精确签名的构造函数 `OperationNode(OperationNode&)`。
依据 C++ 重载决议规则，**精确匹配优先于类型转换**，万能引用模板以更高优先级被贪婪命中，进而尝试用 `node1` 去初始化 `std::string name_`，导致编译失败！

#### 编译期约束防御方案

必须使用 SFINAE 或 C++20 Concepts 显式约束万能引用构造函数，排斥当前类本身及其派生类：

```cpp
// C++11/14 SFINAE 防御：
template <typename T, typename = std::enable_if_t<
    !std::is_same_v<std::decay_t<T>, OperationNode>
>>
explicit OperationNode(T&& name) : name_(std::forward<T>(name)) {}

// C++20 Concepts 现代优雅防御：
template <typename T>
    requires (!std::same_as<std::remove_cvref_t<T>, OperationNode>)
explicit OperationNode(T&& name) : name_(std::forward<T>(name)) {}
```

---

## 5. 形参决议矩阵与工程选型准则

### 5.1 六乘五形参实参绑定决议矩阵

#### 形参实参决议网格

由常量修饰符与引用类型正交组合生成的 6 种形参类型，与 5 种实参形态之间的完整绑定与重载决议矩阵如下：

| 实参形态（调用端传入） | 1. `T`<br>(传值) | 2. `const T`<br>(只读传值) | 3. `T&`<br>(可写左值引) | 4. `const T&`<br>(只读左值引) | 5. `T&&`<br>(可写右值引) | 6. `const T&&`<br>(只读右值引) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **① 非常量左值**<br>`T a;` | ✔️ (拷贝) | ✔️ (拷贝) | **✔️ 精确匹配** | ✔️ (只读绑定) | ❌ 拒绝左值 | ❌ 拒绝左值 |
| **② 常量左值**<br>`const T ca;` | ✔️ (拷贝) | ✔️ (拷贝) | ❌ **拒绝 (权限放大)** | **✔️ 精确匹配** | ❌ 拒绝左值 | ❌ 拒绝左值 |
| **③ 纯右值**<br>`T()` 或 字面量 | ✔️ (移动) | ✔️ (移动) | ❌ **拒绝 (右值不可写)**| ✔️ (延长生命期) | **✔️ 精确匹配** | ✔️ (只读绑定) |
| **④ 非常量亡值**<br>`std::move(a)` | ✔️ (移动) | ✔️ (移动) | ❌ **拒绝 (右值不可写)**| ✔️ (只读绑定) | **✔️ 精确匹配** | ✔️ (只读绑定) |
| **⑤ 常量亡值**<br>`std::move(ca)` | ✔️ (退化拷贝) | ✔️ (退化拷贝) | ❌ **拒绝 (权限放大)** | ✔️ (只读绑定) | ❌ **拒绝 (权限放大)** | **✔️ 精确匹配** |

#### 四大引用绑定核心契约

1. **权限收紧契约（Const-correctness）**：`const` 修饰的实参绝不允许绑定到非常量引用（`T&` 和 `T&&`），从根本上保证了只读内存的物理不可变性；
2. **意图显式区分**：`T&` 明确表达就地修改意图；`T&&` 明确表达资源窃取转移意图，二者天然互斥，右值引用拒绝绑定左值以保护具名变量不被意外掏空；
3. **常量左值引用的右值延寿**：`const T&` 能够接收右值，编译器会自动将该右值临时对象的生命周期延长至当前引用的作用域结束；
4. **常量右值引用的语义矛盾**：`const T&&` 既拒绝左值，又禁止修改右值（无法调用移动构造），导致其只能触发全量深拷贝，在实际工程中仅用于 `= delete` 禁用特定的右值重载版本。

---

### 5.2 经典传参范式与重载决议优先级

#### Sink 模式与传值优化

对于需要**接管外部资源所有权并保存在类成员变量中**的构造函数与 Setter 方法，采用 **Sink Parameter（传值 + `std::move`）** 模式是最优工程实践：

```cpp
class Operation {
public:
    // Sink 模式：operands 在形参栈帧上作为一个中转实体
    void setOperands(std::vector<int> operands) {
        operands_ = std::move(operands);
    }
};
```

- **传入左值实参**：在形参处触发 1 次拷贝构造，在函数体内通过 `move` 赋值给成员 $\implies$ **1 次拷贝 + 1 次移动**；
- **传入右值实参**：在形参处触发 1 次移动构造，在函数体内通过 `move` 赋值给成员 $\implies$ **0 次拷贝 + 2 次移动**（全流程零堆内存分配！）；
- **相比传统双重载的优势**：若采用传统的 `setOperands(const vector&)` 与 `setOperands(vector&&)` 重载，当构造函数拥有 $N$ 个容器参数时，需要编写 $2^N$ 个组合重载；而 Sink 模式仅需 1 个统一函数即可达到最优性能。

#### 精确匹配重载决议优先级

当多个重载函数同时出现时，编译器依据精确匹配优先原则进行决议：

```cpp
void process(const std::vector<int>& v); // 重载 A (const&)
void process(std::vector<int>&& v);      // 重载 B (&&)
```

- **传入左值 `std::vector<int> x; process(x);`**：命中**重载 A**（左值无法绑定到右值引用 `&&`，`const&` 是唯一合法候选）；
- **传入纯右值 `process(std::vector<int>{1, 2});`**：命中**重载 B**（右值引用 `&&` 对右值是**精确匹配（Exact Match）**，优于需附加 const 转换的 `const&`）；
- **传入亡值 `process(std::move(x));`**：命中**重载 B**（`std::move(x)` 产出 `xvalue`，精确匹配右值引用 `&&`）。

---

### 5.3 现代零拷贝视图与参数决策树

在现代高性能系统与编译器代码库（如 MLIR / LLVM）中，对于**纯只读观察参数**，广泛采用**轻量切片视图（Non-owning View）**替代传统的 `const std::string&` 与 `const std::vector<T>&`：

- **`std::string_view`**：固定占用 16 字节（`const char* ptr` + `size_t len`），直接传值开销极小，且能无缝兼容 `std::string`、字面量 `"conv"` 与字符数组，彻底消除临时字符串构造；
- **`llvm::ArrayRef<T>`**：固定占用 16 字节（`const T* data` + `size_t length`），能以零拷贝视图统一接收 `std::vector<T>`、`std::array<T, N>`、C 数组以及 `{1, 2, 3}` 初始化列表。

```text
                               现代 C++ 参数传递工程决策树
                                             │
         ┌───────────────────────────────────┴───────────────────────────────────┐
         ▼                                                                       ▼
【函数内部仅作只读观察，不接管生命周期】                                 【函数需要接管并持久化保存该对象】
         │                                                                       │
     ┌───┴───────────────────┐                                       ┌───────────┴───────────┐
     ▼                       ▼                                       ▼                       ▼
[基础原生标量类型]     [复合对象/字符串/连续容器]                           [业务类实体/算子节点]   [通用模板库/工厂函数]
(int, float, 指针等)   (string, vector 等)                           (如 Operation 构造)    (如 make_unique)
     │                       │                                       │                       │
     ▼                       ▼                                       ▼                       ▼
【直接传值 (Pass-by-value)】 【优先使用零拷贝视图】                      【Sink 传值 + move】    【万能引用 + forward】
 (单寄存器传递，纳秒级)        (std::string_view / llvm::ArrayRef)    (std::string + move)    (Perfect Forwarding)
                               [若必须使用标准库容器 ──► 传 const T&]
```

| 传参技术方案 | 函数签名规范 | 左值实参运行时开销 | 右值实参运行时开销 | 核心适用场景 |
| :--- | :--- | :---: | :---: | :--- |
| **直接传值 (Pass-by-value)** | `void fn(int x, float y)` | 单指令寄存器传递 | 单指令寄存器传递 | 原生标量类型、迭代器、轻量轻量句柄（`sizeof <= 16`）。 |
| **零拷贝非拥有视图** | `void fn(std::string_view sv)`<br>`void fn(llvm::ArrayRef<int> arr)` | **严格为 0**（仅传 16B 视图） | **严格为 0**（仅传 16B 视图） | **编译器只读函数参数首选**（统一兼容各类连续容器）。 |
| **常量左值引用 (const T&)** | `void fn(const BigObject& obj)` | 零拷贝（传 8B 裸指针） | 零拷贝（延长临时生命期） | 结构庞大、非连续内存或不可移动的只读对象。 |
| **Sink 模式 (传值 + move)** | `void fn(std::string s)`<br>`void fn(std::vector<int> v)` | 1 拷贝 + 1 移动 | 0 拷贝 + 2 移动 | **构造函数、Setter、数据下沉注入接口**。 |
| **非常量左值引用 (T&)** | `void fn(std::vector<int>& out)` | 零拷贝（直接操作原对象） | ❌ 拒绝右值（编译拦截） | 出参（Out Parameters）或需就地修改源对象状态。 |
| **右值引用 (T&&)** | `void fn(std::vector<int>&& x)` | ❌ 拒绝左值（编译拦截） | 0 拷贝 + 1 移动 | 显式要求调用方放弃所有权的专用移动优化重载。 |
| **万能引用 + 完美转发** | `template <typename T>`<br>`void fn(T&& x)` | 0 拷贝（转发左值） | 0 拷贝（转发右值） | 通用泛型容器（`emplace_back`）、对象构造工厂函数。 |
