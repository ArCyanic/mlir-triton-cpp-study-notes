# STL 容器、迭代器与异常安全

> 本文系统解构 C++ 标准模板库（STL）的**四层抽象设计**、迭代器概念体系（C++20 Iterator Concepts 与 `iterator_traits`）、**顺序与关联容器的底层物理内存排布**、**迭代器/指针/引用失效矩阵**，以及 `std::move_if_noexcept` 在维持**强异常安全保证（Strong Exception Guarantee）**中的核心机制。

---

## 目录

- [1. STL 四层抽象架构](#1-stl-四层抽象架构)
  - [1.1 算法与容器正交解耦](#11-算法与容器正交解耦)
  - [1.2 迭代器能力层级链](#12-迭代器能力层级链)
- [2. 迭代器机制与 Traits 萃取](#2-迭代器机制与-traits-萃取)
  - [2.1 `iterator_traits` 五大关联类型](#21-iterator_traits-五大关联类型)
  - [2.2 代理引用与 `vector<bool>`](#22-代理引用与-vectorbool)
- [3. STL 容器内存布局](#3-stl-容器内存布局)
  - [3.1 `std::vector` 三指针模型](#31-stdvector-三指针模型)
  - [3.2 `std::deque` 分段缓冲区](#32-stddeque-分段缓冲区)
  - [3.3 `std::list` 链表节点](#33-stdlist-链表节点)
  - [3.4 `std::unordered_map` 哈希桶](#34-stdunordered_map-哈希桶)
- [4. 迭代器失效机制](#4-迭代器失效机制)
  - [4.1 连续内存容器失效](#41-连续内存容器失效)
  - [4.2 范围 for 循环修改隐患](#42-范围-for-循环修改隐患)
  - [4.3 哈希容器 rehash 分歧](#43-哈希容器-rehash-分歧)
- [5. 异常安全与扩容保证](#5-异常安全与扩容保证)
  - [5.1 异常安全三大层级](#51-异常安全三大层级)
  - [5.2 `move_if_noexcept` 扩容回滚](#52-move_if_noexcept-扩容回滚)
- [6. 容器特性与失效速查表](#6-容器特性与失效速查表)

---

## 1. STL 四层抽象架构

### 1.1 算法与容器正交解耦

标准模板库（STL）通过四层正交结构，消除了 $M$ 个算法与 $N$ 个容器组合产生的 $M \times N$ 代码膨胀：

```text
┌─────────────────────────────────────────────────────────────┐
│ 1. 算法层 (Algorithms)    : std::sort, std::find, std::copy │
└──────────────────────────────┬──────────────────────────────┘
                               │ 依赖抽象能力 (区间访问)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. 迭代器层 (Iterators)   : 概念 (Concepts) / Traits 萃取   │
└──────────────────────────────┬──────────────────────────────┘
                               │ 映射底层结构 (产出遍历迭代器)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. 容器层 (Containers)    : vector, deque, list, map        │
└──────────────────────────────┬──────────────────────────────┘
                               │ 申请/释放底层存储
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. 内存分配层 (Allocators): std::allocator, std::pmr        │
└─────────────────────────────────────────────────────────────┘
```

* **核心解耦逻辑**：`std::sort` 并不认识 `std::vector`，它只要求迭代器满足 **随机访问能力（Random Access）**；`std::list` 无法调用 `std::sort`，并非因为容器受到特例排斥，而是其迭代器仅具备 **双向访问能力（Bidirectional）**。

---

### 1.2 迭代器能力层级链

在 C++20 之前，能力层级通过 Tag 结构体区分；C++20 起通过标准 Concepts 严格定义：

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

| 迭代器 Concept | 核心操作要求 | 典型代表容器 |
| :--- | :--- | :--- |
| **`std::input_iterator`** | `*it`, `++it` (单遍扫描，不保证多次重入) | `std::istream_iterator` |
| **`std::forward_iterator`** | `*it`, `++it` (可多次安全保存副本并重复遍历) | `std::forward_list`, `std::unordered_set` |
| **`std::bidirectional_iterator`** | `--it` (支持逆向遍历) | `std::list`, `std::set`, `std::map` |
| **`std::random_access_iterator`** | `it + n`, `it[n]`, `it1 - it2`, `<`, `>` | `std::deque` |
| **`std::contiguous_iterator`** | 满足物理连续性，`&(*(it + n)) == (&*it) + n` | `std::vector`, `std::array`, `std::string` |

---

## 2. 迭代器机制与 Traits 萃取

### 2.1 `iterator_traits` 五大关联类型

当泛型算法面对任意模板参数 `Iterator` 时，通过 `std::iterator_traits<Iterator>` 萃取其元信息：

```cpp
template <typename Iterator>
struct iterator_traits {
    using difference_type   = typename Iterator::difference_type;   // 两个迭代器相减的距离类型 (如 ptrdiff_t)
    using value_type        = typename Iterator::value_type;        // 指向元素的原始类型
    using pointer           = typename Iterator::pointer;           // 指针类型 (如 T*)
    using reference         = typename Iterator::reference;         // 解引用返回值类型 (如 T&)
    using iterator_category = typename Iterator::iterator_category; // 迭代器能力标签 Tag
};

// 针对原生指针 T* 的偏特化 (使裸指针无缝融入 STL 算法体系)
template <typename T>
struct iterator_traits<T*> {
    using difference_type   = std::ptrdiff_t;
    using value_type        = T;
    using pointer           = T*;
    using reference         = T&;
    using iterator_category = std::random_access_iterator_tag;
};
```

---

### 2.2 代理引用与 `vector<bool>`

C++ 标准对 `std::vector<bool>` 进行了空间特化（1 bit 存储 1 个 bool），导致其解引用无法返回真实的 `bool&`（硬件无法对单独的 bit 取物理地址）：

```cpp
std::vector<bool> vb = {true, false};
auto ref = vb[0]; // ref 的实际类型是 std::vector<bool>::reference (代理对象)

// ❌ 常见泛型陷阱：
// auto& bad_ref = vb[0]; // 编译报错：无法将临时代理对象绑定到非常量左值引用 bool&
```

> [!WARNING]
> 在编写通用模板算法时，不能假定 `*it` 返回的必定是 `value_type&`。现代 C++20 通过 `std::iter_reference_t<It>` 与 `std::indirectly_readable` 概念标准化了代理引用的处理。

---

## 3. STL 容器内存布局

### 3.1 `std::vector` 三指针模型

在 64 位主流实现（libstdc++ / libc++）中，`std::vector` 对象本身**严格只占 24 字节**，由 3 个指针构成：

```text
std::vector<T> 控制句柄 (栈上 24 字节)
┌──────────────────────┬──────────────────────┬──────────────────────┐
│  T* _M_start (8B)    │  T* _M_finish (8B)   │ T* _M_end_storage(8B)│
└──────────┬───────────┴──────────┬───────────┴──────────┬───────────┘
           │                      │                      │
           ▼                      ▼                      ▼
堆内存:    [ Elem 0 ][ Elem 1 ]...[ Elem N-1 ]           [ 空闲未构造槽位 ... ]
           ▲                                             ▲
           └─────────── size() = finish - start ─────────┘
           └────────────────── capacity() = end_of_storage - start ──────────┘
```

---

### 3.2 `std::deque` 分段缓冲区

`std::deque`（双端队列）避免了 `vector` 单一大块内存扩容的搬迁开销，采用**中控映射表（Map of Buffers）**：

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

* **特性**：
  * 头尾插入和删除具有 $O(1)$ 复杂度，且**绝不触发已有缓冲区块的物理搬迁**；
  * 下标访问 `deque[i]` 需要两次指针解引用（先查中控表定位 Buffer，再查 Buffer 内偏移），常数开销略大于 `vector`。

---

### 3.3 `std::list` 链表节点

```text
std::list 内存布局 (非连续节点分散在堆中)
┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│ Node A       │ ──────► │ Node B       │ ──────► │ Node C       │
│ - prev (8B)  │ ◄────── │ - prev (8B)  │ ◄────── │ - prev (8B)  │
│ - next (8B)  │         │ - next (8B)  │         │ - next (8B)  │
│ - Data (T)   │         │ - Data (T)   │         │ - Data (T)   │
└──────────────┘         └──────────────┘         └──────────────┘
```

* **开销分析**：每个元素额外负担 16 字节（`prev` + `next`）的指针元数据开销，CPU 缓存局部性（Cache Locality）极差。

---

### 3.4 `std::unordered_map` 哈希桶

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

* **Rehash 机制**：当元素数量超过 `bucket_count * max_load_factor` 时，触发 Rehash：开辟更大的桶数组，重新计算所有节点的桶下标并重新挂接链表。

---

## 4. 迭代器失效机制

迭代器失效的本质：**迭代器内部持有的物理内存地址不再对应合法的目标元素**。

### 4.1 连续内存容器失效

```text
扩容前：[ Obj A ][ Obj B ] (地址 0x1000) ◄── it 指向 0x1000
                               │
               (触发 push_back 扩容，申请新堆内存 0x5000，旧内存释放 free)
                               ▼
扩容后：[ Obj A' ][ Obj B' ][ Obj C' ] (新地址 0x5000)
                                    
此时原 it 仍指向已被释放的 0x1000 ──► 野指针 (Dangling Pointer)！
```

* **扩容插入（`size == capacity` 时 `push_back`/`insert`）**：**所有迭代器、指针、引用全部失效**；
* **非扩容插入（`size < capacity` 时在中间 `insert`）**：插入点之前的迭代器有效，**插入点及其之后的所有迭代器/引用失效**（因元素后移）。

---

### 4.2 范围 for 循环修改隐患

```cpp
// ❌ 极高频的经典未定义行为：
std::vector<int> vec = {1, 2, 3};
for (auto x : vec) {
    if (x == 2) {
        vec.push_back(100); // 💥 若触发扩容，循环内部隐式缓存的 end 迭代器与遍历指针全部失效！
    }
}
```

> [!CAUTION]
> 范围 `for` 循环在进入循环时一次性缓存 `auto __begin = vec.begin(); auto __end = vec.end();`。循环体内对容器结构的修改会导致该遍历逻辑直接崩溃。

---

### 4.3 哈希容器 rehash 分歧

当 `std::unordered_map` 发生 Rehash 时：
* **迭代器（Iterators）**：**全部失效**（因为桶数组重建，遍历桶的游标完全改变）；
* **指针与引用（Pointers & References）**：**依然有效**！因为节点实体仍然留在原来的堆地址上，仅仅是改变了桶头指针的链表指向。

---

## 5. 异常安全与扩容保证

### 5.1 异常安全三大层级

1. **基本异常保证（Basic Guarantee）**：抛出异常后，无内存泄漏，容器仍处于合法但未指定（Valid but Unspecified）的状态。
2. **强异常安全保证（Strong Guarantee / Commit-or-rollback）**：**操作具备事务性**。若抛出异常，整个容器状态完全回滚到操作调用之前的原样。
3. **不抛异常保证（Nothrow Guarantee）**：操作绝不抛出任何异常（标记为 `noexcept`）。

---

### 5.2 `move_if_noexcept` 扩容回滚

为了满足 `vector::push_back` 的**强异常安全保证**，标准库在扩容搬迁元素时必须遵循严格的决策树：

```text
                        元素扩容搬迁决策 (std::move_if_noexcept)
                                           │
             ┌─────────────────────────────┴─────────────────────────────┐
             ▼                                                           ▼
【类型移动构造声明为 noexcept】                                【类型移动构造未标 noexcept 且可拷贝】
             │                                                           │
             ▼                                                           ▼
【采用移动构造搬迁 (Move)】                                    【强制退化为全量深拷贝 (Copy)】
• 0 次堆内存重新分配                                            • 若中途第 i 个对象拷贝抛异常，旧内存毫发无损
• 纳秒级指针转移                                                • 直接销毁新内存，容器安全回滚，保证强异常安全！
```

#### 标准库底层判定逻辑：
```cpp
template <typename T>
using move_if_noexcept_t = std::conditional_t<
    !std::is_nothrow_move_constructible_v<T> && std::is_copy_constructible_v<T>,
    const T&, // 强制走拷贝构造
    T&&       // 走移动构造
>;
```

> [!IMPORTANT]
> **工业级性能准则**：自定义类型若定义了移动构造函数，**必须显式添加 `noexcept` 修饰符**，否则在存入 `std::vector` 等容器后，扩容操作会完全丧失移动语义带来的性能优势。

---

## 6. 容器特性与失效速查表

| 容器 | 迭代器能力 | 物理存储模型 | 随机访问复杂度 | 典型插入/删除复杂度 | 结构调整失效规则 |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **`std::vector`** | Contiguous | 单块连续堆内存 | $O(1)$ | 尾部均摊 $O(1)$，中间 $O(N)$ | 扩容则全失效；未扩容则插入点之后失效 |
| **`std::deque`** | Random Access | 分段中控缓冲区 | $O(1)$ | 头尾 $O(1)$，中间 $O(N)$ | 头尾插入迭代器失效但指针有效；中间修改全失效 |
| **`std::list`** | Bidirectional | 独立双向堆节点 | ❌ 仅线性遍历 | 已知位置 $O(1)$ | 仅被删除节点失效，其他节点迭代器/指针永不失效 |
| **`std::set` / `map`** | Bidirectional | 红黑树平衡节点 | ❌ $O(\log N)$ | 增删查 $O(\log N)$ | 插入永不失效；删除仅被删节点失效 |
| **`std::unordered_map`** | Forward | 哈希桶 + 链表节点 | ❌ 平均 $O(1)$ | 平均 $O(1)$，最坏 $O(N)$ | Rehash 导致迭代器全失效，但节点引用/指针保持有效 |
