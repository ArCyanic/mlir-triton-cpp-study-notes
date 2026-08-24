# Triton LinearLayout 代数体系与实战指南

本文档阐述 Triton 编译器中基于 $\text{GF}(2)$ 有限域的 `LinearLayout`（线性布局）代数体系。内容基于 Triton 官方源码、设计提案与形式化数学定义，推演布局建模、代数变换与硬件排布的生成机制。

---

## 1. 硬件映射与代数建模

在 GPU 与专用加速器体系结构中，张量计算的吞吐受限于存储层级之间的数据搬运效率。如何将多维逻辑张量（Logical Tensor）映射到分层的物理硬件（Thread / Warp / CTA / Register / Shared Memory），是高性能编译器需要解决的关键问题。

在 Triton 3.0 之前，编译器为每种硬件访问模式设计了专用的 C++ 布局属性。随着硬件架构的演进，离散特化的设计增加了维护与转换的复杂度。`LinearLayout` 的引入重构了 Triton 的底层布局模型，将离散特例统一抽象为基于有限域的线性代数系统。

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              Triton 布局系统架构演进                                   │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 早期特化架构 (Triton < 3.0)                                                            │
│   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐                     │
│   │ BlockedEncoding  │  │  SliceEncoding   │  │ DotOpEncoding    │ ... 规则发散        │
│   └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘                     │
│            └─────────────────────┼─────────────────────┘                               │
│                                  ▼                                                     │
│                  针对每对布局手写 convert_layout 转换逻辑                              │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 现代化代数架构 (Triton >= 3.0, LinearLayout 体系)                                      │
│                                                                                        │
│   ┌────────────────────────────────────────────────────────────────────────────────┐   │
│   │                     统一数学对象: class LinearLayout                           │   │
│   │   - 映射函数: L: Hardware Location (t, w, b, reg) ↦ Tensor Coordinate (d0, d1)│   │
│   │   - 核心基石: GF(2) 二进制有限域 + 线性代数 (加法定义为 XOR ⊕)                 │   │
│   └───────────────────────────────────────┬────────────────────────────────────────┘   │
│                                           ▼                                            │
│            统一矩阵运算 (Compose / Invert / Sublayout) 自动生成机器指令                │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1.1 系统物理空间与张量几何的映射矛盾
在 GPU 体系结构中，计算资源划分为四级离散层次：
* **`register`（寄存器）**：单个硬件线程内部持有的向量元素集合；
* **`lane`（线程）**：Warp 内部并行执行单指令流的 32 个物理线程；
* **`warp`（线程束）**：CTA / Block 内部由硬件并发调度的线程束集合；
* **`block`（线程块）**：在流式多处理器（SM）或计算网格上分配的独立线程块。

与离散硬件层级相对的是，算法层面的张量是连续的多维几何空间（例如一个 $M \times N$ 的 2D 矩阵）。为了在硬件上执行张量算子，编译器必须给出**物理硬件位置元组 `(reg, lane, warp, block)` 与张量逻辑多维坐标 `(dim0, dim1, ...)` 之间的映射**。

这种映射不仅要覆盖全图，还必须满足硬件物理约束：
1. **合并访存约束（Global Coalescing）**：同一 Warp 内相邻的 32 个 `lane`，应当访问全局内存中连续对齐的数据块；
2. **片上无冲突约束（Shared Memory Bank Conflict Avoidance）**：同一 Warp 访问共享内存时，32 个线程的物理地址应当分散到不同的 Memory Bank 中；
3. **张量核心对齐约束（Tensor Core Alignment）**：执行矩阵乘法（MMA）时，数据必须符合 Tensor Core 固化的寄存器与共享内存排布格式。

### 1.2 传统专用布局的扩展瓶颈
在引入 `LinearLayout` 之前，Triton 采用了面向特定硬件特性的特化属性集合：
* **`#ttg.blocked`**：用于描述规则的网格分块；
* **`#ttg.slice`**：用于描述张量切片降维；
* **`#ttg.dot_op`**：用于描述供 MMA 消费的寄存器排布；
* **`#ttg.shared`**：用于描述规整的共享内存分配；
* **针对特定架构的专有属性**：如 NVIDIA Hopper MMAv3 的 Swizzled Shared 属性、AMD CDNA 的 MFMA 属性等。

当一个算子的输入输出持有不同布局时，编译器必须插入 `triton_gpu.convert_layout` 算子。在特化架构下，每新增一种布局，编译器就需要为该布局与已有布局之间编写转换规则。

当遇到变维、转置与重排组合（如 `Reshape + Transpose + Reshape`）时，手写转换逻辑难以穷举所有边界条件，增加了编译器的维护成本。

### 1.3 线性函数建模
为了解决特化属性在组合转换时的扩展性问题，Triton 将布局建模为确定性的数学函数：

$$
L: \text{Hardware Location} \mapsto \text{Logical Tensor Index}
$$

该体系选择 **$\text{GF}(2) = \mathbb{F}_2$（二元有限域）** 作为数学载体，原因在于：
1. **硬件寻址的二进制特征**：GPU 中的线程 ID、Warp ID、寄存器编号与内存地址，在物理上均由二进制位表示；
2. **位异或交织的内生表达**：为了消除 Bank 冲突，硬件常采用行号与列号的位异或（例如 $\text{Bank} = \text{col} \oplus \text{row}$）。在 $\text{GF}(2)$ 线性代数中，异或操作即为域上的标准加法。

在 $\text{GF}(2)$ 下，分块、步进、转置、广播以及 Bank Swizzle 变换，均可统一表示为布尔矩阵。

### 1.4 全局概念体系与核心构件
`LinearLayout` 体系由以下四个构件组成：

* **具名维度（Named Dimensions）**：输入空间（如 `lane`, `warp`, `block`, `register`）与输出空间（如 `dim0`, `dim1`）均显式具名，解耦物理硬件与逻辑张量；
* **基底向量（Basis Vectors / Bases）**：每个输入二进制位在输出空间中的映射向量，决定了布局的具体行为；
* **代数运算引擎（Algebraic Engine）**：基于 $\text{GF}(2)$ 矩阵论实现组合（`compose`）、求逆（`invert`）、乘法（`multiply`）与子空间截取（`sublayout`）；
* **代码生成发射器（Emitter / `apply`）**：在后端将矩阵乘法展开为位移（Shift）、掩码（AND）与异或（XOR）指令。

全篇文档的推演主线，围绕“利用基底定义硬件布局 $\rightarrow$ 利用矩阵代数推导自动转换 $\rightarrow$ 在 MLIR 中生成目标机器指令”展开。

---

## 2. GF(2) 线性代数机理

本章推导 `LinearLayout` 的底层数学机理，建立有限域上的向量空间、布尔矩阵乘法与完备的代数闭包算子体系。

### 2.1 二进制异或域与基底向量本质
有限域 $\text{GF}(2)$ 包含两个元素 $\{0, 1\}$，其代数运算法则如下：
* **加法（$\oplus$）**：$0 \oplus 0 = 0,\; 0 \oplus 1 = 1,\; 1 \oplus 0 = 1,\; 1 \oplus 1 = 0$（按位异或 XOR）；
* **乘法（$\cdot$）**：$0 \cdot 0 = 0,\; 0 \cdot 1 = 0,\; 1 \cdot 0 = 0,\; 1 \cdot 1 = 1$（按位与 AND）。

在线性空间中，映射 $L$ 满足线性律：

$$
L(x \oplus y) = L(x) \oplus L(y)
$$

由于加法为异或，零元必然映射到零元：

$$
L(0) = L(x \oplus x) = L(x) \oplus L(x) = 0
$$

根据线性律，任意 $M$ 比特输入 $x \in [0, 2^M-1]$ 可唯一展开为 2 的幂次方基底之和：

$$
x = \bigoplus_{k=0}^{M-1} b_k \cdot 2^k \quad (b_k \in \{0, 1\})
$$

映射值 $L(x)$ 严格等于激活比特对应基底向量的异或和：

$$
L(x) = \bigoplus_{k=0}^{M-1} \left( b_k \cdot L(2^k) \right)
$$

$$
\text{Bases} = \big[ L(1), \, L(2), \, L(4), \, \dots, \, L(2^{M-1}) \big]
$$

这意味着，编译器无需存储 $2^M$ 个映射项，只需存储 $M$ 个基底向量即可确定整个 $M$ 维输入空间的映射函数。

```text
                       基底向量异或求值推演
┌────────────────────────────────────────────────────────┐
│ 设定基底: L(1) = 1,  L(2) = 2,  L(4) = 5               │
├────────────────────────────────────────────────────────┤
│ • L(0) = L(0 ⊕ 0) = (0)                                │
│ • L(1) = L(1) = 1                                      │
│ • L(2) = L(2) = 2                                      │
│ • L(3) = L(1 ⊕ 2) = L(1) ⊕ L(2) = 1 ⊕ 2 = 3           │
│ • L(4) = L(4) = 5                                      │
│ • L(5) = L(1 ⊕ 4) = L(1) ⊕ L(4) = 1 ⊕ 5 = 4           │
│ • L(6) = L(2 ⊕ 4) = L(2) ⊕ L(4) = 2 ⊕ 5 = 7           │
│ • L(7) = L(1 ⊕ 2 ⊕ 4) = 1 ⊕ 2 ⊕ 5 = 6                  │
└────────────────────────────────────────────────────────┘
```

### 2.2 布尔矩阵点积与坐标变换代数
在形式化代数中，多维映射等价于一个布尔变换矩阵 $B$ 与输入列向量 $a$ 的乘法：

$$
y = B \cdot a \pmod 2
$$

设输入空间由 $M$ 个比特构成，输出空间由 $N$ 个比特构成：
* $a = (a_0, a_1, \dots, a_{M-1})^T$ 为输入的二进制比特列向量；
* $B$ 为 $N \times M$ 的布尔矩阵，其第 $k$ 列即为第 $k$ 个输入比特的基底向量 $B_k = L(2^k)$；
* $y = (y_0, y_1, \dots, y_{N-1})^T$ 为输出坐标比特列向量。

```text
               GF(2) 布尔矩阵与列向量乘法示意
┌────────────────────────────────────────────────────────┐
│  | y0 |   | b00  b01  b02  b03 |   | a0 |              │
│  | y1 | = | b10  b11  b12  b13 | · | a1 | (mod 2)      │
│  | y2 |   | b20  b21  b22  b23 |   | a2 |              │
│  | y3 |   | b30  b31  b32  b33 |   | a3 |              │
│     ▲       ▲    ▲    ▲    ▲                           │
│     │       │    │    │    │                           │
│   输出    L(1) L(2) L(4) L(8)                          │
└────────────────────────────────────────────────────────┘
```

#### 紧凑位向量编码与单周期求值
在 C++ 实现中，输入向量 $a$ 与基底列向量 $B_k$ 直接存储为标准整数。矩阵乘法在计算机中无需遍历矩阵元素，转化为位掩码测试与累加异或：

$$
y = \bigoplus_{k=0}^{M-1} \left( \text{if } (a \text{ \& } (1 \ll k)) \text{ then } B_k \text{ else } 0 \right)
$$

#### 空间展平与还原机制
当处理多维输入（如 `(lane, warp)`）与多维输出（如 `(dim0, dim1)`）时，`LinearLayout` 支持维度的展平（Flatten）与重构（Reshape）：
* `flattenIns()`：将多维输入按照具名顺序拼接为单一的 1D 位向量空间；
* `flattenOuts()`：将多维输出坐标拼接为 1D 线性索引；
* `reshapeIns()` / `reshapeOuts()`：按照指定因式分解重构输入或输出空间的几何维度。

### 2.3 核心代数算子推演
`class LinearLayout` 实现了完备的代数算子集合，支持布局的链式组合、空间扩张与方程求解。

#### 1. 复合与链式组合（`compose`）
若 $L_1: A \to B$，$L_2: B \to C$，复合映射 $L = L_2 \circ L_1: A \to C$ 的矩阵表示为布尔矩阵相乘：

$$
B_{\text{comp}} = B_2 \cdot B_1 \pmod 2
$$

复合操作满足结合律：$(L_3 \circ L_2) \circ L_1 = L_3 \circ (L_2 \circ L_1)$。

在代码中通过 `L1.compose(L2)` 实现，可将硬件到共享内存的映射与共享内存到逻辑张量的映射合并为单一步骤。

#### 2. 空间直积与维度拼接（`operator*`）
直积算子用于组合独立的硬件维度或张量空间。根据输入维度的命名关系，分为两种运算语义：

* **异名输入维度拼接（Disjoint InDims）**：
  若 $L_1$ 输入为 `register`，$L_2$ 输入为 `lane`，直积 $L = L_1 \times L_2$ 创建联合输入空间 `(register, lane)`：
  $$L(r, l) = L_1(r) \oplus L_2(l)$$
* **同名输入维度级联（Same InDim）**：
  若两者共享相同输入维度（如 `ret *= identity1D`），直积将该维度的输入空间容量翻倍，追加新的基底向量，实现分层步长扩展。

#### 3. 逆映射与伪逆求解（`invertAndCompose`）
在布局转换中，给定源布局 $L_{\text{src}}$ 与目标布局 $L_{\text{dst}}$，需要求解数据变换矩阵 $X$，满足：

$$
L_{\text{dst}} \circ X = L_{\text{src}} \implies B_{\text{dst}} \cdot X = B_{\text{src}} \pmod 2
$$

Triton 集成 `third_party/f2reduce` 库，在 $\text{GF}(2)$ 有限域上执行高斯消元法（Reduced Row Echelon Form, RREF）：
1. 构造增广矩阵 $[B_{\text{dst}} \mid B_{\text{src}}]$；
2. 利用初等行异或变换将左侧化为简化阶梯形；
3. 解析求出变换矩阵 $X$。

### 2.4 子空间分解与商空间运算
* **子布局提取（`sublayout`）**：
  给定输入维度子集 $I_{\text{sub}}$ 与输出维度子集 $O_{\text{sub}}$，`sublayout` 提取限制映射 $\left.L\right|_{I_{\text{sub}} \to O_{\text{sub}}}$，并将未选中的输入维度固定为 0。用于从 CTA 级布局中提取单 Warp 或单线程内部的局部访存模式。
* **商空间剥离（`quotient`）**：
  当外层循环已经遍历了特定硬件维度（如 `block` 或 `warp`）时，`quotient` 消除该维度的基底贡献，计算剩余内部空间的独立正交基底。

### 2.5 满射性校验与零核空间自由变量
* **满射性（Surjectivity / 像空间完备性）**：
  若布局矩阵的秩满足：
  $$\text{Rank}(B) = \sum_{d} \log_2(\text{outDimSize}_d)$$
  则该映射为满射，保证逻辑张量的每个坐标均有对应的物理存储实体。Triton 默认对所有工作布局执行满射性断言。
* **单射性与核空间（Kernel Space / 自由变量）**：
  当输入比特数 $M$ 大于矩阵秩 $\text{Rank}(B)$ 时，映射非单射，其核空间（Kernel Space / 零空间）维度为：
  $$\text{Nullity}(B) = M - \text{Rank}(B)$$
  核空间基底对应**自由变量（Free Variables）**，表示同一张量元素被多个硬件物理槽位重复持有（例如广播操作）。编译器通过 `getFreeVariableMasks()` 识别冗余维度，在代码生成时生成条件掩码保护（Predication）。

---

## 3. 硬件布局形式化生成机制

在实际编译流程中，基底向量并不是预先硬编码的静态常数，而是由编译器通过几何参数化算法构造生成的。本章深入推导 Triton 官方源码中核心硬件布局的正向生成机制。

### 3.1 规则分块布局的正向推导算法
在 Triton 中，规则网格布局由 `#ttg.blocked` 属性定义，其参数包含四级离散几何约束：
* **`sizePerThread`**：单个线程持有的连续元素形状 $\vec{S}_{\text{reg}}$；
* **`threadsPerWarp`**：单个 Warp 内 32 个线程的多维排布形状 $\vec{S}_{\text{lane}}$；
* **`warpsPerCTA`**：单个 CTA 内各 Warp 的排布形状 $\vec{S}_{\text{warp}}$；
* **`order`**：内存展开的主序 $\vec{O} = (O_0, \dots, O_{R-1})$（$O_0$ 为变化最快的连续轴）。

编译器在将 `#ttg.blocked` 降级为 `LinearLayout` 时，通过官方核心算法 `identityStandardND` 分层构造基底向量并执行空间直积。

```text
┌────────────────────────────────────────────────────────────────────────┐
│             BlockedEncodingAttr::toLinearLayout 构造流水线             │
├────────────────────────────────────────────────────────────────────────┤
│ 步骤 1: 构造各硬件层级的标准 N 维恒等映射 (identityStandardND)           │
│   • L_reg  = identityStandardND("register", sizePerThread, order)      │
│   • L_lane = identityStandardND("lane", threadsPerWarp, order)         │
│   • L_warp = identityStandardND("warp", warpsPerCTA, order)            │
├────────────────────────────────────────────────────────────────────────┤
│ 步骤 2: 硬件维度直积合并 (Product 算子 `*`)                             │
│   • L_cta = L_reg * L_lane * L_warp                                    │
│   • 输入维度合并为 (register, lane, warp), 输出维度对齐到 (dim0, dim1) │
├────────────────────────────────────────────────────────────────────────┤
│ 步骤 3: 全局张量瓦片复制 (combineCtaCgaWithShape)                       │
│   • 沿 block 维度 (CGA 布局) 展开瓦片，覆盖 Tensor 全局 Shape          │
└────────────────────────────────────────────────────────────────────────┘
```

#### 1. `identityStandardND` 算法机理
`identityStandardND` 的任务是为单个硬件输入维度（如 `lane`）生成映射到多维张量坐标 $(\text{dim0}, \dots, \text{dim}_{R-1})$ 的基底集合。

```cpp
// 官方源码: lib/Tools/LayoutUtils.cpp
LinearLayout identityStandardND(StringAttr inDimName, ArrayRef<unsigned> shape,
                                ArrayRef<unsigned> order) {
  auto rank = shape.size();
  SmallVector<StringAttr> outDimNames = standardOutDimNames(ctx, rank);
  LinearLayout ret = LinearLayout::empty();
  for (int i = 0; i < shape.size(); i++) {
    int dim = order[i]; // 沿主序由连续轴向不连续轴推进
    ret *= LinearLayout::identity1D(shape[dim], inDimName, outDimNames[dim]);
  }
  return ret;
}
```

当同名输入维度执行 `ret *= identity1D` 时，算法在内部维护一个累乘步长：
* **连续轴 $d = O_0$**：初始步长为 $1$。为该轴分配 $\log_2(\text{shape}[d])$ 个基底，基底值在输出维 $d$ 上依次为 $1, 2, 4, \dots$；
* **次连续轴 $d = O_1$**：步长更新为前序轴尺寸之积 $\text{shape}[O_0]$。分配 $\log_2(\text{shape}[d])$ 个基底，基底值在输出维 $d$ 上依次为 $1 \times \text{stride}, 2 \times \text{stride}, \dots$。

#### 2. 端到端推导演练：2D 张量 Blocked 布局构造
设张量全局形状为 $16 \times 16$，分块参数配置为：
* `sizePerThread = [1, 2]`（单线程持有 1 行 2 列，共 2 个元素）；
* `threadsPerWarp = [8, 4]`（32 线程划分为 8 行 4 列网格）；
* `warpsPerCTA = [2, 1]`（CTA 包含 2 个 Warp 沿行排布）；
* `order = [1, 0]`（列优先连续，`dim1` 为连续主轴）。

##### 第一步：生成各硬件层级的微观基底向量
编译器首先为每个硬件层级在其局部网格坐标系内推导基底：

1. **寄存器层级（`inDim = "register"`，尺寸 $[1, 2]$）**：
   * 行方向（`dim0`）尺寸为 $1 \implies \log_2(1) = 0$ 个比特；
   * 列方向（`dim1`）尺寸为 $2 \implies \log_2(2) = 1$ 个比特（输入值 $\text{reg}=1$）；
   * 沿主序 `dim1` 生成基底：当 $\text{reg}$ 的第 0 位为 1 时，列坐标偏移 1：
     $$L(\text{reg}=1) = (\text{dim0}=0, \, \text{dim1}=1)$$
2. **线程层级（`inDim = "lane"`，尺寸 $[8, 4]$，共 32 线程需 5 个比特）**：
   * Warp 内 32 个线程由 5 个二进制位 $(b_0, b_1, b_2, b_3, b_4)$ 唯一寻址；
   * 列方向（`dim1`，连续轴）尺寸为 $4 \implies \log_2(4) = 2$ 个比特，分配给低 2 位 $(b_0, b_1)$：
     * $b_0$（输入值 $2^0=1$）：列坐标偏移 1，基底 $L(\text{lane}=1) = (0, 1)$；
     * $b_1$（输入值 $2^1=2$）：列坐标偏移 2，基底 $L(\text{lane}=2) = (0, 2)$；
   * 行方向（`dim0`，次连续轴）尺寸为 $8 \implies \log_2(8) = 3$ 个比特，分配给高 3 位 $(b_2, b_3, b_4)$：
     * $b_2$（输入值 $2^2=4$）：行坐标偏移 1，基底 $L(\text{lane}=4) = (1, 0)$；
     * $b_3$（输入值 $2^3=8$）：行坐标偏移 2，基底 $L(\text{lane}=8) = (2, 0)$；
     * $b_4$（输入值 $2^4=16$）：行坐标偏移 4，基底 $L(\text{lane}=16) = (4, 0)$；
3. **线程束层级（`inDim = "warp"`，尺寸 $[2, 1]$，共 2 个 Warp 需 1 个比特）**：
   * 列方向尺寸为 $1 \implies 0$ 个比特；
   * 行方向尺寸为 $2 \implies \log_2(2) = 1$ 个比特（输入值 $\text{warp}=1$）：
     * $b_0$（输入值 $2^0=1$）：行坐标偏移 1，基底 $L(\text{warp}=1) = (1, 0)$。

##### 第二步：执行空间直积与全局像素对齐
当调用直积算子 $L_{\text{cta}} = L_{\text{reg}} \times L_{\text{lane}} \times L_{\text{warp}}$ 时，直积算法自动将低层级持有的几何跨度作为高层级基底的倍率因子进行缩放，使所有局部网格基底对齐到统一的张量坐标空间中：

1. **`register` 占位**：单线程在列方向跨度为 $2$（尺寸为 $[1, 2]$）；
2. **`lane` 跨度缩放**：
   * `lane` 在列方向移动 1 个线程，在张量坐标中需跨越 1 个线程持有的 2 列：
     * $\text{lane}_0 (2^0) \mapsto (0, 1 \times 2) = (0, 2)$
     * $\text{lane}_1 (2^1) \mapsto (0, 2 \times 2) = (0, 4)$
   * `lane` 在行方向移动 1 个线程，在张量坐标中跨越 1 行：
     * $\text{lane}_2 (2^2) \mapsto (1 \times 1, 0) = (1, 0)$
     * $\text{lane}_3 (2^3) \mapsto (2 \times 1, 0) = (2, 0)$
     * $\text{lane}_4 (2^4) \mapsto (4 \times 1, 0) = (4, 0)$
3. **`warp` 跨度缩放**：
   * 单个 Warp（32 线程）在行方向持有的总行数为 $8 \text{ 线程} \times 1 \text{ 寄存器} = 8$ 行；
   * 因此 `warp` 移动 1 个线程束，在张量行坐标上需跳跃 8 行：
     * $\text{warp}_0 (2^0) \mapsto (1 \times 8, 0) = (8, 0)$。

合并后得到 CTA 瓦片（覆盖 $16 \times 8$）的 7 个基底向量映射：

$$
\begin{aligned}
\text{reg}_0 (2^0) &\mapsto (\text{dim0}=0, \, \text{dim1}=1) \\
\text{lane}_0 (2^0) &\mapsto (\text{dim0}=0, \, \text{dim1}=2) \\
\text{lane}_1 (2^1) &\mapsto (\text{dim0}=0, \, \text{dim1}=4) \\
\text{lane}_2 (2^2) &\mapsto (\text{dim0}=1, \, \text{dim1}=0) \\
\text{lane}_3 (2^3) &\mapsto (\text{dim0}=2, \, \text{dim1}=0) \\
\text{lane}_4 (2^4) &\mapsto (\text{dim0}=4, \, \text{dim1}=0) \\
\text{warp}_0 (2^0) &\mapsto (\text{dim0}=8, \, \text{dim1}=0)
\end{aligned}
$$

##### 第三步：基底向量组装为布尔矩阵
将上述基底排列为布尔矩阵 $B_{\text{cta}}$。定义输入与输出的二进制列向量：

* **输入物理向量 $a$（7 个比特，对应 7 列）**：
  $$a = (\text{reg}_0, \, \text{lane}_0, \, \text{lane}_1, \, \text{lane}_2, \, \text{lane}_3, \, \text{lane}_4, \, \text{warp}_0)^T$$
* **输出坐标向量 $y$（7 个比特，对应 7 行）**：
  * $\text{dim0}$（行坐标，范围 $[0, 15]$）由 4 个比特编码：$(\text{dim0}_0, \text{dim0}_1, \text{dim0}_2, \text{dim0}_3)$，权值分别为 $1, 2, 4, 8$；
  * $\text{dim1}$（列坐标，范围 $[0, 7]$）由 3 个比特编码：$(\text{dim1}_0, \text{dim1}_1, \text{dim1}_2)$，权值分别为 $1, 2, 4$；
  $$y = (\text{dim0}_0, \, \text{dim0}_1, \, \text{dim0}_2, \, \text{dim0}_3, \, \text{dim1}_0, \, \text{dim1}_1, \, \text{dim1}_2)^T$$

将各基底的二进制分解填入矩阵的对应列：
* 第 0 列（$\text{reg}_0 \mapsto (0, 1)$）：$\text{dim1}_0 = 1$，其余行均为 0，列向量为 $(0, 0, 0, 0, 1, 0, 0)^T$；
* 第 1 列（$\text{lane}_0 \mapsto (0, 2)$）：$\text{dim1}_1 = 1$，其余行均为 0，列向量为 $(0, 0, 0, 0, 0, 1, 0)^T$；
* 第 2 列（$\text{lane}_1 \mapsto (0, 4)$）：$\text{dim1}_2 = 1$，其余行均为 0，列向量为 $(0, 0, 0, 0, 0, 0, 1)^T$；
* 第 3 列（$\text{lane}_2 \mapsto (1, 0)$）：$\text{dim0}_0 = 1$，其余行均为 0，列向量为 $(1, 0, 0, 0, 0, 0, 0)^T$；
* 第 4 列（$\text{lane}_3 \mapsto (2, 0)$）：$\text{dim0}_1 = 1$，其余行均为 0，列向量为 $(0, 1, 0, 0, 0, 0, 0)^T$；
* 第 5 列（$\text{lane}_4 \mapsto (4, 0)$）：$\text{dim0}_2 = 1$，其余行均为 0，列向量为 $(0, 0, 1, 0, 0, 0, 0)^T$；
* 第 6 列（$\text{warp}_0 \mapsto (8, 0)$）：$\text{dim0}_3 = 1$，其余行均为 0，列向量为 $(0, 0, 0, 1, 0, 0, 0)^T$。

```text
               B_cta 矩阵点积求值全景 (y = B_cta · a mod 2)
                 输入物理比特 (列):
                 reg0  lane0 lane1 lane2 lane3 lane4 warp0
   dim0_0 (1)  [  0     0     0     1     0     0     0  ]   [ reg0  ]   [ lane2 ]
   dim0_1 (2)  |  0     0     0     0     1     0     0  |   | lane0 |   | lane3 |
   dim0_2 (4)  |  0     0     0     0     0     1     0  |   | lane1 |   | lane4 |
   dim0_3 (8)  |  0     0     0     0     0     0     1  | · | lane2 | = | warp0 | (dim0 行坐标)
   dim1_0 (1)  |  1     0     0     0     0     0     0  |   | lane3 |   | reg0  |
   dim1_1 (2)  |  0     1     0     0     0     0     0  |   | lane4 |   | lane0 |
   dim1_2 (4)  [  0     0     1     0     0     0     0  ]   [ warp0 ]   [ lane1 ] (dim1 列坐标)
```

点积结果表明，张量坐标由硬件比特直接决定：

$$
\begin{aligned}
\text{dim0} &= \text{lane}_2 \cdot 1 + \text{lane}_3 \cdot 2 + \text{lane}_4 \cdot 4 + \text{warp}_0 \cdot 8 \\
\text{dim1} &= \text{reg}_0 \cdot 1 + \text{lane}_0 \cdot 2 + \text{lane}_1 \cdot 4
\end{aligned}
$$

$B_{\text{cta}}$ 是一个置换矩阵（每行每列仅含一个 1），证明了硬件物理槽位与张量坐标之间严格的无冲突双射关系。

> [!NOTE]
> **置换矩阵（Permutation Matrix）**：在线性代数中，每行与每列均恰好只有一个 `1`、其余元素全为 `0` 的方阵称为置换矩阵。它乘以向量时仅调换向量元素的位置，不改变数值，也不发生数据混合。
> 在无数据广播与无空洞的满射场景下，`Blocked` 布局的布尔矩阵必然是置换矩阵。因为 `Blocked` 分块仅将各个硬件物理比特一对一分配给特定的张量坐标比特，不包含任何异或交织计算，物理本质为纯位重排（Bit Rewiring）。

##### 第四步：CGA 瓦片全局扩张
CTA 瓦片尺寸为 $16 \times 8$，而张量全局尺寸为 $16 \times 16$。`combineCtaCgaWithShape` 沿列方向引入 1 个 `block` 比特（尺寸为 $16 / 8 = 2$），生成第 8 个全局基底：

$$
\text{block}_0 (2^0) \mapsto (\text{dim0}=0, \, \text{dim1}=8)
$$

输入空间拓展为 `(register, lane, warp, block)`，完整覆盖全局 $16 \times 16 = 256$ 个张量元素。

### 3.2 切片与降维布局的代数投影算法
在执行张量切片算子（如 `tt.expand_dims` 逆操作）或沿特定轴进行规约（Reduction）时，高维张量降级为低维张量，对应的布局由 `#ttg.slice` 属性表示。

```cpp
// 官方源码: lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp
LinearLayout SliceEncodingAttr::toLinearLayout(ArrayRef<int64_t> shape) const {
  // 1. 重构父布局的原始多维形状 (切片轴插入尺寸 1)
  SmallVector<int64_t> parentShape(shape);
  parentShape.insert(parentShape.begin() + getDim(), 1);
  LinearLayout parentLL = toLinearLayout(parentShape, getParent());

  // 2. 投影消除被切除的输出维度
  auto sliceLL = removeStandardDim(parentLL, getDim());

  // 3. 剪枝 register 维度中的全零基底 (释放无效物理槽位)
  auto bases = sliceLL.getBases();
  std::vector<std::vector<int>> newRegBases;
  for (const auto &basis : bases[S("register")]) {
    if (llvm::any_of(basis, [](int b) { return b != 0; }))
      newRegBases.push_back(basis);
  }
  bases[S("register")] = newRegBases;
  return LinearLayout(std::move(bases), llvm::to_vector(sliceLL.getOutDimNames()));
}
```

代数投影流程包含三个严格步骤：
1. **父空间重构**：在被切片轴 $k = \text{getDim}()$ 处插入尺寸 $1$，求出父布局的完整线性映射 $L_{\text{parent}}$；
2. **输出空间投影（`removeStandardDim`）**：从 $L_{\text{parent}}$ 的输出维度集合中移除 $\text{dim}_k$，将剩余输出维度向前对齐重编号；
3. **物理寄存器剪枝**：沿被切片轴展开的寄存器基底在投影后变为全零向量，算法自动剔除全零基底，使寄存器输入空间收缩为实际所需的有效比特数。

### 3.3 共享内存 Swizzle 的相位解析与交织算法
共享内存由 32 个独立的 Memory Bank 组成（每个 Bank 宽 4 字节）。当同一 Warp 内的多个线程同时访问同一个 Bank 的不同地址时，硬件将产生 Bank 冲突并强制串行化执行。

```text
        无 Swizzle 布局: Bank 冲突示意 (同列线程命中同一 Bank)
        Bank 0   Bank 1   Bank 2   Bank 3
Row 0:  (0, 0)   (0, 1)   (0, 2)   (0, 3)   <-- Thread 0,1,2,3 并发访问
Row 1:  (1, 0)   (1, 1)   (1, 2)   (1, 3)   <-- 若按列读取 (0,0),(1,0)...
Row 2:  (2, 0)   (2, 1)   (2, 2)   (2, 3)       全部命中 Bank 0 (4-way 冲突)
Row 3:  (3, 0)   (3, 1)   (3, 2)   (3, 3)
```

Swizzle 机制通过将行号与列号进行位异或来打散存储位置：

$$
(t, w) \mapsto (t, \, w \oplus t)
$$

在 Triton 官方源码（`swizzledSharedToLinearLayout`）中，Swizzle 基底通过以下解析算法直接构造：

```cpp
// 1. 列基底 (连续轴): 保持标准的 2^k 步长
for (int col = 1; col < numCols; col *= 2) {
  bases2D.push_back({0, col});
}
// 2. 行基底 (跨行轴): 注入 Swizzle 相位异或偏移量
for (int row = 1; row < numRows; row *= 2) {
  int vec = shared.getVec();
  int perPhase = shared.getPerPhase();
  int maxPhase = shared.getMaxPhase();
  
  int colShift = (vec * ((row / perPhase) % maxPhase)) % numCols;
  bases2D.push_back({row, colShift});
}
```

#### 推导实例：$4 \times 4$ Swizzle 映射
取参数 `vec=1`, `perPhase=1`, `maxPhase=4`，算法生成的 4 个基底向量为：
* 列基底：$L(t=0, w=1) = (0, 1),\; L(t=0, w=2) = (0, 2)$；
* 行基底：$L(t=1, w=0) = (1, 1),\; L(t=2, w=0) = (2, 2)$。

异或展开后的完整物理排布如下：

```text
               Swizzle 映射表: L(t, w) 分散排布
        w=0       w=1       w=2       w=3
t=0    (0, 0)    (0, 1)    (0, 2)    (0, 3)
t=1    (1, 1)    (1, 0)    (1, 3)    (1, 2)
t=2    (2, 2)    (2, 3)    (2, 0)    (2, 1)
t=3    (3, 3)    (3, 2)    (3, 1)    (3, 0)
```

在任意一列中，相同的列坐标被分散到不同的 Bank 中，消除了跨步读取时的 Bank 冲突。

### 3.4 张量核心矩阵乘操作数布局的直积构造
在 NVIDIA Tensor Core（Ampere MMAv2 / Hopper WGMMA）和 AMD CDNA（MFMA）架构中，硬件矩阵指令（如 `mma.sync`）对操作数寄存器有严格的微观结构要求。

官方源码通过将硬件最小矩阵核心（Matrix Core Atom）与外层 Warp 重复基底进行直积拼接：

```cpp
// 官方源码: lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp
// 构造 Ampere 架构 MMA 核心瓦片 (nvidiaMmaTile)
ctaLayout = ctaLayout *
            LinearLayout::identity1D(kWidth, S("register"), dimNames[inner]) *
            LinearLayout::identity1D(4, S("lane"), dimNames[inner]) *
            LinearLayout::identity1D(8, S("lane"), dimNames[outer]) *
            LinearLayout::identity1D(m / 8, S("register"), dimNames[outer]) *
            LinearLayout::identity1D(n / (kWidth * 4), S("register"), dimNames[inner]);
```

#### MMA 核心瓦片的直积解构
以 Ampere 架构 $16 \times 8 \times 16$ MMA 为例，硬件在一个 Warp（32 线程）内部处理一个子矩阵块，其内部基底结构按照硬件数据通路严格定义：
1. **内部连续维度（Inner Dim）**：
   * 分配 `kWidth` 个寄存器基底：覆盖单个线程持有的连续打包向量（如 16-bit 浮点数）；
   * 分配 4 个 `lane` 比特：跨越 Warp 内部连续分配的 4 个线程；
2. **外部转置维度（Outer Dim）**：
   * 分配 8 个 `lane` 比特：覆盖 Warp 内其余 8 组线程，形成 $8 \times 4 = 32$ 线程网格；
3. **宏观寄存器重复（Repetition）**：
   * 分配 $m / 8$ 与 $n / (kWidth \times 4)$ 个寄存器基底，将微观 Atom 瓦片铺满整个张量分块。

直积算子将微观硬件指令限制与宏观寄存器分块无缝融合，使得 MMA 操作数布局完全纳入统一的布尔矩阵体系之中。
