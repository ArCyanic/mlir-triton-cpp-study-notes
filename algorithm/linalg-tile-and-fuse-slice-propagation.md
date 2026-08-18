# Linalg 平铺融合切片逆向传播算法

> 本文系统剖析 MLIR `Linalg` 体系与多面体编译模型中的 **Tile and Fuse 机制及逆向切片传播算法（Backward Slice Propagation）**。从传统图级算子融合在片上 SRAM 约束下的物理瓶颈出发，推导基于 MLIR `AffineMap` 仿射变换矩阵与滑动窗口的多维包围盒（Bounding Box）数学模型；深入剖析多算子深层长链（Deep Multi-Op Chain）的逆向递归切片回溯与分支合并（Branch Confluence）机制；量化建模显存带宽节省与边界光环（Halo）重叠搬运冗余之间的权衡边界；最后推导 C++ 仿射平铺推导引擎的完整实现与多维分块敏感度矩阵。

---

## 1. 算子平铺融合架构动因与物理瓶颈

### 1.1 图级算子融合瓶颈

在深度学习加速器（如 GPU、NPU、TPU）中，计算单元的峰值算力已达到数百 TFLOPS，但内存带宽（Memory Bandwidth）的增长远滞后于算力增长。大量神经网络模型的执行瓶颈在于**数据在片外高带宽内存（HBM/DRAM）与片上高速缓存（SRAM/Shared Memory）之间的往返搬运开销**。

传统的计算图模式匹配融合（Graph-Level Fusion，如将 `Conv2D` $\to$ `BiasAdd` $\to$ `ReLU` 合并为单个粗粒度算子）存在明显的物理边界：

- **逐元素算子融合极为自然**：在 `Add + Mul + ReLU` 链路中，每个线程独立处理单个标量元素，中间结果可在寄存器中直接完成传递，内存往返次数为 0；
- **滑动窗口算子受制于 SRAM 容量**：当上游算子为大尺寸卷积（`Conv2D`）或池化（`Pooling`）时，中间特征图尺寸往往高达数十 MB，远超 GPU 片上 Shared Memory（通常仅 100~200 KB）的物理容量上限；
- **中间数据强制回写全局显存**：如果不做空间切分，编译器只能被迫将上游算子的完整输出写回全局 HBM，下游算子再完整读入，造成极其昂贵的显存带宽浪费与延迟停顿。

---

### 1.2 Tile and Fuse 局部计算范式

MLIR `Linalg` Dialect 提出了 **Tile and Fuse（平铺后融合）** 的结构化分块计算范式：

```text
【1. 传统全局执行模型 (大量 HBM 往返)】
Producer [Conv2D] ──(写入全局 HBM 64x64)──► [HBM Buffer] ──(读取全局 HBM 64x64)──► Consumer [ReLU]

【2. Linalg Tile and Fuse 模型 (驻留片上 SRAM)】
Loop Over Tiles (输出平铺尺寸: 16x16):
  ├─► 1. 逆向计算该 16x16 输出切片所需 Producer 必需的最小输入包围盒 (如 18x18)
  ├─► 2. 从全局 HBM 仅读取该 18x18 输入切片并加载至片上 SRAM
  ├─► 3. 片上 SRAM 极速完成 Conv2D 16x16 局部卷积计算
  ├─► 4. 寄存器直接流水线完成 ReLU 激活
  └─► 5. 仅将最终的 16x16 结果写回全局 HBM
```

Tile and Fuse 的核心架构优势在于：**将全图大张量切分为能够完全塞入片上 SRAM / 缓存线物理容量的局部 Tile，在 Tile 粒度完成算子链的连续就地计算，彻底消除了中间大张量在全局 HBM 上的读写往返**。

---

## 2. 切片逆向几何传播与多面体仿射数学模型

### 2.1 点对点与维度置换映射

在执行 Tile and Fuse 时，调度器首先对最下游 Consumer 的输出张量按照给定的 Tile 尺寸进行网格划分，产生目标输出切片 $[offset_i, offset_i + size_i)$。随后通过逆向传播推导上游 Producer 所需的输入切片。

- **逐元素算子（Elementwise）**：输入与输出索引之间存在 $1:1$ 的严格恒等映射：
  $$\text{InputSlice}_i = \text{OutputSlice}_i$$
- **二维转置算子（Transpose 2D）**：空间坐标发生正交置换：
  $$\text{InputSlice}_0 = \text{OutputSlice}_1, \quad \text{InputSlice}_1 = \text{OutputSlice}_0$$

---

### 2.2 卷积滑动窗口多面体包围盒推导

对于卷积、池化或模板计算（Stencil）：
- 设卷积核尺寸为 $K$，卷积步长为 $S$，膨胀系数为 $D$；
- 设下游 Consumer 请求的输出切片在某空间维度上的闭区间为 $[out\_off, \; out\_off + size - 1]$。

```text
                     卷积滑动窗口切片逆向映射几何关系
输出点 0 (out_off)            依赖输入起点: out_off * S
  │                                    │
  ▼                                    ▼
 [ 0 ]         ...            [ size - 1 ]
                                       │
                                       ▼
                     依赖输入终点: (out_off + size - 1) * S + (K - 1) * D
```

#### 标量滑动窗口包围盒公式

通过对输出区间首尾端点的映射关系求极值，推导出上游 Producer 必需的输入包围盒（Bounding Box）：

1. **输入切片起始索引**：
   $$\text{In\_Start} = out\_off \times S$$
2. **输入切片覆盖长度（Bounding Box Size）**：
   $$\text{In\_Size} = (size - 1) \times S + (K - 1) \times D + 1$$

当 $K = 3, S = 1, D = 1$ 时，若下游请求 $16$ 个连续输出点，则对应输入长度为：
$$\text{In\_Size} = (16 - 1) \times 1 + (3 - 1) \times 1 + 1 = 15 + 2 + 1 = 18$$
即计算 $16 \times 16$ 的局部卷积，必须加载 $18 \times 18$ 的输入数据。

#### MLIR AffineMap 仿射矩阵形式化

在多面体编译模型中，Consumer 输出迭代空间向量 $\vec{i}_{\text{out}}$ 与 Producer 输入坐标向量 $\vec{i}_{\text{in}}$ 通过仿射映射矩阵关联：

$$\vec{i}_{\text{in}} = A \vec{i}_{\text{out}} + \vec{b}$$

若输出切片定义为一个多维超长方体盒 $B_{\text{out}} = [\vec{l}_{\text{out}}, \vec{u}_{\text{out}}]$，则逆向传播产生的输入多面体最小外接包围盒 $B_{\text{in}} = [\vec{l}_{\text{in}}, \vec{u}_{\text{in}}]$ 满足：

$$\vec{l}_{\text{in}} = \min_{\vec{i} \in B_{\text{out}}} (A \vec{i} + \vec{b}), \quad \vec{u}_{\text{in}} = \max_{\vec{i} \in B_{\text{out}}} (A \vec{i} + \vec{b})$$

---

### 2.3 边缘边界截断与填充保护

在全图边缘的 Tile 处，若根据理论公式逆向推导出的输入切片跨越了物理张量的边界，编译器必须执行**边界截断（Clamping）**，防御非法的越界内存访问：

$$\text{In\_Start} = \max(0, \text{In\_Start})$$
$$\text{In\_Size} = \min(\text{In\_Size}, \; \text{InputShape} - \text{In\_Start})$$

---

## 3. 多算子长链递归传播与 Halo 冗余权衡

### 3.1 深层长链逆向切片递归传播

#### 跨算子级联切片回溯

在复杂的深度神经网络中，计算流往往跨越多个算子（如 `Input` $\to$ `Conv1` $\to$ `Conv2` $\to$ `Add` $\to$ `ReLU`）。切片传播算法沿着 DAG 的逆拓扑序执行**多级递归回溯（Multi-Op Deep Chain Backward Propagation）**：

```text
               深层网络切片逆向递归传播与分支合并
   [Input Tensor] (必需区域: 20x20) ◄─────────┐ 逆向推导 2
         │                                    │
         ▼                                    │
    [Op 1: Conv1 (3x3)]                       │
         │ (中间切片: 18x18) ◄────────┐ 逆向推导 1│
         ▼                            │       │
    [Op 2: Conv2 (3x3)]               │       │ (跨步残差包围盒并集)
         │                            │       │
         ▼                            │       │
    [Op 3: Residual Add] ─────────────┴───────┘
         │ (目标切片: 16x16)
         ▼
    [Op 4: ReLU] ◄── [下游调度请求: 16x16 Tile]
```

#### 分支汇聚点的包围盒并集计算

当逆向传播遇到多输入汇聚算子（如 `Add`、`Concat`）时，算法分别逆向计算各个输入分支所需的切片区间。对于跨步残差直连（Skip Connection），算法通过求外接包围盒并集（Bounding Box Union）确保一次性将完整数据拉入片上：

$$\text{UnionStart}_d = \min(\text{Slice1\_Start}_d, \text{Slice2\_Start}_d)$$
$$\text{UnionEnd}_d = \max(\text{Slice1\_End}_d, \text{Slice2\_End}_d)$$

---

### 3.2 显存带宽与 Halo 边界重算冗余权衡

#### 全局访存流量对比模型

设输入张量大小为 $S_{in}$，中间激活特征图大小为 $S_{mid}$，最终产出大小为 $S_{out}$：

- **不融合基线（Un-fused）全局访存总量**：
  $$\text{Traffic}_{\text{unfused}} = \text{Read}(S_{in}) + \text{Write}(S_{mid}) + \text{Read}(S_{mid}) + \text{Write}(S_{out}) = S_{in} + 2 S_{mid} + S_{out}$$
- **平铺融合（Tile and Fuse）全局访存总量**：中间张量 $S_{mid}$ 完全在片上 SRAM 消费并销毁，不发生全局显存写回：
  $$\text{Traffic}_{\text{fused}} = \sum_{\text{all tiles}} \text{Read}(\text{InputSlice}_{\text{tile}}) + \text{Write}(S_{out})$$

#### Halo 边界重叠搬运冗余比

由于相邻卷积 Tile 在空间边界上存在重叠的光环区（Halo Region），Tile 边缘的数据会被相邻的两个 Tile 重复加载至 SRAM：

$$\text{Halo Overhead} = \frac{\sum \text{Tile Input Fetches} - S_{in}}{S_{in}}$$

- **Tile 尺寸过小（如 $8 \times 8$）**：边界 Halo 占 Tile 总面积比例过高，重复搬运冗余会显著抵消融合带来的带宽收益；
- **Tile 尺寸适中（如 $16 \times 16 \sim 32 \times 32$）**：Halo 冗余占比急剧收敛，能够实现接近理论上限的全局显存带宽节省；
- **Tile 尺寸过大（如 $64 \times 64$）**：虽然 Halo 冗余降至 0%，但其所需的 SRAM 空间将超出芯片物理硬件上限。

---

## 4. C++ 仿真引擎与调度循环实现

### 4.1 几何切片数据结构与推导闭包

在配套实现的 [`02-linalg-tile-and-fuse-slice-propagation.cpp`](02-linalg-tile-and-fuse-slice-propagation.cpp) 中，引擎采用轻量级几何区间结构并在 `infer_input_slice` 中实现闭包推导：

```cpp
struct Slice1D {
    int64_t offset = 0; // 起始偏移
    int64_t size = 0;   // 连续跨度
    int64_t stride = 1; // 步长
};

struct TensorSlice {
    std::vector<Slice1D> dims;
    int64_t num_elements() const;
};

TensorSlice infer_input_slice(const TensorSlice &out_slice) const {
    TensorSlice in_slice;
    if (kind == OpKind::Conv2D) {
        in_slice.dims.resize(2);
        // H 空间维逆向包围盒推导
        int64_t in_h_start = out_slice.dims[0].offset * stencil.stride_h;
        int64_t in_h_len = (out_slice.dims[0].size - 1) * stencil.stride_h +
                           (stencil.kernel_h - 1) * stencil.dilation_h + 1;
        // 边界截断防御越界
        in_h_len = std::min(in_h_len, input_shape[0] - in_h_start);
        in_slice.dims[0] = {in_h_start, in_h_len, 1};
        // W 空间维同理展开...
    }
    return in_slice;
}
```

---

### 4.2 典型用例仿真与敏感度矩阵

针对 $66 \times 66 \to \text{Conv2D (3x3)} \to 64 \times 64 \to \text{ReLU} \to 64 \times 64$ 流水线进行全网格平铺仿真实测：

```text
================ 📊 编译器 Tile & Fuse 收益与代价评估报告 ================
1. 网格分块总数 (Total Tiles):                   16
2. 最终产出数据量 (Final Output Elems):          4096 elems
3. 不融合全局显存访存量 (Un-fused HBM):        16644 elems
4. 融合平铺全局显存访存量 (Fused HBM):           9280 elems
5. 全局显存带宽节省比例 (HBM Bandwidth):      44.24 % 🚀
6. Halo 边界重叠搬运冗余比 (Halo Overhead):    19.01 %
========================================================================
```

通过参数化扫描不同 Tile 尺寸，得到量化权衡与架构决策矩阵：

| 平铺分块尺寸（Tile Size） | 网格分块总数 | 全局访存节省比例（HBM Saved） | Halo 重叠搬运冗余（Overhead） | 体系结构选型建议 |
| :---: | :---: | :---: | :---: | :--- |
| **$8 \times 8$** | 64 | 36.93 % | 46.92 % | ❌ 边界碎片占比过高，不推荐 |
| **$16 \times 16$** | 16 | **44.24 %** | **19.00 %** | ✔️ **SRAM 受限端侧 NPU 最优解** |
| **$32 \times 32$** | 4 | **47.60 %** | **6.15 %** | ✔️ **GPU Shared Memory 最佳平衡点** |
| **$64 \times 64$** | 1 | 49.21 % | 0.00 % | ⚠️ 需超大片上 SRAM 容量支撑 |

---

## 5. 平铺融合决策速查与工程落点矩阵

| 算子类别 | 仿射映射特征 | 切片逆向推导复杂度 | Halo 边界开销 | 典型工程实现与落点 |
| :--- | :--- | :--- | :--- | :--- |
| **逐元素算子（Elementwise）** | 恒等映射（Identity） | $O(1)$ 直接坐标透传 | 0%（无重叠） | `linalg.generic`、Triton Block 加载 |
| **卷积与模板（Conv / Stencil）** | 滑动窗口与跨步仿射 | $O(D)$ 空间包围盒推导 | $5\% \sim 25\%$ | 卷积/池化平铺、GPU 共享内存分块 |
| **规约算子（Reduction）** | 维度坍缩映射 | $O(1)$ 局部切片扩张 | 0% | GEMM 矩阵乘法 $K$ 维切片累加 |
| **多算子长链（Deep Chain）** | 复合矩阵级联 | $O(L \cdot D)$ 递归回溯 | 复合累加 | 端到端算子融合代码生成器 |
