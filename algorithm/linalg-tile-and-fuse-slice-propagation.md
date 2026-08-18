# Linalg 平铺融合切片逆向传播算法

> 本文系统剖析 MLIR `Linalg` 体系与多面体编译模型中的 **Tile and Fuse 机制及逆向切片传播算法（Backward Slice Propagation）**。深入推导从下游 Consumer 的平铺输出（Tile）逆向计算上游 Producer 必需输入切片（Bounding Box）的几何数学公式，并量化评估 SRAM 显存带宽收益与边界光环（Halo）重算冗余之间的权衡边界。

## 1. 算子平铺融合架构动因

### 1.1 图级算子融合瓶颈

在深度学习模型中，数据搬运（Memory Bandwidth）往往是限制 GPU/NPU 算力利用率的最大瓶颈。

传统的计算图模式匹配与算子融合（Graph-Level Fusion，如将 `Conv2D` $\to$ `BiasAdd` $\to$ `ReLU` 合并为单一 Kernel）存在明显的适用边界：
1. **逐元素算子融合容易**：如 `Add + ReLU + Mul`，每个线程独立计算单个元素，中间值可在寄存器直接流转；
2. **多维滑动窗口/特征图融合受限**：当上游算子为卷积或池化时，特征图尺寸较大（如 $1024 \times 1024$），中间特征图无法全部容纳在 GPU 极小的高速片上共享内存（Shared Memory / SRAM）中；
3. **全局显存往返开销**：如果不做切分，编译器被迫将中间结果完整写回全局显存（HBM/DRAM），下一个算子再重新读取，造成大量的带宽浪费。

### 1.2 Tile and Fuse 局部计算范式

MLIR `Linalg` Dialect 提出了 **Tile and Fuse（平铺后融合）** 范式：

```text
【传统全局执行模型 (大量 HBM 往返)】
Producer [Conv2D] ──(写入全局 HBM 64x64)──► [HBM Buffer] ──(读取全局 HBM 64x64)──► Consumer [ReLU]

【Linalg Tile and Fuse 模型 (驻留片上 SRAM)】
Loop Over Tiles (Tile Size: 16x16):
  1. 逆向计算该 Tile 所需的 Producer 输入切片 (如 18x18)
  2. 从全局 HBM 仅读取该 18x18 切片到 SRAM
  3. 片上完成 Conv2D 16x16 计算
  4. 片上寄存器直接完成 ReLU 激活
  5. 仅将最终 16x16 结果写回全局 HBM
```

核心优势在于：**将大图切分成适应片上 Cache/SRAM 容量的局部 Tile，在 Tile 粒度完成链式计算，彻底消除中间大张量的全局内存往返**。

## 2. 切片逆向几何传播数学模型

### 2.1 点对点及维度置换映射

给定 Consumer 请求的输出区间切片 $[offset_i, offset_i + size_i)$：
- **逐元素算子（Elementwise）**：输入与输出索引 $1:1$ 严格恒等映射：
  $$\text{InputSlice}_i = \text{OutputSlice}_i$$
- **转置算子（Transpose 2D）**：维度坐标直接互换：
  $$\text{InputSlice}_0 = \text{OutputSlice}_1, \quad \text{InputSlice}_1 = \text{OutputSlice}_0$$

### 2.2 卷积滑动窗口包围盒推导

对于卷积、池化或模板计算（Stencil）：
- 设卷积核尺寸为 $K$，步长为 $S$，膨胀系数为 $D$；
- 设下游 Consumer 请求的输出切片在某空间维度上的闭区间为：$[out\_off, \; out\_off + size - 1]$。

```text
                     卷积滑动窗口切片逆向映射几何关系
输出点 0                 依赖输入: out_off * S
  │                       │
  ▼                       ▼
 [ 0 ]   ...   [ size - 1 ]
                 │
                 ▼
               依赖输入: (out_off + size - 1) * S + (K - 1) * D
```

#### 数学推导公式：
1. **输入切片起始索引**：
   $$\text{In\_Start} = out\_off \times S$$
2. **输入切片覆盖长度（Bounding Box Size）**：
   $$\text{In\_Size} = (size - 1) \times S + (K - 1) \times D + 1$$

若卷积核 $K = 3, S = 1, D = 1$，请求 $16$ 个输出点：
$$\text{In\_Size} = (16 - 1) \times 1 + (3 - 1) \times 1 + 1 = 15 + 2 + 1 = 18$$
即计算 $16 \times 16$ 的局部卷积，需要 $18 \times 18$ 的局部输入数据。

### 2.3 边缘边界截断保护（Clamping）

在特征图的边缘 Tile 处，若根据公式计算的切片右边界超出了输入张量的原始物理维度 $Shape$，编译器必须进行边界截断（Clamping），防止越界访存：

$$\text{In\_Start} = \max(0, \text{In\_Start})$$
$$\text{In\_Size} = \min(\text{In\_Size}, \; \text{InputShape} - \text{In\_Start})$$

## 3. 显存带宽及 Halo 冗余权衡模型

### 3.1 全局访存量对比模型

设输入大小为 $S_{in}$，中间结果大小为 $S_{mid}$，最终输出大小为 $S_{out}$：

1. **不融合基线（Un-fused）全局访存总量**：
   $$\text{Traffic}_{\text{unfused}} = \text{Read}(S_{in}) + \text{Write}(S_{mid}) + \text{Read}(S_{mid}) + \text{Write}(S_{out}) = S_{in} + 2 S_{mid} + S_{out}$$

2. **平铺融合（Tile & Fuse）全局访存总量**：
   由于中间结果 $S_{mid}$ 在 SRAM / 寄存器中直接被消费，零全局显存写入：
   $$\text{Traffic}_{\text{fused}} = \sum_{\text{all tiles}} \text{Read}(\text{InputSlice}_{\text{tile}}) + \text{Write}(S_{out})$$

### 3.2 平铺尺寸敏感度理论分析

由于相邻 Tile 之间存在重叠的光环区（Halo Region），Tile 边界处的数据会被多个相邻 Tile 重复读取（或在某些架构下重复计算）：

$$\text{Halo Overhead} = \frac{\sum \text{Tile Input Fetches} - S_{in}}{S_{in}}$$

- **Tile 尺寸过小（如 $8 \times 8$）**：边界光环占 Tile 面积比例偏高，过多的重复读取会显著抵消融合带来的带宽削减收益；
- **Tile 尺寸适中（如 $16 \times 16 \sim 32 \times 32$）**：Halo 冗余占比急剧收敛，能够实现接近理论极限的全局显存带宽节省；
- **Tile 尺寸过大（如 $64 \times 64$）**：虽然 Halo 冗余降至接近零，但会超出片上 SRAM / Shared Memory 的物理容量上限。

## 4. C++ 仿真引擎及实验评估

在 [02-linalg-tile-and-fuse-slice-propagation.cpp](02-linalg-tile-and-fuse-slice-propagation.cpp) 中，我们构建了纯标准 C++17 的仿射平铺推导引擎。

### 4.1 几何切片推导算法实现

引擎通过面向数据设计的轻量几何结构记录各维度区间，并在 `infer_input_slice` 中实现闭包推导：

```cpp
struct Slice1D {
  int64_t offset = 0; // 起始偏移
  int64_t size = 0;   // 连续长度
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
    // H 维逆向包围盒推导
    int64_t in_h_start = out_slice.dims[0].offset * stencil.stride_h;
    int64_t in_h_len = (out_slice.dims[0].size - 1) * stencil.stride_h +
                       (stencil.kernel_h - 1) * stencil.dilation_h + 1;
    // 边界对齐截断
    in_h_len = std::min(in_h_len, input_shape[0] - in_h_start);
    in_slice.dims[0] = {in_h_start, in_h_len, 1};
    // W 维同理...
  }
  return in_slice;
}
```

### 4.2 典型用例仿真与敏感度矩阵

对 $66 \times 66 \to \text{Conv2D (3x3)} \to 64 \times 64 \to \text{ReLU} \to 64 \times 64$ 的流水线进行全网格推导仿真：

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

通过扫描不同 Tile 尺寸，得到量化权衡与架构决策矩阵：

| Tile 尺寸 | 总分块数 | 全局访存节省比（HBM Saved） | Halo 重叠搬运开销（Overhead） | 架构选型建议 |
| :---: | :---: | :---: | :---: | :--- |
| **$8 \times 8$** | 64 | 36.93 % | 46.92 % | ❌ 边界碎片过多，不推荐 |
| **$16 \times 16$** | 16 | **44.24 %** | **19.00 %** | ✔️ **SRAM 受限端侧推荐** |
| **$32 \times 32$** | 4 | **47.60 %** | **6.15 %** | ✔️ **GPU Shared Mem 最优解** |
| **$64 \times 64$** | 1 | 49.21 % | 0.00 % | ⚠️ 需超大片上容量支撑 |
