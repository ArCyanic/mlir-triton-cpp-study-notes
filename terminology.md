# 编译器与体系结构核心术语表

> 本文档用于沉淀现代编译器（MLIR / LLVM）、并行执行模型（SIMD / SIMT / SPMD）以及底层体系结构中的通用核心概念，作为持续补充的速查术语表。

---

## 术语索引

- [通道一致性与发散性模型 (Uniformity & Divergence Model)](#通道一致性与发散性模型-uniformity--divergence-model)
- [类型传染 / 类型毒化 (Type Contagion / Type Poisoning)](#类型传染--类型毒化-type-contagion--type-poisoning)

---

### 通道一致性与发散性模型 (Uniformity & Divergence Model)

#### 1. 概念定义
描述在 SIMD（向量化）或 SPMD（GPU Warp / Wavefront）并发执行环境下，数据在**各个协同物理通道（Lane / Thread）**之间的空间分布特征：
- **Uniform（通道一致）**：当前并发执行单元内，所有活动通道看到的值完全相同（$\forall i, j,\; V_i = V_j$）。例如张量外层维度 $N$、标量步长、全局常量。
- **Varying / Divergent（通道发散 / 通道差异）**：不同通道持有的值不同（$\exists i, j,\; V_i \neq V_j$）。例如线程局部 ID、`arange` 展开序列、基于各通道私有地址加载的数据。

#### 2. 与标量/向量形态的正交性
容器几何形态（Scalar vs. Vector）与通道一致性属性是两个独立维度，不可混淆：
- **Uniform Scalar**：单值标量，且跨通道相同。硬件上通常存放在标量寄存器（如 GPU SGPR、CPU/RVV 通用寄存器 GPR），由标量单元单次计算。
- **Uniform Vector (Splat)**：几何形态为向量，但所有 Lane 的元素均由同一个 Uniform 标量广播填充。
- **Varying Vector**：向量各分量跨通道不同，必须占用向量寄存器（如 GPU VGPR、RVV $V$ 寄存器），由 SIMD/Vector 算力单元并发执行。

#### 3. 编译器应用场景
- **发散度分析（Divergence Analysis）**：区分控制流与数据流是否发生通道分裂，指导循环不变量外提与分支合并。
- **结构化访存推导**：以 Triton `RaiseTensorView` 为例，要将散装指针与 Mask 还原为结构化 2D 矩形块，必须满足“线性递增的 Varying 索引”与“通道一致的 Uniform 标量上界”进行比较，否则无法推导硬件截断边界（如 RVV `vsetvli`）。

---

### 类型传染 / 类型毒化 (Type Contagion / Type Poisoning)

#### 1. 概念定义
- **类型传染（Type Contagion）**：编程语言和编译器根据隐式类型提升规则（Type Promotion / Coercion），在二元操作符作用于不同精度或位宽的操作数时，自动将结果向类型格（Type Lattice）的最小上界（LUB）提升的机制。
- **类型毒化（Type Poisoning，工程俗称）**：指系统局部（如辅助函数或宏）非必要地引入了大位宽或特殊类型（如将 `int32` 索引提升为 `int64`），该类型借助传染机制沿表达式数据流图（DAG）向下静默扩散，导致下游发生性能骤降或编译断裂的现象。

#### 2. 核心危害与失效链条
1. **体系结构资源惩罚**：在定长向量体系结构（如 RVV VLEN=2048）中，32 位整数升格为 64 位将导致单个物理向量寄存器容纳元素折半；为保持相同计算 Block 必须将 LMUL 翻倍，可用逻辑寄存器减半，极易触发寄存器溢出换页（Spill）。
2. **模式匹配与降级中断**：中间层 Pass（如 MLIR 算子重写）通常要求严格的同构类型约束。若偏移计算链路被静默提升为 `i64`，而比较上界仍为 `i32`，编译器模式匹配器会因两端类型域失配而拒绝识别，引发编译期硬中断（Hard Error）。

#### 3. 工程防御准则
- **源头收敛**：基础工具库与元信息获取接口（如 `program_id`）严禁无端提升类型位宽，默认保持为平台天然的 `int32`。
- **显式对齐**：确实需要 64 位大寻址时，必须在边界判定处显式将上界和索引同步转换为相同类型，避免依赖语言隐式规则。
