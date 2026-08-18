# 编译器算法体系与真实场景深度解析 (Compiler Algorithms & Scenarios)

> 本目录收录 AI 编译器（MLIR / Triton / TVM / XLA）与高性能算子体系中**最核心、最具工业价值的算法理论模型与 C++17 可运行仿真实现**。
> 摒弃碎片化刷题与脱离实际的纯玩具代码，全部内容以**真实编译器底层架构与核心 Pass 优化场景**为导向建立。

## 1. 编译器算法分类矩阵

编译器算法处于计算机体系结构、离散数学、计算几何与图论的交叉点。依据在编译流水线（Pipeline）中所处的阶段，核心场景正交划分为以下 5 大领域：

```text
                                AI 编译器算法体系
                                       │
     ┌─────────────────┬───────────────┼───────────────┬─────────────────┐
     ▼                 ▼               ▼               ▼                 ▼
【1. 图调度与显存】 【2. 张量几何与平铺】 【3. 符号与类型】 【4. 硬件微架构映射】 【5. 后端指令与寄存器】
  - DAG 峰值显存调度 - Linalg Tile & Fuse  - 符号 Shape 推导 - GPU 共享内存 Swizzle - 线性扫描寄存器分配
  - 静态生命周期复用 - 逆向切片包围盒传播 - 带权并查集 (DSU) - 32-Bank 冲突消除   - Loop Depth 溢出启发式
```

| 模块类别 | 核心算法与数据结构 | 对应编译器底层机制 | 状态与对应文件 |
| :--- | :--- | :--- | :--- |
| **1. 图调度与显存** | 活跃期分析 / 优先队列 / 异或和 $O(1)$ 查找 | MLIR Use-Def 调度 / 拓扑排序 | 📄 [01-peak-memory-dag-scheduler.cpp](01-peak-memory-dag-scheduler.cpp)<br>📖 [peak-memory-dag-scheduling.md](peak-memory-dag-scheduling.md) |
| **2. 张量几何与平铺** | 仿射映射 / 区间逆推 / Halo 边界包围盒 | `MLIR Linalg Tile and Fuse` | 📄 [02-linalg-tile-and-fuse-slice-propagation.cpp](02-linalg-tile-and-fuse-slice-propagation.cpp)<br>📖 [linalg-tile-and-fuse-slice-propagation.md](linalg-tile-and-fuse-slice-propagation.md) |
| **3. 符号与类型推导** | 带权并查集 (Weighted DSU) / 拓扑约束传播 | `Shape Dialect` 维度等价性分析 | 🚧 *(规划中)* |
| **4. 硬件微架构映射** | Galois 域 $GF(2)$ 位异或 / 线性空间双射 | Triton GPU Layout / Bank 冲突消除 | 🚧 *(规划中)* |
| **5. 后端指令与寄存器** | 扫描线 / 优先队列 / 活跃区间修剪 | LLVM CodeGen / 虚拟寄存器分配 | 🚧 *(规划中)* |

## 2. 核心场景与文档索引

### 场景一：计算图 DAG 峰值显存最小化拓扑调度
* **核心痛点**：端侧 NPU/片上 SRAM 空间极小（几百 KB ~ 数 MB），或者超大模型训练时单卡 HBM 极易 OOM。朴素的 BFS/DFS 拓扑排序导致大量大尺寸中间激活张量长时间滞留在显存中。
* **算法解法**：基于 MLIR 风格的 `Operation` 与 `Value` 解耦，结合张量出度与异或和（XOR-Sum）$O(1)$ 快速识别并激活同伴算子，贪心优先发射能释放显存的算子。
* **文档与代码**：
  - 📖 深度图解文档：[peak-memory-dag-scheduling.md](peak-memory-dag-scheduling.md)
  - 📄 C++17 仿真代码：[01-peak-memory-dag-scheduler.cpp](01-peak-memory-dag-scheduler.cpp)

### 场景二：Linalg 平铺融合切片逆向传播算法
* **核心痛点**：传统算子融合仅适用于逐元素算子，面对多维特征图和滑动窗口算子（如 Stencil / Conv2D）无法将中间结果装入片上 Shared Memory / Cache。
* **算法解法**：在 `linalg.generic` 空间中将下游 Consumer 输出划分为局部 Tile，依据仿射映射与步长/膨胀几何关系，逆向推导 Producer 所需的最小连续输入切片（Bounding Box），量化 SRAM 显存带宽收益与 Halo 光环重叠搬运冗余。
* **文档与代码**：
  - 📖 深度图解文档：[linalg-tile-and-fuse-slice-propagation.md](linalg-tile-and-fuse-slice-propagation.md)
  - 📄 C++17 仿真代码：[02-linalg-tile-and-fuse-slice-propagation.cpp](02-linalg-tile-and-fuse-slice-propagation.cpp)

## 3. 编译与运行方式

所有算法实现均为无第三方依赖的纯标准 C++17 代码，可直接使用主流编译器（Clang / GCC）编译运行：

```bash
# 1. 编译运行 DAG 显存调度器
clang++ -std=c++17 -Wall -Wextra 01-peak-memory-dag-scheduler.cpp -o sched && ./sched

# 2. 编译运行 Linalg 逆向切片推导仿真器
clang++ -std=c++17 -Wall -Wextra 02-linalg-tile-and-fuse-slice-propagation.cpp -o linalg && ./linalg
```
