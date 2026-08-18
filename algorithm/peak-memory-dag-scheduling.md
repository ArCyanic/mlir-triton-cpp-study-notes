# 计算图 DAG 峰值显存最小化拓扑调度算法

> 本文系统解构 AI 编译器中**基于张量活跃期（Liveness Intervals）与异或和（XOR-Sum）$O(1)$ 查找的计算图峰值显存最小化拓扑调度算法**。从显存峰值爆炸的物理动因与形式化数学模型出发，证明 DAG 拓扑排序在显存受限场景下的 NP-Complete 复杂度；对比广度优先（BFS）与显存感知深度调度在时间轴上的内存投影差异；深入剖析算子净显存增量（`NetDelta`）、残差连接前瞻加权、惰性优先队列与异或和同伴算子常数时间跃升机理；最后结合 MLIR 风格的 `Operation`/`Value` 解耦体系，推导面向数据设计（DOD / SoA）的高性能 C++ 调度状态机实现。

---

## 1. 计算图显存峰值形式化数学建模

### 1.1 中间激活张量物理驻留与生命周期形式化

在端侧 AI 芯片（如 NPU、DSP）或片上缓存极其严苛的专用硬件中，快速静态随机存取存储器（SRAM）容量通常仅有数百 KB 到数十 MB；而在云端大模型（LLM）训练与推理中，单卡高带宽显存（HBM）也是决定 Batch Size 与上下文长度的核心瓶颈。

在深度学习计算图中，一个复杂的模型可以形式化表示为一个有向无环图 $G = (V, E)$，其中节点 $v \in V$ 表示算子（`Operation`），有向边 $e = (u, v) \in E$ 表示张量（`Value`）的数据依赖：

```text
Op_Produce ──► [ Tensor T (Size = 500 MB) ] ──┬──► Op_Consumer_1 (执行时刻 t1)
                                              └──► Op_Consumer_2 (执行时刻 t2)
                                                   ; 直至 t2 执行完毕，Tensor T 方可释放！
```

设算子的执行拓扑序列为双射函数 $\pi: V \to \{1, 2, \dots, |V|\}$，则对于任意张量 $T_e$（由算子 $u$ 产出并被集合 $\text{Users}(T_e)$ 消费）：
- **定义时刻（Definition Time）**：$t_{\text{def}}(T_e) = \pi(u)$；
- **最后使用时刻（Last-Use Time）**：$t_{\text{last\_use}}(T_e) = \max_{v \in \text{Users}(T_e)} \pi(v)$；
- **张量活跃期（Liveness Interval）**：张量 $T_e$ 在时间半开区间 $[t_{\text{def}}(T_e), t_{\text{last\_use}}(T_e))$ 内必须持续占据物理显存。

在任意执行时刻 $t$，物理显存的**瞬时显存总和（Instantaneous Memory Footprint）**为所有当前处于活跃期的张量尺寸之和：

$$M_\pi(t) = \sum_{e \in E, \; t \in [t_{\text{def}}(T_e), t_{\text{last\_use}}(T_e))} \text{Size}(T_e)$$

---

### 1.2 拓扑排序等价解空间与 NP 难解性

计算图的合法拓扑排序集合记为 $\Pi(G)$。不同的拓扑序列 $\pi \in \Pi(G)$ 在算子数据依赖上是等价的，但在张量生命周期的交叠程度上存在巨大差异。

调度算法的核心优化目标是在所有合法的拓扑排序中，寻找一个能够使**全局并发存活张量最大值最小化**的最优序列：

$$\pi^* = \arg\min_{\pi \in \Pi(G)} \left( \max_{1 \le t \le |V|} M_\pi(t) \right)$$

> [!IMPORTANT]
> **理论复杂度证明**：计算图峰值显存最小化拓扑调度问题在理论上已被严格证明为 **NP-Complete（NP 完全问题）**。该问题可直接规约到经典体系结构中的寄存器分配（Register Allocation）与区间图着色（Interval Graph Coloring）问题。因此，工业级 AI 编译器普遍采用基于启发式打分（Heuristic Scoring）与动态优先级状态机的贪心搜索算法。

---

## 2. 拓扑调度策略时空演进对比

### 2.1 广度优先遍历与并发显存峰值爆炸

以一个具有三分支（视觉、文本、音频特征提取与融合）的多模态计算图为例：

```text
       [Init_Vision]       [Init_Text]       [Init_Audio]
            │                   │                 │
      [Vision_Expand]     [Text_Expand]     [Audio_Expand]
            │                   │                 │
     [Vision_Compress]   [Text_Compress]   [Audio_Compress]
            \                   │                 /
             └──────────► [Cross_Fusion] ◄────────┘
```

若采用标准入度为 0 入队的 FIFO 队列（广度优先 BFS 拓扑排序）：
- 队列会优先将三个分支的第一层初始化算子全部执行完毕（`Init_Vision` $\to$ `Init_Text` $\to$ `Init_Audio`）；
- 随后并发执行三个分支的中间膨胀算子（`Vision_Expand` $\to$ `Text_Expand` $\to$ `Audio_Expand`）；
- **显存爆炸后果**：三个独立分支的大尺寸展开张量**在同一时间点全部驻留在物理显存中**，全局显存峰值达到各分支峰值的直接累加（如 2100 MB），极易诱发运行时 OOM。

---

### 2.2 深度优先贪心调度与即时内存释放

若采用显存感知的深度优先（DFS）与贪心调度策略：
- 调度器在执行完 `Vision_Expand` 后，立刻调度其直接下游 `Vision_Compress`；
- 在展开文本分支之前，视觉分支的超大中间张量已被**彻底消费并就地释放**，显存回落至低水位基线；
- 随后再依次展开并压缩文本分支与音频分支；
- **时空收敛效果**：多个分支的大张量生命周期在时间轴上被完全错开，全局显存峰值被严格压制在单个分支的局部峰值（如 1000 MB，**净节省超过 52% 显存**）。

---

## 3. 显存感知调度算法模型与同伴跃升

### 3.1 算子净显存增量模型

为了在每一步决策时挑选出对当前显存水位最有利的就绪算子，我们形式化定义算子 $Op$ 的**净显存增量（Net Memory Delta）**：

$$\text{NetDelta}(Op) = \sum_{R \in \text{Results}(Op)} \text{Size}(R) - \sum_{O \in \text{Operands}(Op), \; \text{OutDegree}(O) = 1} \text{Size}(O)$$

- **产出开销（Production Overhead）**：算子执行后新生成的 Results 张量必须分配显存；
- **释放红利（Release Dividend）**：若某个输入操作数 $O$ 当前的**剩余未消费出度恰好等于 1**，说明当前算子是该张量的**最后一个活跃消费者**。当前算子执行完毕后该张量将被立即释放，贡献负向的显存缩减红利；
- **贪心决策准则**：优先调度 $\text{NetDelta}(Op)$ **最小（负值绝对值最大）**的就绪算子。

---

### 3.2 残差直连局部陷阱与前瞻加权启发式

在 ResNet、DenseNet 与 Transformer 结构中，广泛存在大跨度的**残差直连（Residual Skip Connections）**：

```text
       [Conv_Root] ──► [ Large Residual Tensor (500 MB) ] ────────────────────┐
            │                                                                  │
            ▼                                                                  ▼
       [Sub_Op_1] ──► [Sub_Op_2] ──► ... ──► [Sub_Op_N (释放 10 MB)] ──► [Add_Fusion]
```

- **局部贪心陷阱（Local Greedy Trap）**：若纯粹按照单步 $\text{NetDelta}$ 最小进行贪心，调度器可能会被分支链条中微小的局部释放（如释放 10 MB 小张量）诱导，从而迟迟不调度通向 `Add_Fusion` 的关键路径，导致 500 MB 的超大残差张量在显存中长期跨步滞留；
- **前瞻深度加权模型（Lookahead Critical Depth Heuristic）**：将关键路径深度（Critical Path Depth）与显存紧迫度融合为综合优先级函数：

$$\text{Priority}(Op) = -\text{NetDelta}(Op) + \alpha \cdot \text{CriticalDepth}(Op)$$

通过调节前瞻超参数 $\alpha$，引导调度器优先推进能够快速释放大跨度残差张量的关键路径，彻底打破局部贪心停滞。

---

### 3.3 惰性优先队列版本控制

在图的动态调度过程中，随着其它算子的执行，某张量的剩余出度会动态递减（例如从 $2 \to 1$），导致依赖该张量的下游就绪算子的 $\text{NetDelta}$ 发生动态跃升。

由于标准二叉堆不支持低开销的 `decrease-key` 操作，算法采用了**版本号惰性失效机制（Lazy Invalidation via Monotonic Versioning）**：

```text
算子优先级跃升 (从 NetDelta = +200 跃升为 NetDelta = -500)
  │
  ├─► 1. 递增本地版本号: op_versions[op_id]++ (如由 v1 变为 v2)
  ├─► 2. 将全新三元组推入优先队列: push({priority: -500, op_id, version: v2})
  │      ; 旧版本三元组 {-200, op_id, v1} 依然留在堆内
  │
  ▼
【堆顶出队校验 (Pop Validation)】:
  - 若弹出的堆顶版本号 != op_versions[op_id]，或 is_scheduled[op_id] == true:
  - 直接 O(1) 丢弃该过期元素，继续弹出下一合法堆顶！
```

---

### 3.4 异或和同伴算子常数时间跃升机理

#### 异或自反代数性质

当一个张量 $T$ 的剩余出度从大于 1 降为 1 时，意味着全图中只剩下最后 1 个活跃消费者算子。此时必须立即通知该消费者算子刷新其 $\text{NetDelta}$ 并跃升至堆顶。

若通过遍历张量的原始消费者列表来寻找剩余的算子，时间复杂度与出度 $O(\text{Degree})$ 线性相关。利用异或运算的核心自反性质（$A \oplus A = 0$ 与 $A \oplus 0 = A$），可实现 **$O(1)$ 常数时间极速定位**：

1. **图初始化阶段**：将张量 $T$ 的所有消费者 ID 连续执行异或：
   $$\text{active\_user\_xor\_sum}[T] = \bigoplus_{C \in \text{Users}(T)} \text{ID}(C)$$
2. **算子执行消费阶段**：当算子 $Op_A$ 执行并消费 $T$ 时，将其自身的 ID 再次异或入状态：
   $$\text{active\_user\_xor\_sum}[T] \leftarrow \text{active\_user\_xor\_sum}[T] \oplus \text{ID}(Op_A)$$

#### 常数时间提取同伴算子

当 $\text{tensor\_out\_degree}[T] == 1$ 时，此前执行过的所有消费者 ID 均因偶数次异或而互相抵消为 0，当前的异或和**恰好就是唯一未执行的同伴算子 ID**：

```cpp
ctx.tensor_out_degree[op_vid]--;
ctx.active_user_xor_sum[op_vid] ^= op_id; // 异或消除自身 ID

if (ctx.tensor_out_degree[op_vid] == 0) {
    current_mem -= values_[op_vid].getSizeMb(); // 彻底释放该张量显存
} else if (ctx.tensor_out_degree[op_vid] == 1) {
    int last_sibling_op_id = ctx.active_user_xor_sum[op_vid]; // O(1) 提取唯一同伴
    if (ctx.is_ready[last_sibling_op_id]) {
        push_ready_op(last_sibling_op_id); // 刷新优先级并推入优先队列
    }
}
```

---

## 4. C++ 状态机实现与数据导向架构

### 4.1 MLIR 风格 Operation 与 Value 解耦拓扑

为严格对齐现代编译器工业级 IR（如 MLIR）的设计标准，计算图拓扑解耦为两类核心实体：

```text
                  MLIR 风格计算图拓扑设计
┌─────────────────────────┐              ┌─────────────────────────┐
│     mlir::Operation     │              │       mlir::Value       │
│       (计算节点)         │              │    (SSA 数据流 / 边)     │
├─────────────────────────┤              ├─────────────────────────┤
│ - operands: [Val_0, ...]│ ───────────► │ - defining_op: Op_0     │
│ - results:  [Val_2, ...]│ ◄─────────── │ - users: [Op_1, Op_2]   │
│ - (不存下游 Op 指针)    │              │ - size_mb: 500 MB       │
└─────────────────────────┘              └─────────────────────────┘
```

- **`Operation`**：维护输入的 `operands` 与产出的 `results`，绝不直接持有任何下游算子指针；
- **`Value`**：作为显式 SSA 数据流，维护其唯一定义者（`defining_op`）与所有使用者（`users` 构成的 Use-Def 链）。

---

### 4.2 面向数据设计与平铺缓存预取优化

#### 只读图与可变调度上下文分离

为了支持只读计算图的跨线程并发共享与算法的高效可重入性，计算图本身保持完全不可变，所有调度状态统一平铺在面向数据设计（DOD / SoA）的 `ScheduleContext` 中：

```cpp
struct ScheduleContext {
    std::vector<int> tensor_out_degree;      // 下标: val_id -> 剩余活跃消费者数量
    std::vector<int> active_user_xor_sum;    // 下标: val_id -> 活跃消费者的异或和
    std::vector<int> operation_in_degree;    // 下标: op_id  -> 剩余未就绪依赖输入数
    std::vector<int> op_versions;            // 下标: op_id  -> 惰性版本号
    std::vector<uint8_t> is_ready;           // 下标: op_id  -> 是否已就绪
    std::vector<uint8_t> is_scheduled;       // 下标: op_id  -> 是否已发射执行
};
```

#### SoA 平铺数组的 CPU 缓存优势

相较于将所有状态封装在每个算子结构体内部的 AoS（Array of Structures）模式，SoA（Structure of Arrays）将同构的整型数组在物理内存中连续排布。当调度循环遍历入度或更新异或和时，CPU 硬件预取器（Hardware Prefetcher）能够以最大带宽将连续数据拉入 L1 Data Cache，完全消除了指针跳跃解引用带来的 Cache Miss。

---

### 4.3 调度状态机核心循环与算法性能实测

在测试用例 [01-peak-memory-dag-scheduler.cpp](01-peak-memory-dag-scheduler.cpp) 中，针对包含 10 个算子、3 条并行扩张与压缩支路的多模态计算图，仿真调度实测数据如下：

| 调度策略体系 | 核心调度算法机制 | 显存峰值（Peak Memory） | 显存削减优化率 |
| :--- | :--- | :---: | :---: |
| **朴素 BFS 拓扑排序** | 基于队列的先来先服务（FIFO） | **2100 MB** | 基准对照（0%） |
| **显存感知贪心调度** | MLIR Use-Def + XOR $O(1)$ 跃升 + 惰性堆 | **1000 MB** | **净削减 52.38% 显存** 🚀 |

```text
【策略 1: 朴素 BFS 调度执行时间线 (峰值 2100 MB)】
Step 4: Op3_Vision_Expand   -> [1000 MB] ####################
Step 5: Op5_Text_Expand     -> [1500 MB] ##############################
Step 6: Op7_Audio_Expand    -> [2100 MB] ########################################## (💥 峰值爆炸)
Step 7: Op4_Vision_Compress -> [1600 MB] ################################

【策略 2: MLIR Use-Def + XOR 显存感知调度时间线 (峰值 1000 MB)】
Step 4: Op3_Vision_Expand   -> [1000 MB] ####################
Step 5: Op4_Vision_Compress -> [ 500 MB] ########## (立即局部释放视觉大张量)
Step 6: Op5_Text_Expand     -> [1000 MB] ####################
Step 7: Op6_Text_Compress   -> [ 400 MB] ######## (立即局部释放文本大张量)
Step 8: Op7_Audio_Expand    -> [1000 MB] ####################
Step 9: Op8_Audio_Compress  -> [ 300 MB] ######
```

---

## 5. 调度算法全景矩阵与工程决策速查

| 调度算法维度 | 朴素 BFS 调度（FIFO） | 深度优先调度（DFS） | 显存感知贪心调度（NetDelta + XOR） | 前瞻加权调度（Lookahead Heuristic） |
| :--- | :--- | :--- | :--- | :--- |
| **就绪队列机制** | 单端 / 双端队列 FIFO | 栈 LIFO | 惰性二叉最小堆（Min-Heap） | 多目标带权优先队列 |
| **显存感知能力** | 无（盲目并发展开） | 弱（依靠拓扑深度） | **极高（实时计算净释放红利）** | **最优（兼顾局部释放与全局残差）** |
| **同伴定位开销** | 不适用 | $O(N)$ 遍历搜索 | **$O(1)$ 异或和极速常数时间定位** | **$O(1)$ 异或和 + 关键路径前瞻** |
| **时间复杂度** | $O(V + E)$ | $O(V + E)$ | **$O((V + E) \log V)$** | **$O((V + E) \log V)$** |
| **编译器适用场景** | 内存充足下的指令级并行 | 语法树简单线性遍历 | **NPU/SRAM 受限端侧图编译器** | **复杂大模型（LLM/Diffusion）显存规划** |
