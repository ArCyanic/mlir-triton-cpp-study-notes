# 计算图 DAG 峰值显存最小化拓扑调度算法

> 本文系统剖析 AI 编译器中**基于活跃期（Liveness）与异或和（XOR-Sum）$O(1)$ 查找的计算图峰值显存最小化拓扑调度算法**。从显存峰值爆炸的根本物理动因出发，对比朴素拓扑排序与显存感知调度的本质差异，系统推导基于 MLIR 风格 Operation/Value 解耦架构下的调度状态机模型。

## 1. 计算图显存峰值问题建模

### 1.1 中间激活张量物理驻留机制

在端侧 AI 芯片（如 NPU、DSP）或片上缓存极其有限的专用硬件中，快速静态随机存取存储器（SRAM）的容量通常仅有数百 KB 到数十 MB。即使在云端大模型（LLM）训练/推理中，单卡高带宽显存（HBM）也是极其昂贵的瓶颈资源。

在深度学习计算图中，一个复杂的模型由数十到数百个算子节点组成。算子在执行时会产生**中间激活张量（Activation Tensors）**。这些张量必须常驻在物理显存中，直到它的**最后一个消费者算子执行完毕**才能被释放：

```text
Op_Produce ──► [ Tensor T (Size = 500 MB) ] ──┬──► Op_Consumer_1 (执行时刻 t1)
                                              └──► Op_Consumer_2 (执行时刻 t2)
                                                   ; 直至 t2 执行完毕，Tensor T 方可释放！
```

若调度顺序不当，大量尚未消费完毕的大尺寸张量将在显存中长期重叠滞留，造成显存峰值（Peak Memory Footprint）急剧飙升，最终导致运行时内存溢出（OOM, Out-Of-Memory）。

### 1.2 拓扑排序等价解空间

有向无环图（DAG）的合法拓扑排序通常**不是唯一的**。对于同一个计算图，可能存在成百上千种完全满足依赖关系的合法执行次序。

不同的拓扑序在算子依赖上是等价的，但在**张量生命周期（Tensor Liveness Intervals）**和**物理显存峰值**上有着巨大差异。调度算法的核心任务是在所有合法的拓扑排序中，寻找一个能够使**全局并发存活张量总和的最大值最小化**的最优序列。

## 2. 经典拓扑调度时空对比

### 2.1 广度优先并发显存膨胀

以一个具有三分支（视觉/文本/音频特征提取）的多模态计算图为例：

```text
       [Init_Vision]       [Init_Text]       [Init_Audio]
            │                   │                 │
      [Vision_Expand]     [Text_Expand]     [Audio_Expand]
            │                   │                 │
     [Vision_Compress]   [Text_Compress]   [Audio_Compress]
            \                   │                 /
             └──────────► [Cross_Fusion] ◄────────┘
```

若采用标准入度 FIFO 队列（广度优先拓扑排序）：
1. 队列会优先把所有分支的第一层算子全量执行（`Init_Vision` $\to$ `Init_Text` $\to$ `Init_Audio`）；
2. 随后执行所有分支的第二层膨胀算子（`Vision_Expand` $\to$ `Text_Expand` $\to$ `Audio_Expand`）；
3. **后果**：三个分支的大尺寸展开张量**在同一时刻全部驻留在显存中**，显存峰值达到各分支峰值的直接累加（如 2100 MB）。

### 2.2 深度优先局部即时释放

若采用显存感知的深度优先/贪心调度：
1. 先执行 `Vision_Expand` 并紧接着执行 `Vision_Compress`；
2. 在进入下一个分支之前，将视觉分支的超大中间张量**彻底消费并释放**，显存回落至基线；
3. 随后再依次展开并压缩文本分支与音频分支；
4. **效果**：多个分支的大张量生命周期完全在时间轴上错开，显存峰值被严格压制在单个分支的局部峰值（如 1000 MB，**节省超过 50% 显存**）。

## 3. 显存感知调度算法模型

### 3.1 算子净显存增量（NetDelta）模型

为了在每一步决策时挑选出对显存最有利的就绪算子，我们定义算子 $Op$ 的**净显存增量（Net Memory Delta）**：

$$\text{NetDelta}(Op) = \sum_{R \in \text{Results}(Op)} \text{Size}(R) - \sum_{O \in \text{Operands}(Op), \; \text{OutDegree}(O) = 1} \text{Size}(O)$$

- **产出开销**：算子执行后新生成的 Outputs 会占用显存；
- **释放红利**：若某个输入操作数 $O$ 当前的**剩余未消费出度恰好为 1**，说明当前算子是该张量的**最后一个消费者**。当前算子执行完毕后该张量将被彻底释放，贡献负向的显存缩减红利；
- **决策规则**：$\text{NetDelta}(Op)$ **越小越优先**（负数表示执行该算子后显存净下降）。

### 3.2 惰性优先队列版本控制

在图的调度过程中，随着其它算子的执行，某张量的剩余出度会动态递减（例如从 $2 \to 1$），导致依赖该张量的下游就绪算子的 $\text{NetDelta}$ 发生动态变化。

标准优先队列不支持低开销的 `decrease-key` 操作。我们采用**版本号机制（Lazy Invalidation via Versioning）**：
1. 每个算子维护一个单调递增的 `op_versions[op_id]`；
2. 当算子的优先级发生跃升时，递增版本号并将新的三元组 `{net_delta, op_id, new_version}` 推入优先队列；
3. 在出堆时，若发现堆顶元素的版本号与算子当前最新版本号不一致，或该算子已被调度，直接 $O(1)$ 抛弃（Lazy Skip）。

### 3.3 异或和 O(1) 同伴算子跃升机制

当一个张量 $T$ 的出度从大于 1 降为 1 时，意味着只剩下最后 1 个活跃消费者。我们需要立即通知该消费者算子，使其 $\text{NetDelta}$ 刷新并跃升到堆顶。

#### 为什么使用异或和（XOR-Sum）？
若通过遍历张量的消费者列表来寻找剩下的那个算子，时间复杂度与出度成正比。利用异或运算的自反性（$A \oplus A = 0$ 和 $A \oplus 0 = A$）：
1. **初始化**：$\text{active\_user\_xor\_sum}[T] = \bigoplus_{C \in \text{Users}(T)} \text{ID}(C)$；
2. **算子执行时**：当算子 $Op_A$ 执行并消费 $T$ 时，执行：
   $$\text{active\_user\_xor\_sum}[T] \leftarrow \text{active\_user\_xor\_sum}[T] \oplus \text{ID}(Op_A)$$
3. **$O(1)$ 提取剩余算子**：当 $\text{tensor\_out\_degree}[T] == 1$ 时，当前的异或和**刚好就是唯一未执行的同伴算子 ID**！

```cpp
ctx.tensor_out_degree[op_vid]--;
ctx.active_user_xor_sum[op_vid] ^= op_id; // 异或消除自身

if (ctx.tensor_out_degree[op_vid] == 0) {
  current_mem -= values_[op_vid].getSizeMb(); // 彻底释放显存
} else if (ctx.tensor_out_degree[op_vid] == 1) {
  int last_sibling_op_id = ctx.active_user_xor_sum[op_vid]; // O(1) 极速提取
  if (ctx.is_ready[last_sibling_op_id]) {
    push_ready_op(last_sibling_op_id); // 刷新优先级跃升
  }
}
```

## 4. C++ 状态机实现及实验评估

在 [01-peak-memory-dag-scheduler.cpp](01-peak-memory-dag-scheduler.cpp) 中，我们构建了纯标准 C++17 的无锁只读图拓扑与面向数据设计（DOD）调度状态机。

### 4.1 MLIR 风格图拓扑及上下文设计

为严格遵循现代编译器工业级 IR 规范，数据结构解耦为两类核心实体：

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

- **`Operation`**：维护输入的 `operands` 与产出的 `results`，不直接持有任何下游算子指针；
- **`Value`**：作为显式 SSA 数据流，维护其定义者（`defining_op`）与所有使用者（`users` 组成的 Use-Def 链）。

为了实现图的只读共享与算法的可重入性，计算图本身为不可变数据，所有调度过程中的状态存储在扁平的 `ScheduleContext` 结构体中：

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

### 4.2 多分支计算图调度收益实测

在包含 10 个算子、3 条并行扩张与压缩支路的多模态计算图测试用例下，仿真运行结果对比如下：

| 调度策略 | 调度核心逻辑 | 显存峰值（Peak Memory） | 显存优化率 |
| :--- | :--- | :---: | :---: |
| **朴素 BFS 拓扑排序** | 基于队列的先来先服务（FIFO） | **2100 MB** | 基线（0%） |
| **显存感知贪心调度** | MLIR Use-Def + XOR $O(1)$ 跃升 + 惰性堆 | **1000 MB** | **节省 52.38% 显存** 🚀 |

#### 执行时间线时空展开对比

```text
【策略 1: 朴素 BFS 调度 (峰值 2100 MB)】
Step 4: Op3_Vision_Expand   -> [1000 MB] ####################
Step 5: Op5_Text_Expand     -> [1500 MB] ##############################
Step 6: Op7_Audio_Expand    -> [2100 MB] ########################################## (💥 峰值爆炸)
Step 7: Op4_Vision_Compress -> [1600 MB] ################################

【策略 2: MLIR Use-Def + XOR 显存感知调度 (峰值 1000 MB)】
Step 4: Op3_Vision_Expand   -> [1000 MB] ####################
Step 5: Op4_Vision_Compress -> [ 500 MB] ########## (立即局部释放视觉张量)
Step 6: Op5_Text_Expand     -> [1000 MB] ####################
Step 7: Op6_Text_Compress   -> [ 400 MB] ######## (立即局部释放文本张量)
Step 8: Op7_Audio_Expand    -> [1000 MB] ####################
Step 9: Op8_Audio_Compress  -> [ 300 MB] ######
```
