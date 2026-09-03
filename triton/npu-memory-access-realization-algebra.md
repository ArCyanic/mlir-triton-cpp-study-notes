# 从布局到访存：面向 NPU 的内存访问实现代数

`tensor<...xptr>` 不是一种等待编译器“恢复几何”的残缺语法。它已经完整表达了程序要访问哪些地址，只是没有承诺这些地址适合哪条硬件通路。把它一律还原成 TensorView，会让规则切片工作得很好，也会让动态索引、循环携带地址、非矩形谓词和原子写在同一个分析器里依次撞墙。问题不在模式覆盖得不够多，而在编译器一直拿一种硬件友好的结果形式充当源程序的语义。

真正的统一模型必须经受一个更有价值的检验：Qwen3.5 和 DeepSeek V4 中同质的行级向量访问、规则 tile 搬运与异步 TMA 可以选择不同机器实现，但共享逻辑效果、realization 和正确性判据。模型边界由这两类 LLM 的交付闭包决定；动态 gather 和原子反向不会仅因 Triton 能表达就进入实现。矩阵核心已有的卷积语义属于真实硬件能力，应保留到目标模型的 short/causal/depthwise conv 需求确认后再收窄，不能把“不追求传统 CNN 兼容”误写成“删除卷积原语”。

本文提出一个仍在接受攻击的答案：**内存访问实现代数**（Memory Access Realization Algebra，暂称 MARA）。它不试图把任意地址表达式压进一个闭合的仿射代数。它统一的是三件更根本的东西：逻辑内存效果如何描述，硬件执行位置如何实现逻辑实例，目标原语如何证明自己保持了程序可观察行为。

这条路线有论文潜力，也有明确的失败条件。若它最后只能给 `vle`、`vlse`、`vluxei` 各写一套反向模式匹配，它就没有形成理论；若它要求先解决一般程序等价性才会工作，它也没有工程价值。理论必须站在这两种失败之间。

## 1. 先放弃一个不可能的承诺

“完备”至少有三种含义。混淆它们，是这类设计最容易产生宏大幻觉的地方。

第一种是**表示完备**：每一个合法的 Triton 内存操作，无论地址来自仿射坐标、前一次 load、循环回边还是指针选择，都能无损进入核心模型。这可以做到。

第二种是**优化判定完备**：对任意 Triton 程序，编译器都能判定某次访问是否始终等价于连续向量、跨步向量、indexed vector 或 TMA。这做不到。地址可以由一般循环计算；判断两个这样的地址程序是否对所有输入相等，已经触及一般程序等价性。一个既有限、闭合、快速，又能完备决定任意 SSA 和控制流的访问代数并不存在。

第三种是**子语言判定完备**：对一个边界写清楚的地址语言，编译器能完整决定它是否满足某个硬件原语。LinearLayout 属于这一类成功。它把问题限制在有限、静态、2 的幂次、GF(2) 线性映射中，于是复合、求逆、求商和秩判定都获得了闭包。它从来没有承诺表达动态内存内容、任意谓词、整数回绕或循环状态。

MARA 的产品目标因此很具体：

- 对 Qwen3.5 / DeepSeek V4 交付闭包中的内存语义表示完备；
- 对生成的机器计划保证 sound；
- 对 RVV 的非负矩形上界子语言和 TMA 的规则 box/gather 子语言给出完备判定；
- 对闭包以外的程序直接返回 unsupported，不把它们变成实现路线；
- 对没有可执行计划的目标明确失败，而不是伪造一个“理论上存在”的慢速兜底。

这是研究问题的第一条边界。越过它，理论会变成无法实现的全程序定理证明器；退回它，设计又会变成算子模式清单。

## 2. 地址映射还不是内存访问

把访问写成从逻辑坐标到字节地址的函数，比 `{origin, stride, bound}` 深了一层，但仍然遗漏了四个会直接决定正确性的事实。

同一个地址可能被多个硬件槽位同时写入。对 load，这可能只是重复工作；对普通 store，它可能形成数据竞争；对 atomic，它会改变更新次数。一个只记录地址集合的模型看不见 multiplicity。

masked load 在禁用槽位上要返回 `other` 或 poison，masked store 则根本不产生内存效果。二者拥有相同的地址和 guard，却不是相同访问。

volatile、atomic ordering、fence 和异步 token 限制事件可否删除、合并或重排。一个没有顺序的地址集合无法证明 TMA pipeline 或 atomic scatter。

TMA 从全局内存搬到 L1，会产生源程序里没有显式出现的内部读写。要求源事件与目标事件逐项相等，会错误拒绝所有经过 scratchpad 的合法实现。

所以核心对象不能只是 AddressMap，也不能只是一张 polyhedral relation。它必须描述一族**带守卫的内存效果**。

给定一个内存操作 $o$，记它的动态逻辑实例集合为 $I_o$，程序状态为 $\Sigma$。状态包含标量环境、已执行 load 的结果、内存内容和相关控制状态。若只把该操作写成从状态到事件标签的函数，load 的返回值和 atomic 的状态更新就无处安放。更准确的语义是一个带事件标签的状态转移关系：

$$
\mathcal A_o
\subseteq
(I_o\times\Sigma)
\times
(\Sigma\times V_o\times\mathcal E_\bot).
$$

$V_o$ 是该操作产生的 SSA 结果；无结果的 store 使用 unit value。$\mathcal E_\bot$ 中的 $\bot$ 表示没有可观察内存事件，例如 mask 为假的 store。关系而不是普通函数允许源语言的 poison、未指定值、atomic 竞争和内存模型产生多个合法后继状态。对定义良好的确定性子语言，它自然退化为函数。

一个效果标签至少包含：

$$
e = \langle
\text{object},\quad
\text{byteOffset},\quad
\text{width},\quad
\text{action},\quad
\text{payload},\quad
\text{ordering}
\rangle.
$$

`object` 保留 pointer provenance，而不是把所有指针降成无来源整数。`byteOffset` 保留源程序的地址计算语义。`action` 区分 load、store、atomic 和 transfer。`payload` 对 load 描述结果与 masked-off 值，对 store/atomic 描述写入值。`ordering` 保存 volatile、scope、memory order 和同步依赖。

guard 决定哪些实例产生可观察效果。为了避免“偏函数掩盖 masked-off 结果”的歧义，实现里更适合显式写成：

$$
G_o : I_o \times \Sigma \to \mathbb B,
\qquad
E_o : I_o \times \Sigma \to \mathcal E.
$$

对于 load：

$$
\operatorname{Result}_o(i, \sigma) =
\begin{cases}
\operatorname{Memory}_\sigma[\operatorname{Loc}_o(i, \sigma)], & G_o(i, \sigma), \\
\operatorname{Other}_o(i, \sigma), & \neg G_o(i, \sigma).
\end{cases}
$$

对于 store 和 atomic，$G_o=false$ 表示没有内存效果。这个差异不能留给 lowering 临时猜测。

单个效果仍然不够。kernel 产生的是带依赖与顺序约束的效果族：

$$
\mathbf{A} = (E_A,\ \prec_A,\ \lambda_A),
$$

其中 $E_A$ 是动态效果实例，$\prec_A$ 是程序和内存模型要求保留的偏序，$\lambda_A$ 给每个实例赋予上面的效果标签。这是 MARA 真正的语义中心，后文称为 **Memory Effect Family**。

状态转移是不可省略的一层。异步 TMA 的 token 改变后续操作何时能够观察 local buffer；若核心只保存 Location 和事件名字，这个关键时序只是被“归类”，没有被建模。

这个修正也推翻了一个看似优美、实则过强的早期等式：目标原语不需要与源程序“逐事件相等”。它只需要在允许隐藏内部事件和允许的重排之后，保持相同的可观察行为。

## 3. 正交性不来自访问类别，而来自三个可组合的态射

Uniform、contiguous、strided、indexed 和 TMA 不构成正交坐标轴。contiguous 是 strided 的特例；规则仿射访问也能用 indexed 指令执行；同一批地址既可能由 RVV gather 完成，也可能由 TMA gather 完成。把它们画成互不影响的维度，会制造并不存在的笛卡尔积。

真正稳定的分解来自三个对象的定义域和值域：

```text
硬件执行位置 H
      │
      │ R：布局、分块、调度、副本所有权
      ▼
动态逻辑实例 I
      │
      │ A：程序定义的内存效果
      ▼
可观察内存行为 E

目标原语 Pθ：H 上的状态转移与目标/内部事件
```

$\mathcal A$ 是程序语义。它不知道 RVV、TMA、VLEN 或 L1 descriptor。

$R$ 是执行实现（realization）。它说明哪个 program、循环迭代、chunk、lane、register slot 或 L1 位置负责哪个逻辑实例。LinearLayout 是其中负责“硬件槽位到逻辑张量元素”的核心映射，但完整的 $R$ 还必须包括：

- program loop 与 chunk loop；
- register group 和 LMUL 分组；
- 非单射布局中的 replica ownership；
- store/atomic 的唯一执行者；
- TMA box 与 local memdesc 的放置；
- 异步发射和 wait 的时间关系。

实践进一步表明 program domain 与 tensor domain 必须用同一套 restriction 语言理解。kernel
入口的 `if pid >= num_programs: return` 不是任意 CFG；它把动态 program 实例族限制在一个域内。
NPU 将 grid 物化为循环后，这个限制应成为循环体内的结构化 `scf.if`，正如 tensor 的矩形
guard 成为 `boundaryCheck`。二者不共享 IR op，但共享一条定律：restriction 必须先作用于
逻辑效果族，域外实例不产生效果，不能等到目标访存时再猜。

实践迫使这里再精确一步：真实 kernel 不是一个孤立访问，而是一张 value DAG。每个 SSA value $v$ 有自己的逻辑实例域 $I_v$，`expand_dims`、broadcast、elementwise 和 reduction 在这些实例域之间建立不同的 correspondence。不能让整张 DAG 共用一张 root realization；那会把 `[4,1]` 的 row coordinate 在 chunk 化时错误膨胀成 root `[4,128]`，随后连一个 `extsi` 都会得到 operand/result shape 不一致。

因此 realization 实际是一族映射：

$$
R_v:H_v\to I_v,
$$

并且每条 value edge $e:u\to v$ 都要与逻辑 correspondence $C_e:I_v\to I_u$ 和硬件 correspondence $\widehat C_e:H_v\to H_u$ 交换：

$$
C_e\circ R_v = R_u\circ\widehat C_e.
$$

这张交换图把几条此前散落的实现规则统一起来：`expand_dims` 插入 singleton coordinate；broadcast 让多个结果实例读取同一个源实例；纯 pointwise 运算使用相同实例；layout conversion 改变 $H_v$ 而不改变 $I_v$。当一个 chunk 在某个轴上只覆盖一个逻辑实例时，`make_range` 经 $R_v$ pullback 后成为 scalar，且对纯 pointwise $f$ 有：

$$
f(\operatorname{splat}(x_1),\ldots,\operatorname{splat}(x_n))
=
\operatorname{splat}(f(x_1,\ldots,x_n)).
$$

这不是普通常量折叠。它是 uniform 子图从 vector realization 回到 scalar realization 的合法换基。

非单射 layout 则给出另一种换基：短逻辑向量在完整寄存器组中产生 replica。坐标生产者必须按
`PhysicalLaneMapping` 生成重复坐标，load 可以扩展逻辑前缀，store 只允许 canonical owner
提交，reduce 先把 owner pack 成前缀再折叠。DeepSeek V4 的 256-expert top-k 证明这四处不能
各自解释 replication：短 `make_range`、float max、signed-index max 和跨 chunk accumulator
只有共享同一个 ownership witness 才会给出一致的 expert index。

### 3.2 Guard 的一般语义与当前 boundary-only 策略

Memory Effect Family 必须保留一般 guard；否则它无法表示源 Triton。实现策略却不必立刻为
每个 guard 发射硬件 mask。当前 Reexen 主线作了更窄、可验证的选择：只有能完整 realization
为 tensor-view `boundaryCheck` 的 guard 才进入结构化访存，memory instruction 只接受
base、stride 和 AVL。这个选择是当前 planner 的 admissible 子语言，不是把一般 guard 从
MARA 语义中删除。

当前 RVV guard 的基本对象是从零开始的逐维前缀：

$$
D_d=[0,u_d).
$$

offset 必须先由范围分析证明非负，`boundaryCheck` 再把 $o_d+i_d<u_d$ 变成 AVL。
这条合同不需要 masked memory、负地址修正或 slide。

`cols >= rows` 不属于这个矩形子语言，因为列下界依赖行实例。坚持 boundary-only 的下一步
不是把它扩大成矩形读取，也不是偷偷保留 residual predicate。真实 Triu 把这一条件用于
寄存器域选择，访存仍只有矩形上界；直接把它用作 load guard 的程序继续拒绝。除非目标 LLM
出现不可改写的真实消费者，否则 dependent interval 不进入实现。

$\mathcal P_\theta$ 是参数化目标原语。参数 $\theta$ 可以包含 base、stride、VL、descriptor、local buffer、token 和 fence。它不是一个操作名字，而是该目标操作的状态转移与事件语义。

这三个部分可以独立演化：更换目标不会改变 $\mathcal A$，重新选择 layout 不会重写源地址语义，新增一条硬件指令只需给出新的 $\mathcal P_\theta$。它们通过类型正确的函数或关系复合连接。这是比“地址轴、掩码轴、时间轴”更坚实的正交性：不是宣称所有属性互不约束，而是让每种知识只拥有一个语义位置。

严格地说，这也不是三个集合的任意笛卡尔积。store 的效果会约束 $R$ 必须选择唯一 owner，TMA 的目标语义会约束 local placement，volatile 会约束调度。**正交分解不等于独立可任意组合；它意味着耦合只通过显式的组合与证明义务发生。**

### 3.1 “代数”必须由运算和定律挣来

到这里为止，MARA 还只有语义对象与模块分工，尚不配称为代数。若要让这个名字成立，Memory Effect Family 至少需要以下运算：

- **重索引 / pullback** $R^*\mathbf{A}$：用硬件或子块实例映射 $R:H\to I$ 重新索引逻辑效果，即在新的实例空间上观察同一语义；
- **限制** $\mathbf{A}\vert_G$：只保留 guard $G$ 为真的效果；
- **带标签并合** $\mathbf{A}\uplus\mathbf{B}$：合并身份互异的效果实例，但不假设它们的 Location 不重叠；
- **顺序组合** $\mathbf{A};\mathbf{B}$：将前者的状态后继和 SSA 结果送入后者；
- **并发组合** $\mathbf{A}\parallel_M\mathbf{B}$：按内存模型 $M$ 生成允许的交错与同步约束；
- **内部事件隐藏** $\operatorname{hide}_S(\mathbf{A})$：隐藏属于寄存器、L1、descriptor 配置等实现私有集合 $S$ 的事件；
- **行为 refinement** $\mathbf{A}\sqsubseteq\mathbf{B}$：$\mathbf{A}$ 的每个可观察行为都被 $\mathbf{B}$ 允许。

这些运算必须满足可以被实现和测试的定律。例如 pullback 应具备函子性：

$$
id^*\mathbf{A}=\mathbf{A},
\qquad
(R_1\circ R_2)^*\mathbf{A}
=
R_2^*(R_1^*\mathbf{A}).
$$

限制应与重索引交换：

$$
R^*(\mathbf{A}\vert_G)
=
(R^*\mathbf{A})\vert_{G\circ R}.
$$

带标签并合上的重索引应满足分配律，顺序组合应在观察等价意义下满足结合律，隐藏应保持 refinement：

$$
\mathbf{A} \sqsubseteq \mathbf{B}
\Longrightarrow
\operatorname{hide}_S(\mathbf{A}) \sqsubseteq \operatorname{hide}_S(\mathbf{B}).
$$

并发组合不能退化为普通集合并。两个 program 的效果身份彼此不同，却可能访问同一 Location；atomic、barrier 和 data race 的允许行为都由 $M$ 决定。若 MARA 只会组合无 alias 的访问，它无法描述 embedding backward、跨 program reduction 或程序网格串行化。

这些不是装饰性的公式。第一条保证 program layout、chunk layout 和 register layout 可以分层复合，而无需为每种组合重写分析；第二条说明源 mask 经过 layout 后应机械地成为 slot predicate；隐藏的单调性允许 TMA 引入并隐藏真正私有的 L1 事件，同时仍沿用同一个正确性关系。

若实现无法让这些定律在小模型测试中成立，MARA 就应该改名为“访问规划框架”，不应再宣称形成了代数。

## 4. LinearLayout 与整数地址没有“代数域冲突”

LinearLayout 可以写成：

$$
L:H_{GF(2)}\to I_{coord}.
$$

访问定位函数则是：

$$
\operatorname{Loc}:I_{coord}\times\Sigma\to Location.
$$

二者的复合 $\operatorname{Loc}\circ L$ 完全合法。函数复合只要求前一个映射的输出能作为后一个映射的输入，不要求两边共享相同的加法。GF(2) 负责计算逻辑坐标比特；这些比特被解释成普通有界整数坐标后，再进入带进位的地址算术。

真正的困难不是“不能复合”，而是复合后不一定仍在某个简单代数内。带 swizzle 的 $L$ 与整数仿射 $\operatorname{Loc}$ 复合后，物理 lane 到地址的函数可能不再是整数仿射。MARA 不要求闭包。它把“复合后的函数是否属于某个目标原语可执行的子语言”变成一个证明问题。

这个取舍与 LinearLayout 根本不同。LinearLayout 的力量来自一个可计算表示上的闭包；MARA 只在关系语义层对重索引、组合、限制和隐藏闭合，它的高效证明子语言不承诺闭包。若试图让地址证明语言也拥有单一代数闭包，就只能排除 data-dependent index、任意 guard 和控制流，最后退回 TensorView。

## 5. 正确性不是事件相等，而是观察等价下的 trace refinement

设源 Memory Effect Family 产生的可观察 traces 为 $\operatorname{Traces}(\mathbf{A})$。目标计划由执行实现 $R$ 和目标原语程序 $\mathbf{P}_\theta$ 构成。目标可能产生寄存器搬运、L1 读写、descriptor 配置和 token 操作；其中一部分是实现内部事件。执行实现 $R$ 同时提供从目标 slot/event 到源逻辑实例的对应关系，由它诱导出的观察映射记为 $\operatorname{Obs}_R$。该映射隐藏已经证明不可被外部观察的事件，并把聚合的向量或 TMA 事务解释为相应的逻辑效果族。

“内部”是一条证明结论，不是 memory-space 枚举。一个 non-escaping local buffer 的中间写可以隐藏；能够被另一核心 remote-L1 通路访问、通过 alias 暴露或跨异步边界泄漏的 L1 事件不能隐藏。把所有 L1 操作一概标成 $\tau$，会让错误的 TMA 重排轻易通过 refinement。

正确 lowering 的核心判据应写成：

$$
\operatorname{Obs}_R(\operatorname{Traces}(\mathbf{P}_\theta))
\subseteq
\operatorname{Traces}(\mathbf{A}).
$$

这里使用 refinement 而不是机械相等，因为源语言中的 poison、未指定 lane 顺序和数据竞争可能允许多个行为，目标只需不产生源语义之外的新行为。若所讨论的子语言没有未定义或未指定行为，可以强化为观察等价。

早先写成 $\operatorname{Traces}(\mathbf{P}_\theta)\subseteq \operatorname{Traces}(\mathbf{A}\circ R)$ 并不严谨。$R$ 是目标事件与源实例之间的 correspondence，不应通过简单预复合把 replica 和内部事件强塞回源语义；coverage、ownership 和 event aggregation 正是需要证明的内容，不能预先假定在 $\mathbf{A}\circ R$ 中已经正确。交换图的正确方向是：

```text
Target execution H ── Pθ ──► target traces
       │                          │
       │ R                        │ ObsR / hide internal
       ▼                          ▼
Logical instances I ── A ──► allowed source traces

右侧路径产生的行为必须 refinement 下侧路径。
```

这个式子是理论的主心骨，但不能直接作为实现算法。它必须分解成局部可证明的义务：

| 证明义务 | 必须排除的错误 |
|---|---|
| Coverage | 漏掉一个 guard 为真的逻辑效果 |
| Exclusion | 对 guard 为假的实例产生额外可观察访问 |
| Location | 访问了错误 object、offset 或 width |
| Payload | load 的结果、store/atomic 的值或 masked-off 值错误 |
| Multiplicity | replica 导致 store/atomic 重复执行或 load 结果缺失 |
| Ordering | 破坏 volatile、atomic、fence、token 或依赖要求 |
| Placement | 目标值没有落到消费者所声明的硬件布局位置 |
这些义务比 `proveStrided`、`proveIndexed` 更基础。一个硬件原语可以通过不同方法完成同一义务；一个证明器也可以同时服务多种硬件原语。新增 primitive 不应改变这张表，只会改变如何构造 witness。

语义正确之外还有两个刻意分开的判断。**Target admissibility** 检查 index EEW、对齐、VL、descriptor、memory space、L1 容量和寄存器约束；不满足意味着计划在该机器上不存在。**Profitability** 比较吞吐、setup、spill 和 overlap；不满足只意味着不该选择这个正确且可执行的计划。把三者混在一个 `isSupported()` 中，会让“证明失败”“硬件不收”和“性能不好”重新变成同一种模糊错误。

这也是当前理论最有可能形成论文贡献的地方：把向量 LSU、显式 scratchpad DMA 和异步 TMA 放进同一个效果 refinement 框架，而不是仅统一它们的地址公式。

### 5.1 三个必须真正证明的命题

第一条是**语义嵌入定理**。对形式化范围内的每一个 Triton 内存操作及其组合，存在一个 Memory Effect Family，其状态转移与源语言操作语义双模拟。这个命题应通过对源 IR 操作和控制流组合做结构归纳证明。它保证“表示完备”不是因为模型保留了一个万能 `opaque` 字段而获得的空洞胜利。

第二条是**局部实现健全性定理**。若 correspondence $R$ 与 primitive program 满足 Coverage、Exclusion、Location、Payload、Multiplicity、Ordering 和 Placement，那么：

$$
\operatorname{Obs}_R(\operatorname{Traces}(\mathbf{P}_\theta))
\subseteq
\operatorname{Traces}(\mathbf{A}).
$$

这条定理把 checker 的局部结果接到真正的编译正确性上。若这些义务不足以推出 refinement，就必须补充义务，不能用更多测试替代证明。

第三条是**受限判定定理**。对静态有限 layout、边界明确的 loop-free address calculus 和一个参数已给定的 primitive，局部实现义务可归约为可判定的逻辑公式；若 primitive 参数语法和候选数也有限，则 synthesis 可由有限枚举加验证决定。这个命题不会覆盖一般循环，却为最常见的 tile 内访问划出一块拥有终止性与完备判定的坚硬内核。

目前三条都只是待证明命题。源 Triton memory model 尚未固定，局部义务的充分性尚未证明，可判定子语言及其编码也未构造。把它们明确写成 proof obligations，比在摘要里提前称为“统一定理”更接近论文工作真正开始的位置。

## 6. 目标原语是语义模板，不是算子模式

一条 unit-stride RVV load 可以定义为一个参数化效果族：

$$
P_{vle}(s) = \operatorname{Load}(\text{base} + s \cdot \operatorname{sizeof}(T)),
\qquad 0 \le s < vl.
$$

`vlse` 将 offset 改为 $base+s\cdot stride$。TMA box 的实例空间是 descriptor 给出的多维 box；TMA gather 的一维索引来自 L1 index buffer，剩余维度来自 descriptor。

规划器面对的统一问题是：能否求出某个 $\theta$，让该 primitive 或 primitive 组合满足上一节的全部 refinement obligations？

这仍然需要枚举 `vle`、`vlse`、`vluxei` 和 TMA，因为 ISA 本来就是有限原语集合。理论统一从来不意味着抹掉 ISA。它意味着枚举的是**目标语义模板**，而不是：

```cpp
if (looksLikeSoftmax(...)) ...
if (offsetProducerIsLoad(...)) ...
if (maskLooksLikeTriu(...)) ...
```

连续性也不再是某个 parser 给地址贴的标签。设某个原语对其 slot 定义了后继关系 $\operatorname{succ}_P$，复合后的定位函数为：

$$
F(h,\sigma)=Loc(R(h),\sigma).
$$

unit-stride 的证明目标是：

$$
F(\operatorname{succ}_P(h), \sigma) - F(h, \sigma) = \operatorname{sizeof}(T).
$$

constant-stride 把右侧换成一个与 $h$ 无关的符号 $s$。uniform 则要求 $F$ 对相关 slot 坐标不变。使用原语自己的后继关系比笼统写 `Address(lane + 1)` 更准确：一个 RVV group 的物理枚举可能跨 register slot，TMA box 更不是单一 lane 轴。

indexed 不是函数类别。只要能够找到公共 base，并证明每个 slot 的剩余 offset、guard 和 provenance 满足 indexed primitive，它就是合法候选。一个规则仿射访问也能满足 indexed 模板，只是成本通常高于 `vle` 或 `vlse`。

## 7. 循环边界不是地址分析的断崖

循环携带值最容易诱使实现增加 `parseLoopIterArg`。那条路把控制流语法写进每一种数据分析，最终每个 dialect 的 `for`、`while`、`if` 和 yield 都会拥有一套手工恢复逻辑。

Memory Effect Family 把循环迭代放进动态实例，把循环状态放进 $\sigma$。设第 $k$ 次迭代的当前基址为一个暂时无法求闭式的状态函数 $p_k(\sigma)$：

$$
F(k,s,\sigma)=p_k(\sigma)+s.
$$

为了证明单次迭代内部可以用 `vle`，根本不需要先证明 $p_k=p_0+k\cdot step$。只需证明 $p_k$ 不依赖当前 vector slot：

$$
F(k,s+1,\sigma)-F(k,s,\sigma)=1.
$$

这是一条重要的分层原则：**局部向量化只证明 slot-relative geometry；跨迭代合并、指针外提和 TMA pipeline 才要求 recurrence closed form。** 一个不透明但 lane-uniform 的循环状态不应阻断当前迭代的规则访问。

控制流分析仍然有工作。若 loop-carried tensor 的每个 lane 各自推进，状态函数本身依赖 slot；若分支选择不同 allocation，common-base 可能不存在；若 while 的退出依赖 load，跨迭代变换可能需要不现实的全程序证明。统一理论消除了错误的边界，不会凭空消除这些真实困难。

## 8. Guard 与 ownership 必须分开

这是只看地址和 mask 时极易漏掉的一条正交关系。

逻辑 guard 回答“这个逻辑元素是否应产生内存效果”。ownership 回答“当布局把同一个逻辑元素复制到多个硬件槽位时，哪个槽位负责提交效果”。二者都会表现成机器 predicate，却来自完全不同的语义层。

设 LinearLayout 或更完整的 realization 不是单射，同一逻辑实例 $i$ 对应多个硬件槽位。对于 load，多个槽位重复读取在非 volatile、无竞争条件下可能合法，并且可避免额外 broadcast。对于 store，若这些槽位写入同一值，结果有时表面相同，但源程序并没有授权多次可观察写；对于 atomic add，重复两次会直接改变结果。

因此目标活动谓词通常是：

$$
\operatorname{Active}_P(h) =
G_A(R(h))
\land \operatorname{Owner}_R(h)
\land \operatorname{ResourceGuard}_P(h).
$$

load 可以在证明允许复制时放宽 `Owner`；store、atomic 和 volatile 默认不能。LinearLayout 的非单射与伪逆能力提供了选择 canonical owner 的基础，但选择结果必须作为 realization witness，而不是偷偷塞进访存 mask。

这一点同时解释了 stride-0 维为什么不能参与“最快内存维”排序。stride-0 表达的是逻辑值复制或地址不随某个坐标变化，不是内存遍历优先级。把 replica geometry 和 memory order 混在一个 `inferOrder()` 中，必然得到错误结论。

## 9. 一个证明问题可以有多个决定过程

统一理论并不要求一个万能正规化器。那种正规化器会迅速变成另一份 SSA IR：既要支持 add/mul，又要支持 div/rem、select、bitwise、load、pointer provenance 和循环 phi；任何未覆盖节点又会掉进 opaque residual，吞掉整条表达式的优化机会。

更可靠的实现是按目标义务发起查询，在共享的精确语义上使用多个 sound 决定过程：

```text
原始 SSA
      │
      ├── 依赖/一致性：表达式是否依赖某个 hardware slot？
      ├── provenance：所有位置是否属于同一 memory object？
      ├── GF(2)：layout 坐标如何由 slot bits 产生？
      ├── Presburger：index/no-wrap 子语言中的线性约束是否成立？
      ├── quasi-affine：常量 floordiv/mod 能否消去或分段？
      └── memory/alias：load-derived index 与效果重排是否安全？
```

这些求解器不直接生成 `vle` 或 TensorView，也不各自产生一份互相竞争的访问分类。它们共同回答同一种 refinement obligation。增加一条 `rem` 规则提升的是证明覆盖率，不会改变核心语义或目标 primitive 的接口。

为了控制编译时间，符号项应按需构造、hash-cons，并设置明确预算。常见路径先走结构化快速证明；只有失败且候选收益足够大时才进入 Presburger 或 SMT。预算耗尽返回 unknown，绝不能把“没证明出来”当成“不成立”。

一个值得追求的可判定子语言是有限静态 tile 上的 loop-free address expression：常量、逻辑坐标、uniform symbols、线性算术、常量 `floordiv/rem` 和由这些表达式构造的 predicate。将 layout 的静态 GF(2) 坐标函数代入后，**给定参数的 primitive 验证**可以归约为反例可满足性。证明等价就是证明“存在反例”的公式不可满足。这个子语言有限、可判定，而且已经覆盖大量深度学习 kernel 的地址骨架。

这里不能偷换成无限制 synthesis。验证已经给定的 `base/stride/vl/mask` 是一个判定问题；从任意 SSA 中发明这些参数是一个存在性搜索。只有当参数来自有限候选集、受限语法或可直接求解的 affine 系数时，synthesis 才继承可判定性。实用系统可以先结构化提取候选，再用统一 checker 验证；候选生成允许不完备，正确性不依赖它完备。

load-dependent index 不需要假装仿射。它可以作为 array read 出现在公式中；`vluxei` 的目标语义含有同一个 index value，于是等价证明仍可能成立。真正困难的是证明 index 范围、不同 base provenance 和并发更新，这些应成为独立义务。

## 10. Layout 选择是全图问题，不属于单个访问

一个中间 tensor 可能同时连接 row-contiguous load、column reduction 和 transposed store。三方偏好的布局相互冲突。让每个访问局部选择“最适合自己的 LinearLayout”，只会在边上制造昂贵的 `convert_layout`，或者得到根本不一致的类型。

这不是 Memory Effect Family 的漏洞，而是 realization 选择必须提升到图级的证据。对每个 SSA tensor value 选择布局 $L_v$，对每个操作选择计划 $P_o$，整体目标近似为：

$$
\min_{\{L_v\}, \{P_o\}}
\sum_o \operatorname{Cost}(P_o, L_{\text{operands}}, L_{\text{results}})
+
\sum_e \operatorname{ConvertCost}(L_{\text{src}}, L_{\text{dst}}).
$$

全局最优通常不值得也未必能够廉价求出。编译器可以使用有限候选、greedy propagation、动态规划或 bounded search。理论要求的是每个候选计划都经过 refinement 检查，不是要求 cost model 成为完美预言家。

当前 NPU 后端若只有固定 row-major blocked layout，可以把 $R$ 当作已给定，只做目标计划选择。这是合理的第一阶段，却不能成为终极模型的公理。否则 column reduction、transpose 和多输入融合永远被一个历史默认值支配。

这里还暴露出一个尚未解决的研究问题：layout 决策改变 $R$，$R$ 又改变哪些 primitive 能通过证明，primitive 的成本反过来影响 layout。MARA 给出了这个循环依赖的共同目标函数，却没有自动给出高质量求解算法。论文若只停留在局部访问 planning，会避开最难、也最可能产生真正贡献的一半问题。

## 11. 真实案例不是分类演示，而是同一判据的不同求解结果

### Softmax 与 RMSNorm

逻辑地址包含一个对 vector slot 一致的行基址，以及沿列坐标变化的项。把 NPU layout 代入后，若连续 slot 的 Location 差为元素宽度，矩形上界又能被 AVL 精确承载，`vle` 模板通过全部义务。

### Triu

真实 FlagGems Triu 的 diagonal 条件已经位于 `where`，访存 guard 只是 row/column 边界。它需要解决的是 replica-aware realization，不需要发明“斜向 mask 剥离”。实践已经验证：让每个 SSA value 拥有自己的 correspondence 后，原始 2D 与 batch kernel 均逐位正确；让 LLVM 保留最内层物理 chunk loop 后，整组 RVV spill 消失。前者验证语义映射，后者说明 realization witness 还必须跨越优化层，不能在 TTNPU→LLVM 之间丢失。

若另一个 kernel 把 diagonal predicate 直接用作 load guard，当前 boundary-only planner 会拒绝它。这是产品边界，不是等待用 base/AVL/slide 补齐的缺口。只有 Qwen3.5 / DeepSeek V4 出现必须这样访存的真实 kernel 时，才重新决定它归 TMA 还是扩展新的 boundary 表示。

### RoPE：indexed memory 可以退化为连续访存加值域置换

Qwen 风格 RoPE 会把同一个半维 cos/sin 表读两遍：逻辑地址是
`base + (lane mod half)`，直接把它当 indexed memory 会要求 RVV gather load，既违背当前
“memory mask 全部进入 boundaryCheck”的约束，也把一个很小的寄存器重排错误地推给 LSU。
实际可行的 realization 是：先连续 load 唯一的 `half` 元素，由 blocked layout 展开 replica，
再用 `tt.gather` 在寄存器域完成周期置换。原生 FlagGems RoPE 的 F32 与 BF16 Q/K 都已在
QEMU 上逐位一致。

这次 BF16 失败进一步澄清了 target primitive 的语义。`vrgather` 搬运的是 SEW 位模式，
不是某一种数值；rx-LLVM 18 却不能直接为 BF16 intrinsic 选指令。因此 periodic gather、
普通 `tt.gather` 与 short-load replica expansion 不能各自补 BF16 分支，它们必须共享一个
register-gather realization：浮点向量先 bitcast 为同宽整数，i32 逻辑坐标在 SEW16 时用
`vnsrl.wi 0` 缩窄，执行整数 `vrgather.vv` 后再恢复数值类型。该实现同时覆盖 F32、BF16、
显式 gather 与隐式 replica，不再让 producer 各自拥有一套置换规则。

### 融合 QNorm+RoPE：计算域不等于副作用域

DeepSeek V4 融合 kernel 对完整 512 维 Q 做 RMSNorm，但 NoPE store 只提交前 448 维，最后
64 维由 RoPE 的偶/奇 store 接管。旧 chunk pass 要求链上所有 view 的 shape、boundaryCheck
和 padding 形式相同，因而把这个合法程序拒绝了。实践说明这不是缺一条 shape pattern，
而是模型把 value defined-domain 与 memory effect-domain 合并错了。

修正后的义务是集合包含：无界 load 和带 padding 的有界 load 定义完整逻辑值；未 padding
的有界 load 只定义自己的 boundary domain；store 的 effect-domain 必须包含于每个实际输入
的 defined-domain。每个 load 保留自己的 boundaryCheck，store 也只携带自己的 boundaryCheck。
于是 `compute[0:512] -> store[0:448]` 自然成立，而 `load[0:n] -> unbounded store` 仍被拒绝。
这条 containment 关系取代了 equal-shape、equal-boundary 与 equal-padding 三组 Legacy Logic。

一个最小的 out-of-place BF16 `RMSNorm[512] -> store[448]` 已在 QEMU 正确；完整 DSV4 kernel
的 KV RoPE/cache insert 也逐位正确。完整 kernel 的 Q 原地写回目前仍在 QEMU 上出现不稳定的
归约比例，而生成的 TTNIR、LLVM IR 和汇编都保持“完整归约先于写回”的顺序。这个现象应继续
沿 memory ordering / runtime execution 证据诊断，不能倒逼 domain 模型恢复错误的等形限制。

### KV cache：坐标正规化决定 RVV/TMA 边界

FlagGems `reshape_and_cache_flash` 把连续 head 坐标写成
`(i // head_size) * head_size + (i % head_size)`。`head_size` 是运行时标量，通用 canonicalize
没有消去它，旧地址恢复器因看到 `divsi` 将连续 store 误判为 indexed。这里需要的不是
cache 算子 pattern，而是 quasi-affine 坐标恒等式：在 signed div/rem 均有定义的输入上，
`q*d+r=i`。将这个等式放进地址 normalization 后，原生 BF16 flash-cache 的 key/value 与
守卫区均在 QEMU 上逐位正确。

同一组 cache 算子也画清了 RVV 与 TMA 的界限。`concat_and_cache_mla` 的两个 section 各自是
uniform cache base 加连续行，已经由 RVV 正确执行；flash layout 在商余正规化后也是连续行。
传统 blocked `reshape_and_cache` 则把 flat head 数据真正散布到
`[head_size/x, block_size, x]` 和 `[head_size, block_size]`，其 lane-relative location 并非一条
连续或常 stride run。这部分应交给 TMA descriptor/box realization，而不是为了让 RVV 覆盖
所有 cache layout 引入 masked scatter。

### 跨归约存活：宽值必须成为 staged realization

FlagGems `fused_add_rms_norm` 把 F32 中间值 `x+r` 同时用于三件事：写回 BF16 residual、
计算标量方差、以及方差归约后的归一化。单独规划每个 store/reduce 会在后续 stage 重新遍历
producer DAG；此时 residual 已被早期 store 改写，所谓 rematerialization 实际变成再次相加。
这不是 replica 或 gather 问题，而是值的生命周期跨过了 scalar/effect barrier。

现在同一 realization 显式生成三个 stage：第一阶段归约原始 F32 sum；第二阶段重新计算一次
sum，同时将完整 F32 位值写入 program-private scratch、将转换后的值写入公开 residual；第三
阶段从 scratch 恢复精确值并完成归一化。scratch store 与公开 store 先共享 chunk value，后
统一提交 effects，因此不存在一个 store 改写另一个 store 的输入。Reexen 将 TTNPU 的
`global_scratch_alloc` 接到隐藏 runtime 参数；scratch 按物理 core 切片，同一 core 顺序执行的
逻辑 program 可以复用它。未来换成 L1/TMA placement 时，staged realization 本身不变。

这次实测还揭示了更底层的 ownership 缺口：设备会在八个物理 core 上启动同一 kernel，旧
`MaterializeProgramLoops` 却让每个 core 遍历完整逻辑 grid。out-of-place kernel 多次写同值，
长期掩盖了问题；原地 add 明确表现为一次 launch 累加八次。现在有 program-id 的最低维 grid
循环采用 `core_id; step=core_num` 分片，没有 program-id 的 singleton domain 只由 core 0
执行。原地 add 从 10 恢复到 3，原生 BF16 fused-add RMSNorm 与完整 DSV4 QNorm+RoPE+KV
insert 随之逐位正确。物理 core ownership 因此属于 program-domain realization，不属于某个
算子的访存或 replica 策略。

### Embedding 前向

row index 来自一次 scalar load，但对当前 vector slot 一致。MARA 不会因为地址“依赖内存”就把整个访问归类为 indexed；slot-relative difference 仍可证明为 unit stride，因此 row load、scalar base update 和 `vle` 自然组成合法计划。

### TMA

TMA 不是“大号 TensorView”。它引入全局到 L1 的内部 transfer、descriptor 可表达域、local placement、异步 token 与 fence。目标 refinement 必须隐藏内部 L1 事件，同时证明消费者读取的 local tensor 与源逻辑 load 相同，并保持 wait 前后的时序。仅凭 `Base + Strides + Box` 无法覆盖这条证明链。

## 12. 这套理论现在仍有九个硬缺口

一篇理论文档最没有价值的写法，是把未解决问题藏进“未来工作”。以下缺口任何一个处理不好，MARA 都可能退化为漂亮外壳。

### 内存模型尚未形式化

普通 global load/store 在 Triton、MLIR 和目标 NPU 上究竟允许怎样的 lane 间重排？data race、volatile、atomic scope 和异步 fence 的语义边界必须写清。没有正式 memory model，trace refinement 只能是一句方向正确的话，无法成为定理。

### Dynamic instance 的身份尚未固定

program loop、source `scf.for`、chunk loop 和 layout slot 都可能复制、合并或重排实例。必须定义稳定的逻辑 event identity，否则 coverage 和 multiplicity 无从比较。简单用 SSA op 加 tensor index 可能不足以区分循环动态实例和分支路径。

### Pointer provenance 不能只靠公共整数 base

两个数值相同的地址未必来自可互换的 allocation；pointer select 和 tensor-of-pointers 还可能让每个 slot 指向不同对象。MARA 需要明确的 object identity 与允许的 pointer arithmetic，不能在进入 LLVM 前就把一切折成整数。

### Poison 与越界行为会改变 refinement 方向

masked-off load 的未定义 lane、Triton 的 padding、MLIR poison 与 LLVM poison 不是同一个模糊概念。目标代码可以选择源允许的一个行为，但不能让 poison 意外控制地址、mask 或分支。需要为 undefined/poison 单独定义观察关系。

### 目标原语的形式语义成本很高

给 `vle` 写效果模板容易；给 TMA gather、在线 transpose、layout transform 和异步 completion 写出可信语义明显更难。若 primitive semantics 最后只是 C++ lowering 代码的非正式注释，refinement checker 就失去权威来源。

### 私有事件隐藏尚未获得逃逸证明

TMA 使用的 L1 buffer 只有在不逃逸、没有 remote observer、alias 受控且 wait/fence 完整时才能从观察中隐藏。内部 memory space 不是天然不可观察。没有一套跨 memdesc、remote-L1 和 token 生命周期的可见性分析，$\operatorname{Obs}_R$ 仍可能删除本应保留的行为。

### Primitive validation 不会自动产生 synthesis

SMT 能验证一个候选，不代表它会廉价找到那个候选。base factorization、stride 提取、guard 分解和 TMA box 选择需要有限语法、代数求解或 CEGIS。若候选生成重新依赖不断增长的 SSA pattern，系统会在 checker 前面复活旧架构；若直接做无界合成，编译时间又无法控制。

### 编译时间可能吞掉收益

GF(2) substitution、Presburger、alias 与图级 layout search 同时存在，最坏情况会迅速失控。理论必须给出 query budget、缓存边界和 fast path 命中率；论文也必须报告 compile-time tail latency，而不是只展示成功 kernel 的运行性能。

### Cost model 仍可能选择合法但糟糕的计划

soundness 只保证结果正确。RVV 与 TMA 的真实成本受 VLEN、LMUL、descriptor setup、L1 容量、banking、spill 与 pipeline overlap 共同影响。当前不需要在任意原语间做全局搜索：小型行级链固定归 RVV，大块规则搬运固定归 TMA；cost model 只在各自闭包内选择 chunk 或 box。

这九个缺口里，前七个决定理论是否成立，后两个决定理论是否值得使用。它们不能通过给 `AccessSummary` 多加几个 enum 解决。

## 13. 一条可证伪的实施路线

MARA 不应直接替换现有编译流水线。第一步应只建立观察能力，不改写任何访问：从 TTIR 构造 Memory Effect Family 的逻辑身份、guard、Location、payload 和 effect，打印或序列化出稳定结果。相同访问经过 canonicalization、CSE、loop unroll 前后，应映射到观察等价的逻辑效果。

第二步给少量 RVV primitive 写可执行语义模板和本地 refinement checker，只覆盖
unit-stride、constant-stride 和非负逐维前缀 domain。当前实现坚持把这类 guard 全部
realization 为 boundaryCheck/AVL，不接一般 masked memory。生成的计划与现有 TensorView
路径并行比较，但不形成永久双轨；一旦语义、性能和编译时间证据齐全，就删除旧恢复器。

实践已经完成其中一块结构性工作：`ValueRealization` 成为独立模块，统一拥有 producer DAG
分类、per-value root correspondence、chunk type 投影与 uniformity 证明；原先散落在
`MaterializeChunkLoops` 的 `coversSameElements`、共享 `rootDims` 和 `chunkIsSplat` 已删除。
这不是完整 refinement checker，却建立了 checker 需要的 realization seam。BF16 Triu 又验证了
同一 correspondence 必须贯穿 slot 决策：i32 坐标的 `expand_dims` 两侧要跟随 BF16 消费链的
128-slot realization，而不能各自按元素宽度选布局。

第三步直接接 TMA。TMA 要求 memory spaces、placement、internal event hiding、token 和 fence
一起进入 refinement，并承接 attention tile、KV box 和 MoE bulk movement。RVV indexed、
负 offset 与多-run 拼接不再排在它前面。

第四步只在目标 LLM 暴露实际冲突时做有限的图级 layout 选择，并把 convert cost 纳入目标函数。
固定布局已经满足的行级链不为理论完整性支付搜索成本。

每一步都要用两类测试反驳自己。一类是当前 LLM 子语言的小规模 property test：枚举短行、矩形尾块和 replica，比较源与 primitive 的效果 traces。另一类是真实 Qwen3.5 / DeepSeek V4 形态的 kernel：保留原源码，按数值、越界、汇编、寄存器、spill 和编译时间共同验收。传统 CNN 兼容矩阵与一般 mask 穷举不提供产品证据；目标模型真实使用的卷积形态除外。

## 14. 顶级论文的门槛

目前这套模型还不是论文结果，而是一条有资格继续投入的研究假设。要达到顶级编译会议的标准，至少需要四项站得住的贡献。

第一项是形式化。必须给出源 Triton 子语言、Memory Effect Family、realization、目标 primitive 和观察关系，并证明局部 obligations 足以推出 trace refinement。最有价值的定理不是“所有访问可以分类”，而是：

$$
\begin{aligned}
& \text{Coverage} + \text{Exclusion} + \text{Location} + \text{Payload} + \text{Multiplicity} \\
& + \text{Ordering} + \text{Placement} \Longrightarrow \text{Observable Trace Refinement}
\end{aligned}
$$

第二项是可判定子语言。需要明确一个实际覆盖率高的 address calculus，分别证明 candidate validation 和受限 primitive synthesis 的可判定性，并给出复杂度或至少严格终止性。Presburger、GF(2) layout 与其他求解器的组合边界必须精确定义，不能笼统写“交给 SMT”。

第三项是系统。它必须用一个共享 checker 生成或验证 LLM 闭包内的 RVV contiguous/strided 和 TMA 计划，并真正删除重复的 pointer/mask 恢复分支。总代码减少不是论文定理，却是架构统一最有力的工程证据之一。

第四项是评估。至少应回答：真实 Triton/FlagGems kernel 的覆盖率提高多少；产生的访问计划与专家代码相比性能如何；错误计划是否被 exhaustive checker 捕获；编译时间增加多少；graph-level layout planning 相比固定 row-major 在哪些融合 kernel 上产生决定性收益。

论文的新颖性仍需经过严肃 related-work 调研。多面体 access relation、MLIR affine/value-bounds、LLVM ScalarEvolution、e-graph、SMT translation validation、vector transfer、GPU layout algebra、Halide/Tiramisu/TVM schedule 与 superoptimization 都覆盖了其中一部分。真正可能的新意不是“地址可以写成关系”，而是：

> 将张量 layout、动态执行 realization、带 guard/ownership/order 的内存效果，以及 SIMD LSU 与异步 TMA 的目标语义，统一为可合成并可验证的实现问题。

如果 related work 已经完整覆盖这句话，就不该包装成新理论。如果它们只分别解决 layout、polyhedral access、vectorization 或 DMA scheduling，MARA 才可能拥有清晰的论文位置。

## 15. 最终立场

这项工作不应追求一个“比 LinearLayout 更大的 LinearLayout”。那条路会把动态状态、内存效果和控制流强行塞进失去闭包的代数，最终留下一个比原始 SSA 更难维护的符号系统。

更强也更诚实的目标，是建立一套**访问实现理论**：

$$
\boxed{
\text{Program Memory Effects}
\xleftarrow{\ \text{observation}\ }
\text{Target Primitive Program}
\xleftarrow{\ \text{synthesis}\ }
\text{Execution Realization}
}
$$

LinearLayout 继续统一“硬件位置持有什么逻辑元素”。Memory Effect Family 统一“逻辑实例产生什么内存行为”。Realization 将两者连接。Primitive refinement 决定机器是否真的能够实现它。图级优化在所有合法 realization 中选择代价更低者。

理论的统一性落在一个判据上：目标计划隐藏内部事件后的行为，必须 refinement 源程序允许的行为。工程上的多种分析、SMT、Presburger 和目标能力表只是证明这一个判据的工具，不再拥有各自的访存世界。

这套模型已经比 `AccessMapping`、TensorView recovery 和“多个正交 facts”深了一层。它也明确暴露了自己的软肋：memory model、event identity、provenance、poison、primitive semantics、private-event visibility、synthesis、compile time 和 cost model。只有逐一击穿这些问题，它才有资格从一篇漂亮的设计文档变成一篇真正的论文。
