# MLIR Pass 调度引擎与状态管理

> 本文系统解构 MLIR 与 Triton 编译器中 **`OpPassManager` 调度引擎** 的生命周期管理与多线程并发执行模型。从 `std::unique_ptr<Pass>` 异构类型擦除容器与树状嵌套流水线拓扑出发，深入多线程并行调度下的**两阶段克隆（Two-Stage Cloning）**与 `AnalysisManager` 分析缓存分支派生（Fork/Join）与增量失效机制；系统拆解单次执行上下文 **`PassExecutionState`** 的就地构造（`emplace`）、`PointerIntPair` 紧凑状态存储与 `function_ref` 回调的生命周期借用边界；最后剖析基于 `PassInstrumentor` 的耗时统计、IR 差异转储与崩溃现场复现（Crash Reproducer）工程闭环。

---

## 1. 异构 Pass 队列与流水线嵌套架构

### 1.1 类型擦除与异构容器管理

在 MLIR 体系中，一个完整的编译器流水线（Pipeline）由许多算法逻辑迥异、内部配置参数独立的具体 Pass 组合而成（如规范化 Pass、Warp 特化 Pass、死代码消除 Pass）。

为了在统一的流水线队列中调度这些异构对象，`OpPassManager` 采用了**基类指针类型擦除（Type Erasure via Base Pointer）+ 唯一所有权（Unique Ownership）**的容器架构：

```cpp
namespace mlir {
class OpPassManagerImpl {
    // 异构容器：以 std::unique_ptr<Pass> 统一收拢各类具体的动态 Pass 实例
    std::vector<std::unique_ptr<Pass>> passes;

public:
    void addPass(std::unique_ptr<Pass> pass) {
        passes.push_back(std::move(pass));
    }
};
}
```

```text
                    OpPassManager 的异构队列存储拓扑
std::vector<std::unique_ptr<Pass>> passes
  │
  ├─► [0]: unique_ptr<Pass> ──► 堆上实体: VerifyWarpSpecializationPartitions
  ├─► [1]: unique_ptr<Pass> ──► 堆上实体: CanonicalizerPass (死代码消除与折叠)
  └─► [2]: unique_ptr<Pass> ──► 堆上实体: TritonGPUAutomaticWarpSpecialization
```

- **类型擦除的解耦价值**：调度器对外暴露统一的虚接口基类 `Pass*`，使顶层流水线管理器彻底解除了对数百个具体派生类头文件的编译期依赖；
- **内存驻留确定性**：在堆上分配的具体派生类对象在整个 Pipeline 装配与调度生命周期内物理地址保持恒定，所有成员数据完整驻留。

---

### 1.2 动态流水线自适应嵌套拓扑

#### 树状嵌套流水线模型

MLIR 的中间表示具有高度结构化的嵌套特征（例如 `builtin.module` 内部包含多个 `func.func`，`func.func` 内部包含 `gpu.launch`）。为了精确控制 Pass 的执行范围，`OpPassManager` 并非单一维度的平铺数组，而是演进为**递归嵌套的树状流水线架构（Nested Pass Pipeline）**：

```text
                      MLIR 树状嵌套流水线拓扑
OpPassManager (挂载根: builtin.module)
  │
  ├─► Pass 1: TritonGPUInliner (作用于 ModuleOp 级别)
  │
  ├─► OpPassManager (嵌套子流水线，锚定: func.func)
  │     │
  │     ├─► Pass 2: TritonGPUCombineOps (作用于每个 FuncOp 局部)
  │     └─► Pass 3: TritonGPUCanonicalizer (作用于每个 FuncOp 局部)
  │
  └─► Pass 4: TritonGPUAutomaticWarpSpecialization (返回 ModuleOp 级别)
```

#### 动态流水线自适应收缩机理

当 `OpPassManager` 遇到针对更低层级 IR 节点的嵌套流水线时，调度引擎会自动执行 **自适应收缩（Adaptive Nesting & Slicing）**：
1. 调度器在顶层提取出所有匹配目标类型的子算子（如 Module 内部的所有 `FuncOp`）；
2. 调度器自动将嵌套子流水线分发给这些子算子，使其在局部子图上自包含运行；
3. 子流水线执行完成后，控制权平滑归还给外层 PassManager，继续执行后续的全局 Pass。

---

### 1.3 工厂构造与虚析构级联释放

在工程实践中，Pass 通常通过声明式工厂函数创建并注册进 Pipeline：

```cpp
// 典型的 Pass 创建与所有权移交
auto addPassWithVerifier = [&](std::unique_ptr<Pass> pass) {
    pm.addPass(std::move(pass)); // 所有权转移至 PassManager
    pm.addPass(createVerifyWarpSpecializationPartitionsPass());
};
```

当 `OpPassManager` 自身生命周期终结或调用 `clear()` 时，容器内部的 `std::unique_ptr<Pass>` 会依次对其持有的裸指针调用 `delete`。由于静态指针类型为 `Pass *`，基类 `Pass` 必须显式声明 **`virtual ~Pass() = default;`**，确保触发多态级联析构链：

```text
                     Pass 销毁时的级联虚析构时序
delete (Pass*)ptr
  │
  ▼ 1. 读取 ptr->vptr，跳入派生类的真实析构入口
~VerifyWarpSpecializationPartitions()   ; 释放派生类自有的动态容器与分析字段
  │
  ▼ 2. 自动调用 CRTP 模板基类析构
~PassWrapper()                          ; 释放 PassWrapper 内部资源
  │
  ▼ 3. 自动调用调度基类析构
~OperationPass<ModuleOp>()              ; 释放调度相关元数据
  │
  ▼ 4. 自动调用根基类析构
~Pass()                                 ; 最终回收 Pass 头部基础字段与堆内存
```

> [!CAUTION]
> 若根基类 `Pass` 未声明虚析构函数，`delete (Pass*)ptr` 将仅执行基类 `~Pass()`，导致派生类中的私有成员变量发生严重的内存泄漏与未定义行为。

---

## 2. 多线程并行调度与分析缓存隔离

### 2.1 线程实例隔离与两阶段克隆机制

在现代编译器中，为了极限压榨多核 CPU 性能，PassManager 会将一个 Module 中的各个独立函数（`func.func`）分发到线程池的工作线程上并行执行（Multi-threaded Pipeline Execution）。

如果多个线程并发共享同一个 Pass 实例，其内部状态变量将不可避免地发生**数据竞争（Data Race）**。为此，MLIR 强制要求 **Pass 实例必须做到线程局部隔离**。

为了在克隆整个流水线时既能保留派生类的完整多态性，又能无缝同步外部动态传入的命令行选项，MLIR 设计了 **两阶段克隆机制（Two-Stage Cloning）**：

```cpp
namespace mlir {
class Pass {
public:
    // 顶层对外统一克隆接口
    std::unique_ptr<Pass> clone() const {
        // 阶段 1: 虚函数多态分派，深拷贝具体 C++ 派生对象
        auto newInst = clonePass();
        // 阶段 2: 遍历反射复制 Options 的当前运行时配置值
        newInst->copyOptionValuesFrom(this);
        return newInst;
    }

protected:
    // 由 CRTP (PassWrapper) 自动重写的深拷贝实现
    virtual std::unique_ptr<Pass> clonePass() const = 0;
};
}
```

```text
                     Pass::clone() 的两阶段执行时序
调用: pass->clone()
  │
  ├─► 【阶段 1: clonePass()】
  │   - 虚函数分派至 CRTP 实现: make_unique<PassT>(*static_cast<const PassT*>(this))
  │   - 调用 PassT 的 Copy Constructor，在堆上分配全新的独立对象
  │   - 重新初始化成员变量，重置 passState 为 std::nullopt
  │
  ▼
【阶段 2: copyOptionValuesFrom(this)】
  - 遍历原 Pass 的 Pass::Option 选项列表
  - 将原 Pass 中经过 CLI 或程序动态修改的参数值精准同步到新对象中
  - 重置统计量（Pass::Statistic），确保多线程各自独立累加
```

---

### 2.2 AnalysisManager 分析缓存分支派生与失效传播

#### 分析缓存多线程派生与归并

Pass 在执行过程中需要高频查询昂贵的 IR 分析结果（如 `DominanceInfo` 支配树、`Liveness` 活跃度分析）。MLIR 设计了 **`AnalysisManager`（分析缓存管理器）**，其与多线程流水线协同工作，形成了精密的 **Fork/Join 分支派生与增量失效机制**：

```text
               AnalysisManager 多线程分支派生与失效传播
                   Module 级全局 AnalysisManager (根节点)
                                     │
             ┌───────────────────────┴───────────────────────┐
             ▼ Fork 派生                                     ▼ Fork 派生
   线程 1 子 AnalysisManager                       线程 2 子 AnalysisManager
   (绑定 func_A，独立缓存支配树)                   (绑定 func_B，独立缓存支配树)
             │                                               │
             │ Pass 1 变换修改了 func_A                      │ Pass 2 未修改 func_B
             ▼                                               ▼
   局部缓存精准失效:                                保留所有分析缓存:
   am.invalidate(preservedAnalyses)                 无需重新计算，后续 Pass 直接复用！
             │
             └───────────────────────┬───────────────────────┘
                                     │ Join 归并
                                     ▼
                      全局未被破坏的分析缓存安全保留！
```

#### 增量失效与局部性优化原则

1. **分支派生（Forking）**：主线程在将子流水线分发给 Worker 线程时，调用 `am.slice(funcOp)` 派生出一个受限的子 `AnalysisManager`，子线程只能查询和修改当前 `FuncOp` 作用域内的分析结果，**完全阻断跨线程分析缓存污染**；
2. **选择性失效（Selective Invalidation）**：Pass 执行完成后，通过声明 `PreservedAnalyses` 告知管理器本次变换保留了哪些分析。管理器仅将未保留的局部缓存标记失效，**绝不轻易向上传播破坏外层 Module 级的全局分析缓存**。

---

### 2.3 threadingSibling 溯源与统计聚合

当复制出用于多线程并行的 Sibling Pass 副本后，框架必须保留其与原始主 Pass 的溯源指针，以便在编译流水线结束时将各个线程产生的耗时数据与优化计数器汇总归并：

```cpp
for (const std::unique_ptr<Pass> &pass : mainPipeline.passes) {
    std::unique_ptr<Pass> threadPass = pass->clone();
    // 关键：记录当前线程副本的兄弟溯源指针
    threadPass->threadingSibling = pass.get();
    workerQueue.push_back(std::move(threadPass));
}
```

```text
                  多线程 Sibling 诊断与统计聚合拓扑
 主线程 Main Pass (threadingSibling = nullptr) ◄────── 统计汇总归并
       ▲                                 ▲
       │ threadingSibling                │ threadingSibling
 线程 1 副本 Sibling Pass          线程 2 副本 Sibling Pass
 (独立执行 func_A)                 (独立执行 func_B)
```

---

## 3. 单次执行状态与上下文生命周期

### 3.1 长期静态配置与瞬态执行状态解耦

在 Pass 的整个生命周期中，存在两种不同时间维度的数据：
1. **长期静态配置（Long-Term Configuration）**：如 `numStages`、循环展开因子、优化级别。这些数据在 Pass 构造时确定，并在 `clone()` 时跨线程同步延续；
2. **瞬态动态执行状态（Transient Execution Context）**：如当前正在遍历的 `Operation*` 根节点、当前线程的 `AnalysisManager` 句柄、动态子流水线回调。这些数据**仅在单次 `runOnOperation()` 执行期间具有物理意义**。

若将动态 IR 指针保存在 Pass 的普通成员变量中，在多线程克隆或流水线复用时极易引发悬空指针与脏数据污染。

---

### 3.2 optional 就地构造与零堆分配管理

MLIR 在基类 `Pass` 中维护一个 `std::optional<PassExecutionState>`，在每一次 `runOnOperation()` 执行前后实现执行上下文的**就地挂载与清空**：

```cpp
namespace mlir {
class Pass {
    // 仅在单次 runOnOperation 期间具有有效值的状态容器
    std::optional<PassExecutionState> passState;

    friend class OpPassManager; // 仅允许调度管理器控制其生命周期
};
}
```

```text
                    一次 Pass 执行中的状态挂载与回收时序
PassManager 调度器准备执行当前 Pass
  │
  ├─► 1. pass->passState.emplace(currentOp, am, callback); 
  │      ; 就地构造 PassExecutionState，绑定当前 IR 根与分析缓存 (0 次堆分配)
  │
  ├─► 2. pass->runOnOperation(); 
  │      ; 进入 Pass 核心算法，期间通过 getOperation() 读取 passState
  │
  ├─► 3. bool failed = pass->passState->irAndPassFailed.getInt();
  │      ; 提取本次执行的成功/失败标记位
  │
  ▼ 4. analysisManager.invalidate(pass->passState->preservedAnalyses);
         ; 依据声明使未保留的分析缓存失效
```

通过 `passState.emplace(...)`，调度器在原有的 `std::optional` 内存槽位内直接执行**就地构造（In-place Construction）**，彻底免除了频繁的 `new`/`delete` 堆内存开销。

---

### 3.3 紧凑状态存储与 function_ref 回调安全边界

`PassExecutionState` 结构体的内部字段经过了极致的内存紧凑优化：

```cpp
struct PassExecutionState {
    // 1. 紧凑指针：将 64 位 Operation* 与 1-bit 失败标志位压缩在单个 8 字节中
    llvm::PointerIntPair<Operation *, 1, bool> irAndPassFailed;

    // 2. 分析缓存管理器句柄
    AnalysisManager analysisManager;

    // 3. 本次执行保留的分析标记
    PreservedAnalyses preservedAnalyses;

    // 4. 动态嵌套 Pipeline 执行器借用视图
    function_ref<LogicalResult(OpPassManager &, Operation *)> pipelineExecutor;
};
```

在 Triton 的嵌套 Pass（如 Automatic Warp Specialization）中，Pass 需要在自身内部动态调度执行一个子流水线：

```cpp
if (failed(runPipeline(pm, getOperation())))
    return signalPassFailure();
```

- **执行机理**：`runPipeline` 内部调用了 `pipelineExecutor`。该 `function_ref` 借用了调用方栈上的 Lambda 闭包（16 字节只读视图）；
- **安全约束**：该回调视图的生命周期被严格锁定在当前 `runOnOperation()` 的调用栈帧内，**严禁逃逸或异步持久化**，在函数返回时随栈帧弹出自然销毁。

---

## 4. 观察者仪器化与工程闭环

### 4.1 PassInstrumentor 观察者链路

在编译器开发中，排查 Pass 导致的 IR 损毁或分析编译性能瓶颈时，若直接在每个 Pass 内部插入打印代码，会造成极其严重的侵入性污染。

MLIR 引入了 **`PassInstrumentor`（Pass 仪器化观察者模式）**，在每一次 Pass 和 Analysis 执行的关键时序节点进行无侵入式切面拦截：

```cpp
class PassInstrumentation {
public:
    virtual void runBeforePass(Pass *pass, Operation *op) {}
    virtual void runAfterPass(Pass *pass, Operation *op) {}
    virtual void runAfterPassFailed(Pass *pass, Operation *op) {}
    virtual void runBeforeAnalysis(StringRef name, TypeID id, Operation *op) {}
    virtual void runAfterAnalysis(StringRef name, TypeID id, Operation *op) {}
};
```

---

### 4.2 编译耗时统计与 IR 变换 Diff 转储

基于 `PassInstrumentor`，MLIR 内置了多项强大的编译器调试工具链：

1. **耗时分析（`-pass-timing`）**：自动监控每个 Pass 在各个线程上的 CPU 运行耗时，生成分层树状报告；
2. **IR 差异转储（`-print-ir-after-all` / `-print-ir-after-change`）**：在每次 Pass 运行后自动比对 IR 变化，仅在 IR 发生实质性变换时生成彩色的 Diff 输出；
3. **崩溃现场复现（Crash Reproducer）**：当 Pass 内部触发断言崩溃或致命错误时，`runAfterPassFailed` 拦截器会自动捕获失败瞬间的最小化 IR 片段与 Pipeline 文本配置，生成可独立复现的 `.mlir` 崩溃脚本。

---

## 5. 生命周期与状态流转全景矩阵

```text
                   MLIR Pass 完整生命周期流转矩阵
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. 流水线装配期 (Pipeline Construction)                                     │
│    - 工厂函数在堆上分配 ConcretePass 实例                                   │
│    - 通过 std::move 移交给 OpPassManager (unique_ptr<Pass> 队列)            │
│    - 状态: options 赋予默认值 / CLI 参数，passState 为 std::nullopt         │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. 多线程并行分发期 (Parallel Forking)                                       │
│    - Manager 对每个 Pass 调用 clone() 进行两阶段克隆                         │
│    - clonePass() 深拷贝派生对象 + copyOptionValuesFrom() 同步动态配置       │
│    - 设置 threadingSibling 指向主 Pass 溯源节点                             │
│    - AnalysisManager.slice() 派生出当前线程专属的受限子缓存                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. 动态执行期 (Execution Loop)                                              │
│    - passState.emplace(op, am, callback) 零堆分配就地绑定当前上下文          │
│    - PassInstrumentor.runBeforePass() 触发拦截钩子                          │
│    - 虚调用执行具体算法 runOnOperation()                                     │
│    - signalPassFailure() 将失败状态写入 irAndPassFailed 最低 bit            │
│    - AnalysisManager 依据 preservedAnalyses 执行增量失效                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ 4. 销毁与归并期 (Destruction & Collection)                                   │
│    - 多线程副本将 Statistics 统计数据汇总归并给 threadingSibling 主 Pass     │
│    - unique_ptr<Pass> 触发虚析构链 (~ConcretePass -> ... -> ~Pass)          │
│    - 释放所有堆内存                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```
