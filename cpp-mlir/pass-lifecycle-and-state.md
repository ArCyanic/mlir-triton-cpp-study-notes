# MLIR Pass 调度引擎与状态管理

> 本文系统剖析 MLIR 与 Triton 编译器中 **`OpPassManager` 调度引擎** 的生命周期管理模型。从 `std::unique_ptr<Pass>` 异构类型擦除容器出发，深入多线程并行流水线（Multi-threaded Pipeline）下的**两阶段克隆（Two-Stage Cloning）**与 `threadingSibling` 诊断追踪；进而系统拆解单次执行上下文 **`PassExecutionState`** 的就地构造（`emplace`）、`PointerIntPair` 紧凑状态存储与 `function_ref` 回调的生命周期边界。

## 目录

- [1. 异构 Pass 队列与所有权](#1-异构-pass-队列与所有权)
  - [1.1 类型擦除与容器管理](#11-类型擦除与容器管理)
  - [1.2 工厂构造与所有权转移](#12-工厂构造与所有权转移)
  - [1.3 虚析构级联释放](#13-虚析构级联释放)
- [2. 多线程并行流水线与克隆](#2-多线程并行流水线与克隆)
  - [2.1 线程实例隔离](#21-线程实例隔离)
  - [2.2 两阶段克隆机制](#22-两阶段克隆机制)
  - [2.3 `threadingSibling` 溯源](#23-threadingsibling-溯源)
- [3. 单次执行状态（`PassExecutionState`）](#3-单次执行状态passexecutionstate)
  - [3.1 长期配置与执行状态分离](#31-长期配置与执行状态分离)
  - [3.2 `optional` 就地构造](#32-optional-就地构造)
  - [3.3 紧凑状态与回调生命周期](#33-紧凑状态与回调生命周期)
- [4. 生命周期与状态流转矩阵](#4-生命周期与状态流转矩阵)

## 1. 异构 Pass 队列与所有权

### 1.1 类型擦除与容器管理

在 MLIR 中，一个编译流水线（Pipeline）由许多功能不同、拥有各自独立配置成员的具体 Pass 组成（如常量折叠 Pass、Warp 特化 Pass、死代码消除 Pass）。

为了在统一的队列中管理这些异构对象，`OpPassManager` 采用了**基类指针类型擦除（Type Erasure via Base Pointer）+ 唯一所有权（Unique Ownership）**的设计：

```cpp
namespace mlir {
class OpPassManagerImpl {
  // 异构容器：以 Pass* 统一接口保存任意具体类型的动态对象
  std::vector<std::unique_ptr<Pass>> passes;

public:
  void addPass(std::unique_ptr<Pass> pass) {
    passes.push_back(std::move(pass));
  }
};
}
```

```
                   OpPassManager 的异构队列存储拓扑
std::vector<std::unique_ptr<Pass>> passes
  │
  ├─► [0]: unique_ptr<Pass> ──► 堆上实体: VerifyWarpSpecializationPartitions
  ├─► [1]: unique_ptr<Pass> ──► 堆上实体: CanonicalizerPass (死代码消除与折叠)
  └─► [2]: unique_ptr<Pass> ──► 堆上实体: TritonGPUAutomaticWarpSpecialization
```

- **类型擦除的作用**：对外暴露统一的接口 `Pass*`，消除了上层 PassManager 对具体派生类头文件的编译期依赖；
- **内存驻留保证**：堆上分配的具体派生类对象在整个 Pipeline 生命周期内内存地址保持稳定，数据字段完整驻留。

### 1.2 工厂构造与所有权转移

在 Triton 源码中，Pass 通常通过公开的工厂函数创建并注册进 Pipeline：

```cpp
// Triton 中的流水线装配
auto addPassWithPartitionVerifier = [&](std::unique_ptr<Pass> pass) {
  pm.addPass(std::move(pass)); // 所有权由调用方转移给 PassManager
  pm.addPass(createVerifyWarpSpecializationPartitionsPass());
};
```

```
                          Pass 实例的所有权转移时序
工厂函数 create...Pass()
  │
  ├─► 1. std::make_unique<ConcretePass>() (在堆上分配完整派生对象)
  │
  ├─► 2. 隐式向上转换为 std::unique_ptr<Pass> (静态接口收束为 Pass*)
  │
  ▼ 3. 通过 std::move(pass) 转移控制权
OpPassManager.passes (成为该 Pass 堆内存的唯一所有者)
```

### 1.3 虚析构级联释放

当 `OpPassManager` 自身销毁或调用 `clear()` 时，容器内部的 `std::unique_ptr<Pass>` 会依次对其持有的裸指针调用 `delete`。

由于静态指针类型为 `Pass *`，基类 `Pass` 必须显式声明 **`virtual ~Pass() = default;`**，确保触发多态级联析构链：

```
                    Pass 销毁时的级联虚析构时序
delete (Pass*)ptr
  │
  ▼ 1. 读取 ptr->vptr，跳入派生类的真实析构入口
~VerifyWarpSpecializationPartitions()   ; 释放派生类自有数据成员
  │
  ▼ 2. 自动调用 CRTP 模板基类析构
~PassWrapper()                          ; 释放 PassWrapper 内部资源
  │
  ▼ 3. 自动调用调度基类析构
~OperationPass<ModuleOp>()              ; 释放 ModuleOp 调度相关元数据
  │
  ▼ 4. 自动调用根基类析构
~Pass()                                 ; 最终回收 Pass 头部基础字段与堆内存
```

> [!CAUTION]
> 若基类 `Pass` 未声明虚析构函数，`delete (Pass*)ptr` 将仅执行 `~Pass()`，导致派生类中的成员变量发生内存泄漏与未定义行为。

## 2. 多线程并行流水线与克隆

### 2.1 线程实例隔离

在现代编译器中，为了加速编译，PassManager 会将一个大型 Module 中的各个独立函数（`func.func`）分发到不同的 CPU 线程上并行执行（Multi-threaded Pipeline Execution）。

```
【多线程共享 Pass 实例的问题】
Thread 1 (处理 func_A) ──┐
                         ├─► 读写同一个 Pass 实例 (内部状态产生数据竞争)
Thread 2 (处理 func_B) ──┘
```

- **隔离约束**：**Pass 实例必须是线程隔离的**。每个工作线程持有整个 Pipeline 中所有 Pass 的一份**独立的深拷贝副本**。

### 2.2 两阶段克隆机制

为了在复制 Pipeline 时既能保证派生类的完整多态性，又能保留用户动态修改过的命令行参数，MLIR 设计了 **两阶段克隆机制（Two-Stage Cloning）**：

```cpp
namespace mlir {
class Pass {
public:
  // 顶层对外克隆接口
  std::unique_ptr<Pass> clone() const {
    // 阶段 1: 虚函数分派，深拷贝具体 C++ 派生对象
    auto newInst = clonePass();
    // 阶段 2: 遍历反射复制 options 的当前运行时配置值
    newInst->copyOptionValuesFrom(this);
    return newInst;
  }

protected:
  // 由 CRTP (PassWrapper) 或派生类覆盖的多态拷贝入口
  virtual std::unique_ptr<Pass> clonePass() const = 0;
};
}
```

```
                     Pass::clone() 的两阶段执行时序
调用: pass->clone()
  │
  ├─► 【阶段 1: clonePass()】
  │   - 虚函数分派至 CRTP 实现: make_unique<PassT>(*static_cast<const PassT*>(this))
  │   - 调用 PassT 的 Copy Constructor，生成全新的独立堆对象
  │   - 重新生成派生类成员，重置 passState 为 std::nullopt
  │
  ▼
【阶段 2: copyOptionValuesFrom(this)】
  - 遍历当前 Pass 的 Pass::Option 选项列表
  - 将原 Pass 中经过 CLI 或程序动态修改的值同步到新对象中
  - 重置统计量（Pass::Statistic），保证多线程各自独立统计
```

### 2.3 `threadingSibling` 溯源

当复制出用于多线程并行的 Sibling Pass 副本后，框架需要保留它与原始主 Pass 的溯源关系，以便在编译结束时将多线程产生的耗时数据与优化统计量汇总到主 Pipeline：

```cpp
for (const std::unique_ptr<Pass> &pass : mainPipeline.passes) {
  std::unique_ptr<Pass> threadPass = pass->clone();
  // 核心：记录当前副本的兄弟溯源指针
  threadPass->threadingSibling = pass.get();
  workerQueue.push_back(std::move(threadPass));
}
```

```
                  多线程 Sibling 诊断与统计聚合拓扑
 主线程 Main Pass (threadingSibling = nullptr) ◄────── 统计汇总归并
       ▲                                 ▲
       │ threadingSibling                │ threadingSibling
 线程 1 副本 Sibling Pass          线程 2 副本 Sibling Pass
 (独立执行 func_A)                 (独立执行 func_B)
```

## 3. 单次执行状态（`PassExecutionState`）

### 3.1 长期配置与执行状态分离

在 Pass 的生命周期中，存在两种不同时间维度的数据：
1. **长期静态配置（Long-Term Configuration）**：如 `numStages`、优化级别、调试开关。这些数据在 Pass 构造时确定，随 `clone()` 延续；
2. **单次动态执行状态（Transient Execution Context）**：如当前正在遍历的 `Operation*`、IR 分析缓存管理器 `AnalysisManager`、动态流水线回调函数。这些数据**仅在单次 `runOnOperation()` 执行期间有效**。

若将动态 IR 指针保存在 Pass 的普通成员变量中，可能导致多线程克隆时复制失效的旧指针。

### 3.2 `optional` 就地构造

MLIR 通过在基类 `Pass` 中维护一个 `std::optional<PassExecutionState>`，在每一次 Pass 执行前后实现上下文的**就地挂载与清空**：

```cpp
namespace mlir {
class Pass {
  // 仅在执行期间具有有效值的可选状态容器
  std::optional<PassExecutionState> passState;

  friend class OpPassManager; // 仅允许调度管理器控制其生命周期
};
}
```

```
                    一次 Pass 执行中的状态挂载与回收时序
PassManager 调度器准备执行当前 Pass
  │
  ├─► 1. pass->passState.emplace(currentOp, am, callback); 
  │      ; 就地构造 PassExecutionState，绑定当前 IR 根与分析缓存
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

- **`emplace()` 的优势**：每次执行均在已有的 `optional` 内存槽位内**就地构造（In-place Construction）**，避免了堆内存的动态分配。

### 3.3 紧凑状态与回调生命周期

`PassExecutionState` 结构体的内部实现：

```cpp
struct PassExecutionState {
  // 1. 紧凑指针：将 64 位 Operation* 与 1-bit 失败标记压缩在单个 8 字节中
  llvm::PointerIntPair<Operation *, 1, bool> irAndPassFailed;

  // 2. 分析缓存管理器
  AnalysisManager analysisManager;

  // 3. 本次执行保留的分析标记
  PreservedAnalyses preservedAnalyses;

  // 4. 动态嵌套 Pipeline 回调借用视图
  function_ref<LogicalResult(OpPassManager &, Operation *)> pipelineExecutor;
};
```

#### `function_ref` 的借用生命周期安全边界

在 Triton 的嵌套 Pass（如 Automatic Warp Specialization）中，Pass 需要在自身内部动态执行一个子流水线：

```cpp
if (failed(runPipeline(pm, getOperation())))
  return signalPassFailure();
```

- **底层机制**：`runPipeline` 内部调用了 `pipelineExecutor`。该 `function_ref` 借用了调用方栈上的 Lambda 对象（16 字节只读视图）；
- **安全约束**：该回调**不应逃逸**出当前 `runOnOperation()` 的调用栈帧。一旦 `runOnOperation()` 返回，栈帧弹出，借用立即失效，由下一次 `emplace()` 覆盖。

## 4. 生命周期与状态流转矩阵

```
                MLIR Pass 生命周期流转矩阵
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. 构造与装配期 (Pipeline Construction)                                     │
│    - 工厂函数分配 ConcretePass 堆对象                                       │
│    - 通过 std::move(pass) 转移给 OpPassManager (unique_ptr<Pass> 队列)        │
│    - 状态: options 赋予默认值 / CLI 参数，passState 为 nullopt               │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. 多线程并行分发期 (Parallel Forking)                                       │
│    - Manager 对每个 Pass 调用 clone()                                       │
│    - clonePass() 深拷贝派生对象 + copyOptionValuesFrom() 同步配置           │
│    - 设置 threadingSibling 指向主 Pass 溯源节点                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. 动态执行期 (Execution Loop)                                              │
│    - passState.emplace(op, am, callback) 绑定当前 IR 上下文                 │
│    - 虚调用执行 runOnOperation()                                             │
│    - signalPassFailure() 写入 irAndPassFailed 最低位                        │
│    - 依据 preservedAnalyses 使失效缓存失效                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ 4. 销毁与归并期 (Destruction & Collection)                                   │
│    - 多线程副本将 Statistics 统计数据归并给 threadingSibling 主 Pass         │
│    - unique_ptr<Pass> 触发虚析构链 (~ConcretePass -> ... -> ~Pass)          │
│    - 释放所有堆内存                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

> [!TIP]
> **总结**：
> - **类型擦除与虚析构**：确保了异构 Pass 可以在同一个队列中统一管理与多态销毁；
> - **两阶段克隆与 Sibling 溯源**：在保证多线程执行环境隔离的同时，维持了全局配置一致性与监控诊断聚合；
> - **`PassExecutionState` 就地分离**：清晰划分了长期配置与单次上下文，避免了多线程状态污染与生命周期问题。\n
