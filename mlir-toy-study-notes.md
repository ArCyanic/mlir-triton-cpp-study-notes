# MLIR 核心架构与底层机制解析

> 本笔记整理自 MLIR / LLVM 体系核心机制的深度探讨与架构推演（以 MLIR Toy 教程为主线），覆盖基础抽象、重写规则、函数内联、方言降级、符号系统与底层内存模型。

## 1. MLIR 核心算子抽象模型

### 1.1 算子双层句柄架构

在 MLIR 的设计体系中，对算子（Operation）采用了经典且精妙的 **“实体与视图分离（Handle-Body Pattern）”** 设计模式。

```
                    ┌──────────────────────────────────────────────┐
                    │               class Operation                │
                    │──────────────────────────────────────────────│
                    │ - operands : SmallVector<Value>              │
                    │ - results  : SmallVector<Value>              │
                    │ - attrs    : DictionaryAttr                  │
                    │ - regions  : SmallVector<Region>             │
                    │ - ... 真实堆内存实体 (通用无类型容器)           │
                    └──────────────────────┬───────────────────────┘
                                           ▲
                                           │ 内部持有指针 Operation* state
                    ┌──────────────────────┴───────────────────────┐
                    │               class ConstantOp               │
                    │           (或 AddIOp, TransposeOp 等)         │
                    │──────────────────────────────────────────────│
                    │ - getValue() -> Attribute                    │
                    │ - getResult() -> Value                       │
                    │ - ... 强类型智能指针视图 (零额外成员变量)         │
                    └──────────────────────────────────────────────┘
```

#### 核心概念对照

| 维度 | `Operation`（底层通用物理容器） | `Op` 派生类（如 `ConstantOp`, `AddIOp`） |
| :--- | :--- | :--- |
| **角色定位** | IR 的真实物理实体与数据存储者（Data Container） | 作用在 `Operation*` 上的强类型语义化接口视图（Semantic View） |
| **内存占用** | 包含多个动态数组（Operands, Results, Attrs 等），体积较大 | **仅占一个指针大小**（`sizeof(Operation*)`，通常为 8 字节） |
| **传递方式** | **传指针**（`Operation*`） | **按值传递**（`ConstantOp op`，Pass by Value） |
| **生命周期** | 托管在 `MLIRContext` 内存池中 | 轻量栈对象句柄，随用随复制 |
| **主要用途** | 编写**通用 IR Pass**（如遍历、死代码消除、内存回收，无需关心具体算子语义） | 编写**特定领域逻辑**（如加法类型推导、常量求值，提供 IDE 补全与编译期安全） |

#### 简化 C++ 伪代码理解

```cpp
// 1. Operation：真实的数据存储结构（底层通用容器）
class Operation {
public:
    SmallVector<Value> operands;   // 存储输入参数
    SmallVector<Value> results;    // 存储输出结果
    DictionaryAttr     attributes; // 存储属性（编译期常量、配置等）
    SmallVector<Region> regions;   // 存储嵌套的代码块区域
    // ... 真正占用大量堆内存的地方
};

// 2. Op 派生类：强类型智能指针包装（零字段视图）
class AddIOp {
private:
    Operation *state; // 唯一的成员变量

public:
    // 从底层通用 Operation 构造视图
    AddIOp(Operation *op) : state(op) {}

    // 提供强类型、语义化的 C++ 接口
    Value getLhs()    { return state->operands[0]; } // 封装了底层脆弱的数组索引
    Value getRhs()    { return state->operands[1]; }
    Value getResult() { return state->results[0];  }

    // 隐式/显式转换回底层指针
    Operation *getOperation() const { return state; }
    Operation *operator->() const   { return state; }
};
```

> [!TIP]
> **为什么 `Op` 派生类必须 "defines no class fields" 且按值传递？**
> 1. **零内存冗余**：所有数据均存在于底层的 `Operation` 中，`Op` 仅作为 C++ 语义接口外壳。
> 2. **语法自然**：因为内部仅有一个 8 字节指针，按值传递（`void process(AddIOp op)`）与传指针开销完全一致，但编写业务逻辑时省去了繁琐的解引用 `*` 或箭头 `->`，体验如同普通值对象（类似 `std::string_view`）。

### 1.2 Pure 语义死代码消除

#### Pure 特性的本质

在 MLIR 中，`Pure`（早期版本称为 `NoSideEffect`）Trait 描述了一个算子除了产生其声明的返回值之外，**不会对程序运行环境产生任何可观测的副作用**。

具体满足以下三点：

1. **无内存修改**：不写入任何物理指针、内存缓冲区或硬件寄存器。
2. **无状态变更**：不修改全局变量、不触发 I/O（如控制台打印、写文件、发网络包）。
3. **可安全删除**：只要它的返回值没有被任何其他下游 Op 使用（即 `User` 引用计数为 0），直接将其抹去对整个程序的运行结果毫无影响。

```tablegen
// 在 TableGen (ODS) 中为算子标记 Pure
def TransposeOp : Toy_Op<"transpose", [Pure]> {
    let summary = "transpose operation";
    let arguments = (ins F64Tensor:$input);
    let results = (outs F64Tensor);
}
```

#### 编译器的保守防御原则（Conservative Assumption）

当规范化器（Canonicalizer）发现一个算子 `y = op(x)` 的结果 `y` 没有被任何人使用时：

- **未标记 `Pure`**：编译器必须采取保守策略，假设该 Op 可能会向外部写日志、触发串口或产生隐式副作用，**不敢删除**。
- **标记了 `Pure`**：编译器确信它只是一个纯粹的数学映射，既然没人需要它的计算结果，就可以安全将其整条指令抹除，触发**死代码消除（DCE）**。

> [!IMPORTANT]
> **设计准则：安全第一（Safe by Default）**
> 编译器无法自动推断算子内部是否含有隐式副作用（如写文件、发网络包或断言崩溃）。因此 MLIR 默认保守假定所有算子均有副作用；纯计算算子必须由开发者显式声明 `[Pure]` 标签。

#### 为什么不默认将所有 Tensor 算子都设为 Pure？

虽然 90% 以上的 Tensor 算子都是纯计算，但以下情况的 Tensor 算子**绝对不能**标记为 `Pure`：

1. **I/O 与调试算子**（如 `toy.print %tensor`、`vector.print %tensor`）：
   - 目的在于向终端输出日志，即使无返回值或结果未被使用，也绝不能被 DCE 删掉。
2. **随机数生成与有状态计算**（如 `tensor.generate_random`、`torch.aten.rand`）：
   - 会隐式修改全局 RNG 随机数生成器的种子状态，两次调用结果不同，不具备纯函数的“可重现性”。
3. **带动态断言/安全检查的算子**（如 `check_shape_or_panic(%tensor)`）：
   - 当运行时 Tensor 形状不合法时会触发崩溃/中断（Panic/Abort）。若因未被使用而被优化器删除，会掩盖潜在的严重运行时错误。
4. **跨线程/硬件通道通信算子**（如 `async.send %tensor`、`channel.write %tensor`）：
   - 将 Tensor 发送至 GPU 队列或网络通道，具有显式的系统级通信副作用。

### 1.3 内存副作用接口建模

当算子从高层抽象降级到硬件/内存层（如 `memref` 方言）时，简单的二元状态（`Pure` / `非 Pure`）已无法满足深度优化需求。MLIR 提供了 `MemoryEffectOpInterface` 进行细粒度副作用建模。

#### 四大核心内存副作用

`mlir::SideEffects::Effect` 下定义了 4 种基础内存副作用：

| 效果类型（Effect） | 语义说明 | 典型 Op 示例 |
| :--- | :--- | :--- |
| `MemoryEffects::Read` | 读取内存内容，不改变内存数据与状态 | `memref.load`, `vector.transfer_read` |
| `MemoryEffects::Write` | 修改/覆盖已有的内存数据 | `memref.store`, `vector.transfer_write` |
| `MemoryEffects::Allocate` | 申请分配一段全新的物理/抽象内存资源 | `memref.alloc`, `gpu.alloc` |
| `MemoryEffects::Free` | 销毁或归还已分配的内存资源 | `memref.dealloc`, `gpu.free` |

#### 副作用的两种标注方式

**方式一：在 ODS (TableGen) 中使用粗粒度 Trait**

适用于副作用单一固定的算子：

```tablegen
include "mlir/Interfaces/SideEffectInterfaces.td"

// 标记该 Op 仅对参数具有“读取内存”的副作用
def MyLoadOp : My_Op<"load", [MemoryEffects<[MemRead]>]> {
    let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$src);
}

// 标记该 Op 对目标具有“写入内存”的副作用
def MyStoreOp : My_Op<"store", [MemoryEffects<[MemWrite]>]> {
    let arguments = (ins AnyType:$value, Arg<AnyMemRef, "", [MemWrite]>:$dst);
}
```

**方式二：在 C++ 中重写 `getEffects` 实现值级别（Value-level）精细控制**

粗粒度 Trait 仅说明“该 Op 会读写内存”，但无法精确指出“读写了哪一个具体指针”。通过实现 `MemoryEffectOpInterface` 可以将效果绑定到具体 `Value` 和 `Resource`：

```cpp
void MyCustomCopyOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
    
    // 1. 精确标注：仅对 $src (输入 Operand 0) 产生 Read 效果
    effects.emplace_back(MemoryEffects::Read::get(), getSrc(),
                         SideEffects::DefaultResource::get());
                         
    // 2. 精确标注：仅对 $dst (输出 Operand 1) 产生 Write 效果
    effects.emplace_back(MemoryEffects::Write::get(), getDst(),
                         SideEffects::DefaultResource::get());
}
```

> [!NOTE]
> **抽象资源建模**：`DefaultResource::get()` 可替换为自定义硬件资源句柄（如 `GPUResource::get()` vs `CPUResource::get()`），从而精确区分不同硬件空间/寄存器堆的内存副作用。

#### 精细副作用带来的编译优化收益

编译器 Pass 依赖精确的 `MemoryEffects` 信息进行激进且安全的优化：

1. **死写消除（Dead Store Elimination, DSE）**：连续两次对同一 `Value` 执行 `Write` 操作，且中间没有任何 `Read` 操作，编译器可安全裁撤前一次写入。
2. **别名分析与指令重排（Alias Analysis & Instruction Reordering）**：若两个 Op 作用于不同的 `Value`（且分析证明无指针别名重叠），即便它们均包含 `Write` 效果，编译器仍可自由调整它们的执行顺序以最大化流水线效率。
3. **循环不变代码外提（Loop-Invariant Code Motion, LICM）**：若循环体内的 `Read` 操作所读取的 `MemRef` 在循环内部不存在任何 `Write` 副作用，该读取指令可被安全提至循环体外部（Hoist）。

## 2. DRR 声明式重写体系

### 2.1 DRR 编译期流水线

DRR 是建立在 LLVM TableGen DSL 之上的**元编程声明式重写框架**，专门用于通过高层声明式语法自动生成 C++ 图重写代码（`mlir::OpRewritePattern`）。其核心哲学在于：**用声明式树模式描述取代手写冗长、脆弱且易错的 C++ 图遍历与指针替换逻辑**。

```text
┌───────────────────────────────┐
│ 1. DSL 声明阶段 (.td 文件)     │ 开发者使用 TableGen DAG (S-expression) 声明匹配与替换规则
└──────────────┬────────────────┘
               │  mlir-tblgen -gen-rewriters
               ▼
┌───────────────────────────────┐
│ 2. 代码生成阶段 (.inc C++代码) │ 自动生成 matchAndRewrite()、dyn_cast、类型校验与 rewriter 调用
└──────────────┬────────────────┘
               │  编译并注册到 RewritePatternSet
               ▼
┌───────────────────────────────┐
│ 3. 驱动执行阶段 (PatternDriver)│ 由 GreedyRewriteDriver 调度，按 Benefit 排序并递归迭代至不动点
└───────────────────────────────┘
```

DRR 的完整工作流包含三大严密衔接的生命周期阶段：

1. **DSL 声明阶段（`.td` 文件）**：开发者使用 Lisp 风格的 S-表达式（S-expression）编写模式匹配和结果替换，语法形式如 `(OpName $arg1, (ChildOp $arg2))`。
2. **代码生成阶段（`mlir-tblgen`）**：构建系统调用 `mlir-tblgen -gen-rewriters` 解析 `.td`，生成对应的 `.inc` C++ 代码。TableGen 自动将源模式展开为一连串的 `isa<...>`、`dyn_cast<...>`、操作数提取及类型检查，并将结果模式展开为 `rewriter.create<...>()` 与 `rewriter.replaceOp()`。
3. **驱动执行阶段（Pattern Driver）**：生成的 C++ Pattern 被批量注册到 `RewritePatternSet` 中。MLIR 的 `GreedyRewriteDriver`（贪婪重写驱动器）按照每个 Pattern 声明的 `Benefit`（收益值）排序，优先应用高收益规则，并在 IR 树上递归迭代，直到 IR 状态达到收敛（不动点）。

### 2.2 Pattern 模式重写基类

在 TableGen 中，定义重写规则的核心类是 `Pattern`，而最常用的是其快捷派生类 `Pat`。

#### Pattern 基类定义与四大参数

```tablegen
class Pattern<
    dag sourcePattern,                      // 1. 源匹配模式
    list<dag> resultPatterns,               // 2. 目标替换模式列表 (支持 1-to-N 生成多个 Op)
    list<dag> additionalConstraints = [],   // 3. 附加约束条件列表 (全为 true 才允许匹配)
    dag benefitsAdded = (addBenefit 0)      // 4. 匹配优先级/收益增量
>;
```

| 参数 | 类型 | 语义说明与用法 |
| :--- | :--- | :--- |
| `sourcePattern` | `dag` | **源匹配模式**：用 DAG 语法描述要匹配的 IR 子树结构（如 `(OpA (OpB $x))`）。根节点必须是一个 MLIR Operation。 |
| `resultPatterns` | `list<dag>` | **目标替换模式列表**：匹配成功后生成的新 IR 结构列表。支持一次重写生成多个全新的算子。 |
| `additionalConstraints` | `list<dag>` | **附加约束条件**：结构匹配之外的 C++ 逻辑谓词（如要求常量大于 0 或元素类型为 f32）。 |
| `benefitsAdded` | `dag` | **匹配优先级**：附加收益值（如 `(addBenefit 10)`）。当多条 Pattern 同时命中同一节点时，优先执行 Benefit 最高的规则。 |

#### Pat 快捷包装类与等价对比

在 MLIR 底层定义（`PatternRewriter.td`）中，`Pat` 继承自 `Pattern`：

```tablegen
// MLIR 源码中的定义：
class Pat<dag pattern, dag result, list<dag> preds = [], dag benefit = (addBenefit 0)>
    : Pattern<pattern, [result], preds, benefit>;
```

- **`Pattern`（基类）**：第二个参数接收 `list<dag>`，允许一次生成多个 Op（N-to-M 场景）。
- **`Pat`（快捷类）**：专用于最常见的**单结果重写（1-to-1 或 N-to-1）**场景。它的第二个参数只接收单个 `dag result`，内部自动用中括号包装为单元素列表 `[result]`。

```tablegen
// 方式 1：使用 Pat（最常用，语法简洁）
def ReshapeReshapeOptPattern : Pat<
    (ReshapeOp (ReshapeOp $arg)),
    (ReshapeOp $arg) // 单条 dag，无需加方括号
>;

// 方式 2：显式使用 Pattern 基类（完全等价）
def ReshapeReshapeOptPattern : Pattern<
    (ReshapeOp (ReshapeOp $arg)),
    [(ReshapeOp $arg)] // 必须使用中括号，表示 list<dag>
>;
```

### 2.3 DAG 模式语法解析

在 LLVM/MLIR TableGen DSL 中，`dag`（有向无环图）是最核心的数据类型，专门用来声明式描述 IR 算子语法树（AST/DAG）。

#### dag 节点三要素

`dag` 采用类似 Lisp 的 **S-表达式（S-expression）** 语法：

```
(operator argument1, argument2, ...)
```

一个 `dag` 节点由三部分组成：
1. **操作符（Operator）**：位于圆括号后的第一个元素（必须是 TableGen Record，如 `AddIOp` 或辅助控制符）。
2. **实参列表（Arguments）**：括号内由逗号分隔的子节点（可以是嵌套的 `dag`、变量符号、属性或常量）。
3. **标签/绑名（Tags/Names）**：以 `$` 开头的标识符（如 `$arg`、`:$res`），用来捕获并引用节点。

```
       (ReshapeOp:$res (ConstantOp $arg))
        │         │     │           │
        │         │     │           └── 标签 (绑定 ConstantOp 的属性值)
        │         │     └── 嵌套的子 dag (匹配第 0 个 Operand)
        │         └── 标签 (使用冒号绑定外层 ReshapeOp 本身)
        └── Operator (匹配的 Op 类)
```

#### 节点变量的三种绑定方式

| 绑定目标 | TableGen 语法示例 | 捕获内容与 C++ 映射 |
| :--- | :--- | :--- |
| **绑定输入参数（Operand）** | `(AddIOp $lhs, $rhs)` | `$lhs` 与 `$rhs` 绑定到匹配到的输入 `mlir::Value` |
| **绑定属性（Attribute）** | `(ConstantOp $attr)` | 当匹配目标位置为属性时，`$attr` 直接绑定对应的 `Attribute`（如 `DenseElementsAttr`） |
| **绑定算子本身（Operation）** | `(ReshapeOp:$res $arg)` | 使用 `:$res` 冒号语法，将整个匹配到的 `Operation*` 指针绑定到 `$res` 上，方便获取类型或位置（如 `$res.getType()`） |

> [!WARNING]
> **常见语法陷阱：`$res` vs `:$res`**
> - `$res` 放在参数位置时，匹配的是算子的**操作数（Operand）**或**属性（Attribute）**。
> - `:$res` 紧跟在算子类名后（使用冒号），绑定的是**算子实例本身（`Operation*`）**。

#### resultPatterns 中的特种 Operator

在替换模式中，`dag` 的 Operator 不仅可以是具体的 MLIR Op，还可以是 TableGen 预设的辅助控制符：

1. **`NativeCodeCall`（嵌入 C++ 计算）**：将底层 C++ 运算逻辑包装为 DAG 节点，在重写阶段计算新值：
   ```tablegen
   def ReshapeConstant : NativeCodeCall<"$0.reshape(($1.getType()).cast<ShapedType>())">;

   // 在 Result Pattern 中调用：
   (ConstantOp (ReshapeConstant $arg, $res))
   ```
2. **`replaceWithValue`（直接用已有 Value 替换）**：当优化只是消除冗余算子、直接返回输入时使用：
   ```tablegen
   // 匹配 Reshape(Reshape($x)) -> 直接替换为输入 $x
   def : Pat<(ReshapeOp (ReshapeOp $x)), (replaceWithValue $x)>;
   ```
3. **`location`（显式透传源码位置）**：默认情况下，新生成的 Op 会继承被替换根 Op 的源码位置（Location）。如需显式指定：
   ```tablegen
   (location $res) // 强制将新 Op 的 Location 设为 $res 的 Location
   ```

#### 底层代码生成：dag 如何展开为 C++

当你写下一个简单的 `dag` 重写规则时，`mlir-tblgen` 在后台展开为严谨的 C++ 命令式逻辑：

```tablegen
// TableGen 规则：
def : Pat<(ReshapeOp:$res (ConstantOp $arg)),
          (ConstantOp (ReshapeConstant $arg, $res))>;
```

展开后的 C++ 逻辑模拟：

```cpp
// 1. 匹配阶段 (Match)
LogicalResult match(Operation *op) {
    // 检查根节点是否为 ReshapeOp
    auto res = dyn_cast<toy::ReshapeOp>(op);
    if (!res) return failure();
    
    // 检查第 0 个 Operand 是否来自 ConstantOp
    auto constantOp = res.getOperand().getDefiningOp<toy::ConstantOp>();
    if (!constantOp) return failure();
    
    // 提取绑定的属性值 $arg
    auto arg = constantOp.getValue();
    return success();
}

// 2. 重写阶段 (Rewrite)
void rewrite(Operation *op, PatternRewriter &rewriter) {
    auto res = cast<toy::ReshapeOp>(op);
    auto constantOp = res.getOperand().getDefiningOp<toy::ConstantOp>();
    auto arg = constantOp.getValue();
    
    // 执行 NativeCodeCall 计算新属性
    auto newAttr = arg.reshape((res.getType()).cast<ShapedType>());
    
    // 原地创建新 ConstantOp 并替换旧 ReshapeOp
    auto newOp = rewriter.create<toy::ConstantOp>(res.getLoc(), newAttr);
    rewriter.replaceOp(res, newOp.getResult());
}
```

### 2.4 常量折叠原地替换

#### 为什么需要 NativeCodeCall 进行常量折叠？

TableGen 擅长做**算子图结构的模式匹配与拓扑替换**，但无法在编译期直接处理复杂的 C++ 数据运算（例如将一个 1D 数组的数据在二进制内存层面重排为 2D 矩阵）。

`NativeCodeCall` 充当了 TableGen 声明式语法与底层 C++ 运算之间的桥梁：

```tablegen
// 定义 C++ 代码内联宏：$0 映射为 $arg, $1 映射为 $res
def ReshapeConstant : NativeCodeCall<"$0.reshape(($1.getType()).cast<ShapedType>())">;

// 声明常量折叠重写规则
def FoldConstantReshapeOptPattern : Pat<
    (ReshapeOp:$res (ConstantOp $arg)),
    (ConstantOp (ReshapeConstant $arg, $res))
>;
```

#### 逐步折叠过程演练（Step-by-Step Walkthrough）

以一段连续三次 `reshape` 的初始 Toy IR 为例：

```mlir
// 初始 IR
module {
  toy.func @main() {
    %0 = toy.constant dense<[1.0, 2.0]> : tensor<2xf64>
    %1 = toy.reshape(%0 : tensor<2xf64>) to tensor<2x1xf64>
    %2 = toy.reshape(%1 : tensor<2x1xf64>) to tensor<2x1xf64>
    %3 = toy.reshape(%2 : tensor<2x1xf64>) to tensor<2x1xf64>
    toy.print %3 : tensor<2x1xf64>
    toy.return
  }
}
```

折叠演变流程：

1. **第 1 步：折叠 `%1`**
   - 匹配：`%1` 为 `reshape(%0)`，其输入 `%0` 是常量，命中重写模式。
   - 计算：C++ 将 `dense<[1.0, 2.0]>` 重排为 `dense<[[1.0], [2.0]]>`。
   - 替换：生成新常量 `%0_new`，并将下游所有使用 `%1` 的位置替换为 `%0_new`。
2. **第 2 步：折叠 `%2`**
   - 匹配：`%2` 是 `reshape(%0_new)`，输入再次为常量，再次命中重写模式。
   - 替换：生成新常量 `%0_new2` 替换 `%2`。
3. **第 3 步：折叠 `%3`**
   - 匹配：`%3` 变为 `reshape(%0_new2)`，再次命中模式。
   - 替换：生成最终常量 `%0_final` 替换 `%3`。
4. **第 4 步：死代码消除（DCE Cleanup）**
   - 规范化器（Canonicalizer）清理掉所有中间无用的 `reshape` 与旧 `constant`。

```mlir
// 最终优化后的 IR
module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00], [2.000000e+00]]> : tensor<2x1xf64>
    toy.print %0 : tensor<2x1xf64>
    toy.return
  }
}
```

#### 临界状态剖析：replaceOp 底层行为与死代码形成

在 Pattern Rewriting 刚刚结束、DCE 尚未介入的临界状态下，IR 中堆积了所有历史旧常量：

```mlir
module {
  toy.func @main() {
    // 💀 死代码 1：原始常量，%1 被替换后 %0 的引用计数降为 0
    %0 = toy.constant dense<[1.0, 2.0]> : tensor<2xf64>
    
    // 💀 死代码 2：折叠第 1 个 reshape 时生成的新常量，%2 被替换后失联
    %cst1 = toy.constant dense<[[1.0], [2.0]]> : tensor<2x1xf64>
    
    // 💀 死代码 3：折叠第 2 个 reshape 时生成的新常量，%3 被替换后失联
    %cst2 = toy.constant dense<[[1.0], [2.0]]> : tensor<2x1xf64>
    
    // ✅ 唯一有效节点：折叠第 3 个 reshape 时生成的最终常量
    %cst3 = toy.constant dense<[[1.0], [2.0]]> : tensor<2x1xf64>
    
    // ✅ 唯一有效使用者：打印最终结果
    toy.print %cst3 : tensor<2x1xf64>
    toy.return
  }
}
```

> [!NOTE]
> **为什么会出现旧常量堆积？**
> 1. **`replaceOp` 仅处理根节点**：执行 `rewriter.replaceOp(reshapeOp, newConstantOp)` 时，仅将下游引用重定向到新常量，并销毁 `reshapeOp` 本身，**不会递归向上回溯删除上游输入节点**。
> 2. **依赖 DCE 统一回收**：因为 `toy.constant` 声明了 `Pure` 特性，随后的 DCE Pass 遍历 Block 发现 `%0`、`%cst1`、`%cst2` 的用户列表为空且无副作用，便安全将它们一次性全部清除。

## 3. 跨方言函数内联架构

### 3.1 跨方言内联接口解耦

#### 背景问题与跨函数优化阻断

在单函数内部，MLIR 可以顺畅进行规范化（Canonicalize）与常量折叠。但在实际程序中，业务逻辑通常被拆分为独立的子函数调用（如 `toy.generic_call`）。

**函数调用的存在直接阻断了代码的数据流连续性**：
1. 调用点的返回值通常只能保留为动态形状（如 `tensor<*xf64>`）。
2. 跨函数边界的常量无法穿透折叠。
3. 跨函数边界的冗余算子（如 `transpose(transpose(x))`）无法被模式匹配识别。

解决这一问题的核心手段是将子函数直接展开到调用点，即**函数内联（Inlining）**。

#### 核心挑战：通用 Inliner Pass 与方言专有语义的解耦

MLIR 框架内置了一个通用函数内联优化 Pass（`-inline`），该 Pass 具有方言中立（Dialect-neutral）特性，**不直接感知任何具体方言的专有语义**（例如其并不预设 `toy.func` 或 `toy.generic_call` 的内部结构）。

当 `-inline` Pass 遍历 IR 遇到调用算子时，必须通过标准化接口完成三项核心校验：
1. **调用点合法性判定**：当前调用点（Call-site）是否允许被展开。
2. **被调指令合法性判定**：被调函数体内的具体指令移动至调用者所在区域（Region）是否合法。
3. **终结符与返回值衔接**：被调函数末尾的返回指令（如 `toy.return`）如何映射至调用点的返回值 SSA 变量。

```
┌────────────────────────────────┐
│   MLIR Generic -inline Pass    │ ◄── 通用内联优化算法引擎（方言无关）
└──────────────┬─────────────────┘
               │ 动态多态查询
               ▼
┌────────────────────────────────┐
│      DialectInlinerInterface   │ ◄── 方言注册的语义扩展接口
│  (如 ToyInlinerInterface)       │
└────────────────────────────────┘
```

为了实现算法与方言的解耦，MLIR 采用了**接口抽象（Interface）机制**：通用 Inliner 负责控制流遍历与 IR 变换框架，而具体的内联合法性判定与返回值重定向则委托给方言实现的 `DialectInlinerInterface`。

#### 为什么在 C++ 虚函数重写中标记 final？

```cpp
struct ToyInlinerInterface : public DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;

    bool isLegalToInline(Operation *call, Operation *callable,
                         bool wouldBeCloned) const final {
        return true;
    }
};
```

> [!NOTE]
> **C++11 `final` 的三大工程价值**：
> 1. **架构约束**：明确告知编译器与协作者当前类已完成最终实现，禁止派生类二次重写。
> 2. **去虚拟化优化（Devirtualization）**：编译器确信无后续子类更改此行为，可绕过虚函数表（vtable）间接寻址，优化为直接调用甚至直接内联，消除多态开销。
> 3. **代码自文档化（Self-Documenting）**：向阅读者清晰传达接口实现封顶的设计意图。

### 3.2 Inliner 调用时序演变

#### 源码场景与内联前 IR

以一个工具函数 `multiply_transpose` 被 `main` 函数调用的场景为例：

```toy
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  var a<2,3> = [[1, 2, 3], [4, 5, 6]];
  var b<2,3> = [[1, 2, 3], [4, 5, 6]];
  var c = multiply_transpose(a, b);
  print(c);
}
```

内联前的 MLIR IR 如下：

```mlir
// 1. 被调用的子函数
toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
  %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
  %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64>
  %2 = toy.mul %0, %1 : tensor<*xf64>
  toy.return %2 : tensor<*xf64>
}

// 2. 主函数
toy.func @main() {
  %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
  %1 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
  
  // 关键节点：跨函数调用，此时 %2 只能是未定形状 tensor<*xf64>
  %2 = toy.generic_call @multiply_transpose(%0, %1) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
  
  toy.print %2 : tensor<*xf64>
  toy.return
}
```

#### ToyInlinerInterface 接口交互协议与时序

当 PassManager 执行 `-inline` Pass 时，内联引擎与方言接口按以下协议时序协同执行：

1. **调用点合法性校验 (`isLegalToInline(call, callable, ...)` )**：
   - **输入**：`call`（`toy.generic_call`）、`callable`（`toy.func @multiply_transpose`）。
   - **判定**：Toy 方言中无递归约束与不可见链接属性，返回 `true`。
2. **指令级合法性校验 (`isLegalToInline(op, region, ...)` )**：
   - **输入**：遍历子函数体内的每条指令（`toy.transpose`、`toy.mul`）。
   - **判定**：指令均为纯张量计算，无跨硬件执行域冲突，返回 `true`。
3. **终结符处理与返回值重定向 (`handleTerminator(op, ...)` )**：
   - **输入**：遍历至子函数末尾的 `toy.return %2`。
   - **执行**：将调用者中 `toy.generic_call` 的返回值引用重定向至已内联生成的 `%2`，随后安全销毁该 `toy.return`。

#### 内联后的 IR 演变与跨函数形状推导

```mlir
toy.func @main() {
  %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
  %1 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
  
  // === 原 multiply_transpose 的函数体直接内联到此处 ===
  %2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %3 = toy.transpose(%1 : tensor<2x3xf64>) to tensor<3x2xf64>
  %4 = toy.mul %2, %3 : tensor<3x2xf64>
  // ===================================================
  
  // 原 generic_call 被替换为内联计算出的最终结果 %4
  toy.print %4 : tensor<3x2xf64>
  toy.return
}
```

> [!TIP]
> **优化价值**：内联完成后，原子函数若无其他引用将被 DCE 清除；同时，`main` 函数内暴露出了确切的静态张量类型（`tensor<3x2xf64>`），消除了跨函数调用的动态形状阻断，为后续的**跨算子形状推导（Shape Inference）**和**常量折叠**提供了完整的数据流上下文。

### 3.3 内联合法性断言防线

`DialectInlinerInterface` 提供了两个不同粒度的 `isLegalToInline` 重载函数，分别对应内联过程中的两道安全防线：

```
[发现 toy.generic_call]
         │
         ▼
Step 1: 宏观把关 isLegalToInline(call, callable, ...)
         │
         ├─► 返回 false ──► 放弃内联，保持原样
         │
         ▼ (返回 true)
Step 2: 微观把关 逐条遍历被调函数内部指令 (Op)
         │
         ├─► isLegalToInline(op, region, ...)
         │    │
         │    ├─► 只要有任何一条指令返回 false ──► 放弃内联，保持原样
         │    │
         ▼ (全部指令均返回 true)
Step 3: 执行内联操作，平铺代码、建立 SSA 映射并处理 Terminator
```

#### 宏观把关：isLegalToInline(call, callable, ...)

- **关注核心**：**调用点（Call-site）层级**。回答 *“能不能在这个特定的调用位置，把目标函数展开？”*
- **关键参数**：
  - `Operation *call`：当前调用算子（如 `toy.generic_call`）。
  - `Operation *callable`：被调用的函数算子（如 `toy.func`）。
  - `bool wouldBeCloned`：指示函数体是否会被克隆（若多次调用则为 `true`；若为私有函数且单次调用则为 `false`，原函数可直接移入）。
- **典型拒绝场景（返回 `false`）**：
  1. **防止递归无限展开**：若发现 `call` 所在的宿主函数与 `callable` 为同一个函数，必须返回 `false`。
  2. **特殊函数属性约束**：函数被标记了 `noinline` 属性或属于外部 C/Fortran 链接库。

#### 微观把关：isLegalToInline(op, region, ...)

- **关注核心**：**指令（Instruction）层级**。回答 *“被调函数体内的这条指令 `op`，搬到目标区域 `region` 中是否合法？”*
- **关键参数**：
  - `Operation *op`：被调函数内的具体单条指令（如 `toy.transpose` 或 `gpu.launch`）。
  - `Region *region`：目标放置区域（即调用者所在的 `Region`）。
  - `IRMapping &mapper`：内联器维护的值映射表（将旧 SSA Value 映射为调用者作用域中的新 SSA Value）。
- **典型拒绝场景（返回 `false`）**：
  1. **硬件/执行域限制（Domain Restriction）**：若被调函数内包含 GPU 专属同步指令 `gpu.barrier`，而调用者是运行在 CPU 上的普通函数，跨域搬运非法，必须返回 `false`。
  2. **嵌套结构约束**：某些特定指令只能存在于特定的父容器内（如并行算子必须包裹在 `scf.parallel` 内部）。

#### 为什么 Toy 语言中可以全部 return true？

在 Toy 语言的实现中：
1. 语言规范不支持递归调用，也没有复杂的外链接属性，因此调用点级无需拦截。
2. 所有的 Toy 算子均为纯 CPU 标量/张量计算，可无缝运行在任意 `toy.func` 的 Region 内，因此指令级也无需拦截。

## 4. Dialect Conversion 降级框架

### 4.1 算子模式降级机制

在 MLIR Toy 教程第 5 章中，`TransposeOpLowering` 标志着程序从高层语义抽象方言（Toy Dialect）走向贴近底层硬件方言（Affine / MemRef Dialect）的关键转变。

#### 降级核心任务与值语义到内存语义的转变

它的核心任务是：将语义化、不可变的 `toy.transpose` 张量转置算子，降级（Lowering）为显式的物理内存分配（`memref.alloc`）与嵌套循环数据搬运（`affine.for` + `affine.load`/`store`）。

```
┌────────────────────────────────────────────────────────────┐
│ 降级前 (值语义 / 数据不可变)                                 │
│ %1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>│
└─────────────────────────────┬──────────────────────────────┘
                              │ Dialect Conversion 降级
                              ▼
┌────────────────────────────────────────────────────────────┐
│ 降级后 (内存语义 / 显式指针与循环)                           │
│ %alloc = memref.alloc() : memref<3x2xf64>                  │
│ affine.for %arg1 = 0 to 3 {       // 外层循环 j             │
│   affine.for %arg2 = 0 to 2 {     // 内层循环 i             │
│     %val = affine.load %input[%arg2, %arg1]                │
│     affine.store %val, %alloc[%arg1, %arg2]                │
│   }                                                        │
│ }                                                          │
└────────────────────────────────────────────────────────────┘
```

#### 官方 lowerOpToLoops 骨架与 Lambda 解耦设计

为了避免在 `AddOp`、`MulOp`、`TransposeOp` 等各类算子降级时重复编写物理内存分配与多维循环嵌套的样板逻辑，官方实现将通用流程抽象为 `lowerOpToLoops` 骨架函数。

```cpp
// 官方 TransposeOpLowering 实现
struct TransposeOpLowering : public OpConversionPattern<toy::TransposeOp> {
    using OpConversionPattern<toy::TransposeOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        toy::TransposeOp op, OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const final {
        auto loc = op->getLoc();
        
        // 核心解耦：通过 Lambda 回调仅表达“如何在当前循环变量下读取输入数据”
        lowerOpToLoops(op, rewriter,
            [&](OpBuilder &builder, ValueRange loopIvs) {
                Value input = adaptor.getInput();
                
                // 巧妙利用 llvm::reverse 将循环索引 [i, j] 原地翻转为 [j, i]
                SmallVector<Value, 2> reverseIvs(llvm::reverse(loopIvs));
                
                // 返回读出来的数值，外层的 Alloc 与 Store 全部交由骨架处理
                return affine::AffineLoadOp::create(builder, loc, input, reverseIvs);
            });
            
        return success();
    }
};
```

#### llvm::reverse 零拷贝索引翻转

- **`loopIvs`（Loop Induction Variables）**：`lowerOpToLoops` 传给 Lambda 的当前循环迭代变量列表。对于 2 维输出矩阵，`loopIvs` 即为 `[i, j]`（代表输出矩阵的写入坐标）。
- **`llvm::reverse(loopIvs)`**：转置的数学定义为将原矩阵 $(j, i)$ 位置的元素写入新矩阵 $(i, j)$。`llvm::reverse` 对传入的指针范围进行反向遍历并填入栈上的 `SmallVector`，**零堆内存分配**，高效完成转置寻址。

### 4.2 转换模式三大要素

#### OpConversionPattern 与普通 OpRewritePattern 的区别

- **`OpRewritePattern`**：通常用于同一个 Dialect 内部的局部图重写与规范化（如 Canonicalize）。
- **`OpConversionPattern`**：专门用于**跨 Dialect、跨类型系统**的方言降级。它内部绑定了 `TypeConverter`（类型转换器），能够感知整个编译环境中类型的映射关系（如自动将 `tensor<2x3xf64>` 映射为 `memref<2x3xf64>`）。

#### C++11 using 继承构造函数机制

```cpp
using OpConversionPattern<toy::TransposeOp>::OpConversionPattern;
```

这是 C++11 的继承构造函数语法（`using Base::Base;`）。它的作用是直接继承父类 `OpConversionPattern` 内部已经定义好的复杂构造函数（接收 `TypeConverter&`、`MLIRContext*`、`PatternBenefit` 等框架对象），免去子类手写无意义的模板转发代码。

#### matchAndRewrite 三大核心参数（op vs adaptor vs rewriter）

```cpp
LogicalResult matchAndRewrite(
    toy::TransposeOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const final;
```

参数签名明确划分了“降级前的原始 IR 状态”与“降级后的已转换状态”：

| 参数 | 代表角色 | 核心职责与底层细节 |
| :--- | :--- | :--- |
| `toy::TransposeOp op` | **转换前的原始算子（Pre-lowering Op）** | 当前准备被降级替换掉的**原始 Toy 算子**。它保留着降级前的旧类型（如 `tensor`）。用于获取源码位置 `op->getLoc()`、原始属性（Attributes）或原结果类型。 |
| `OpAdaptor adaptor` | **转换后的操作数适配器（Lowered Adaptor）** | **核心关键**。在降级流水线中，上游算子可能已被先一步降级。`op.getInput()` 拿到的依然是旧类型 `tensor`，而 `adaptor.getInput()` 拿到的是被上游转换后的新类型 `memref` 指针。`OpAdaptor` 是 TableGen 自动生成的强类型包装器。 |
| `ConversionPatternRewriter &rewriter` | **降级专用重写器** | 继承自 `PatternRewriter`。除了创建新算子与删除旧算子外，在后台维护类型转换账本；若出现类型断层，会自动插入临时转换指令（如 `builtin.unrealized_conversion_cast`）。 |

> [!IMPORTANT]
> **为什么有了 `op` 还需要 `adaptor`？**
> `op.getOperand(0)` 仅指向旧 IR 树中的节点；而 `adaptor.getOperand(0)` 自动解析了 `TypeConverter` 转换后的最新操作数指针。在 Lowering Pass 中读取输入数据时，**必须使用 `adaptor`**。

#### const final 修饰符与 Devirtualization 优化

- `const`：声明该 Pattern 对象的匹配重写逻辑是无状态且只读的。
- `final`：阻止后续子类进一步重写该虚函数，帮助 C++ 编译器进行去虚拟化（Devirtualization）优化，直接消除虚函数表调用的开销。

### 4.3 插入点光标生命周期

#### 插入点（Insertion Point）位置与代码生成时间线

很多初学者容易误以为新生成的 `alloc` 和 `affine.for` 是插入在 `transpose` 后面的，但实际机制是：**Rewriter 默认的插入点正好位于被匹配到的 `toy.transpose` 的正上方**。

```
【初始状态】
... -> [toy.transpose] -> [toy.print] ...

【步骤 1：在 transpose 前方插入 alloc 和 affine.for 循环】
... -> [%alloc = memref.alloc] -> [affine.for 循环群] -> [toy.transpose] -> [toy.print] ...

【步骤 2：执行 rewriter.replaceOp(op, alloc)，接管引用并销毁旧算子】
... -> [%alloc = memref.alloc] -> [affine.for 循环群] -> [toy.print (输入重定向为 %alloc)] ...
```

#### replaceOp 底层行为解构（RAUW + 节点销毁）

`replaceOp` 并不是原地文本替换，而是拆解为两个独立的底层动作：
1. **重定向引用（RAUW: Replace All Uses With）**：遍历整个 IR 块，将所有原本将 `transpose` 输出结果（`%1`）作为输入的下游算子（如 `toy.print %1`），全部重定向为输入新内存句柄 `%alloc`。
2. **安全擦除旧算子（Erase Operation）**：调用底层代码将 `toy.transpose` 节点从 Block 的双向链表中彻底摘除并销毁（`op->erase()`）。

#### MLIR 替换与改写 API 家族全景对比

| 函数 API | 是否创建新 Op？ | 是否替换下游 Use (RAUW)？ | 是否销毁旧 Op？ | 适用场景 |
| :--- | :---: | :---: | :---: | :--- |
| `replaceOp` | ❌（使用前文已建好的 Value） | ✅ | ✅ | 绝大多数 1-to-1 / N-to-1 算子重写 |
| `replaceOpWithNewOp<Op>` | ✅（在旧 Op 位置创建） | ✅ | ✅ | 一步到位将 `OpA` 原位替换为 `OpB` |
| `replaceAllUsesWith` | ❌ | ✅ | ❌ | 只想重定向数据流，旧 Op 留作他用 |
| `modifyOpInPlace` | ❌ | ❌（仅内部修改） | ❌ | 仅修改旧 Op 的 Operand 或 Attribute |
| `eraseOp` | ❌ | ❌ | ✅ | 明确无用的死代码清理（需满足 UseCount == 0） |

### 4.4 递归转换驱动引擎

```cpp
void ToyToAffineLoweringPass::runOnOperation() {
    // ...
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, patterns)))
        signalPassFailure();
}
```

#### Operation 层次树状容器结构

在 MLIR 中，顶层容器（如 `ModuleOp` 或 `FuncOp`）本身就是一个 `Operation`。它通过 `Region` 和 `Block` 嵌套包含了程序中的所有子算子：

```
getOperation() 返回的根节点 (ModuleOp)
└── Region
    └── Block
        ├── toy.func (Operation)
        │   └── Region
        │       └── Block
        │           ├── toy.transpose (Operation)
        │           ├── toy.mul (Operation)
        │           └── toy.return (Operation)
        └── toy.func (Operation) ...
```

#### applyPartialConversion 深度优先遍历与目标校验

当把 `getOperation()` 传给转换驱动时，框架在底层执行以下流程：
1. **深度优先遍历（Recursive Walk）**：以根 `Operation*` 为起点，递归遍历其内部所有的 Region、Block 和子 Operation。
2. **合法性检查（Target Check）**：对遇到的每条子指令询问 `ConversionTarget` 是否合法。
3. **模式匹配与降级**：若算子被标记为非法（如深层的 `toy.transpose`），框架自动从 `patterns` 集合中找到对应的 Pattern 进行降级。

#### 三大 Conversion API 对比

| API | 行为特征 | 适用场景 |
| :--- | :--- | :--- |
| **`applyPartialConversion`** | **部分转换**：允许 IR 中保留未转换的合法算子，只要所有被标记为 Illegal 的算子都成功降级即可。若有 Illegal 残留则返回失败。 | **最常用的分阶段 Pass 降级入口**（如 Toy 逐步降级到 Affine/MemRef）。 |
| **`applyFullConversion`** | **完全转换**：极其严格。转换结束后，整个 IR 树中绝对不能留有任何旧 Dialect 的算子，否则宣告失败。 | 面向 LLVM IR 或目标硬件方言的最终代码生成阶段。 |
| **`applyAnalysisConversion`** | **仅分析不改写**：只试探性执行匹配流程，检查 IR 是否能够被成功转换，不实际修改 IR 结构。 | 用于 Pass 内部的分析与条件决策。 |

## 5. 全局符号系统与光标控制

### 5.1 SymbolTable 符号解析管理

MLIR 的 **Symbol 机制** 是专门用于在不同的 Operation 之间建立跨作用域名称引用（如函数调用、全局变量访问）的抽象层。为了避免直接使用 C++ 指针导致的跨区域生命周期混乱，MLIR 借由定义端、引用端与容器端三者协同工作。

```
       ┌────────────────────────────────────────────────────────────┐
       │                 MLIRContext (类型与属性内存池)               │
       │                                                            │
       │            SymbolRefAttr::get(context, "printf")           │
       └─────────────────────────────┬──────────────────────────────┘
                                     │ 返回享元属性句柄 "@printf"
                                     ▼
        ┌────────────────────────────┴─────────────────────────────┐
        ▼                                                          ▼
  LLVM::CallOp(..., "@printf")                             LLVM::LLVMFuncOp
  (引用端/消费者：保存符号名字面量)                        (定义端/生产者：sym_name = "printf")
```

#### 符号体系的三大核心要素

| 要素 | 扮演角色 | 核心载体 / API | 作用与示例 |
| :--- | :--- | :--- | :--- |
| **定义端** | 符号生产者 | `SymbolOpInterface` | 实现了该接口的 Op（如 `func.func`, `llvm.func`）必须拥有一个 `sym_name` 属性，声明自己的符号名称（如 `sym_name = "printf"`）。 |
| **引用端** | 符号消费者 | `SymbolRefAttr` | 想要使用该符号的 Op（如 `func.call`, `llvm.call`）不直接持有目标内存指针，而是保存一个 `SymbolRefAttr` 字符串属性（如 `@printf`）。 |
| **容器端** | 符号管理者 | `SymbolTable` | 充当符号命名空间的容器（如 `ModuleOp`），负责在其内部的 Region/Block 树中维护符号的唯一性与映射表。 |

#### SymbolTable 哈希缓存与嵌套作用域解析

直接在 `ModuleOp` 内部遍历指令链表查找 `sym_name` 的时间复杂度为 $O(N)$。`SymbolTable` 提供了高效的管理机制：

1. **$O(1)$ 哈希映射表缓存（Caching）**：
   - 用 `SymbolTable symbolTable(module)` 包装 Module 时，会在后台维护 `DenseMap<StringAttr, Operation*>`，将查找时间降至 $O(1)$。
   - 插入新符号时若发生重名，`symbolTable.insert(op)` 可自动生成唯一后缀（如 `printf_0`）避免冲突。
2. **嵌套作用域解析（Nested Symbol Resolution）**：
   - 支持多层嵌套模块（如 GPU 编程中 `gpu.module` 嵌套在 `ModuleOp` 内部），通过层级路径语法（如 `@GPUModule::@my_kernel`）逐级向下解析。
3. **全局引用安全重命名**：
   - 当重构或删除符号时，`SymbolTable::replaceAllSymbolUses(oldOp, newName, module)` 会自动遍历整个 Module 视图，一次性更新所有持有该符号的引用者，彻底杜绝悬空引用（Dangling Reference）。

#### 符号引用的享元创建（FlatSymbolRefAttr Uniquing）

- `FlatSymbolRefAttr` 本质上是一个被包裹成属性（Attribute）的字符串。
- 得益于 `MLIRContext` 的享元模式（Uniquing），多次调用 `SymbolRefAttr::get(context, "printf")` 都会从上下文缓存中直接返回指向同一个单例属性的轻量句柄。

### 5.2 外部辅助函数惰性声明

在 Toy 教程第 6 章中，当要把 `toy.print` 降级为调用 C 标准库的 `printf` 时，必须确保外层 Module 中存在 `printf` 的函数声明，并返回其符号引用。

```cpp
static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                           ModuleOp module,
                                           LLVM::LLVMDialect *llvmDialect) {
    auto *context = module.getContext();
    
    // Step 1: 查重（检查 Module 中是否已有 printf 声明）
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
        return SymbolRefAttr::get(context, "printf");
        
    // Step 2: 用 LLVM Dialect 构造 printf 签名：i32 (i8*, ...)
    auto llvmI32Ty = IntegerType::get(context, 32);
    auto llvmI8PtrTy = LLVM::LLVMPointerType::get(context);
    auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI32Ty, llvmI8PtrTy,
                                                  /*isVarArg=*/true);
                                                  
    // Step 3: 使用 RAII 保护当前 Rewriter 光标位置
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    
    // Step 4: 将插入光标切至 Module 头部
    rewriter.setInsertionPointToStart(module.getBody());
    
    // Step 5: 创建函数声明 (因无 Block 函数体，自动生成 declare/extern 声明)
    LLVM::LLVMFuncOp::create(rewriter, module.getLoc(), "printf", llvmFnType);
    
    // Step 6: 离开作用域，insertGuard 析构，光标自动还原至原位置
    return SymbolRefAttr::get(context, "printf");
}
```

#### 场景动机与标准范式

该辅助函数展示了在 MLIR 中按需插入全局辅助声明的标准范式（Idiom）：
1. **查重**（`lookupSymbol` 检查是否存在）
2. **保护并切换光标**（借助 `InsertionGuard` 切换至 Module 头部）
3. **构建类型并声明**（创建 `LLVMFuncOp`）
4. **自动复位光标并返回符号**（供调用点 `LLVM::CallOp` 绑定）

#### 使用 LLVM Dialect 构建 C 函数签名

C 语言标准库中的原型为 `int printf(const char *format, ...);`，在 MLIR 中通过 LLVM 方言精确复刻：
- 返回值：`IntegerType::get(context, 32)`（`i32`）
- 首参数：`LLVM::LLVMPointerType::get(context)`（不透明指针 `ptr`，对应 `char*`）
- `/*isVarArg=*/true`：明确声明该函数支持 C 风格变长参数。

#### MLIR LLVM 方言与 LLVM 原生 API 的桥接边界

| 维度 | MLIR LLVM 方言（编译 Pass 期间使用） | LLVM 原生 C++ API（后端机器码生成期间） |
| :--- | :--- | :--- |
| **C++ 命名空间** | `mlir::LLVM::*` | `llvm::*` |
| **指针类型** | `mlir::LLVM::LLVMPointerType` | `llvm::PointerType` |
| **接收 Context** | `mlir::MLIRContext*` | `llvm::LLVMContext&` |
| **基类** | 继承自 `mlir::Type` | 继承自 `llvm::Type` |
| **用途** | 在 MLIR 中表达、变换与优化 LLVM 语义的 IR | 最终生成汇编（`.s`）或二进制目标文件（`.o`） |

> **桥接转换**：在降级 Pass 全部完成后，调用 `mlir::translateModuleToLLVMIR(mlirModule, llvmContext)` 统一将 MLIR 的 LLVM Dialect 树翻译为原生 `llvm::Module`。

### 5.3 InsertionGuard 光标保护

`PatternRewriter::InsertionGuard`（继承自 `OpBuilder::InsertionGuard`）利用 C++ RAII 机制管理 Rewriter 的插入游标（Insertion Point）。

#### 隐式状态痛点与手动恢复的异常隐患

在生成辅助指令时，必须手动将光标切到其他位置（如 Module 头部）。

- **若无 Guard（存在状态泄露隐患）**：
  ```cpp
  // 必须手动保存
  auto point = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(module.getBody());
  if (failed(someCheck())) {
      return failure(); // 异常隐患：提前 return 导致光标未复位，污染后续 Pass 插入点
  }
  rewriter.restoreInsertionPoint(point); // 容易漏写
  ```
- **使用 Guard（安全优雅）**：
  ```cpp
  {
      PatternRewriter::InsertionGuard guard(rewriter); // 构造时自动记住旧光标
      rewriter.setInsertionPointToStart(module.getBody());
      // ... 任意分支或提前 return ...
  } // 离开作用域时析构，自动复位至进入前的位置
  ```

#### InsertionGuard 的 C++ RAII 底层实现模拟

```cpp
namespace mlir {
class OpBuilder::InsertionGuard {
public:
    explicit InsertionGuard(OpBuilder &builder)
        : builder(builder), savedPoint(builder.saveInsertionPoint()) {}
        
    ~InsertionGuard() {
        builder.restoreInsertionPoint(savedPoint);
    }
private:
    OpBuilder &builder;
    OpBuilder::InsertPoint savedPoint; // 保存 Block* 指针与迭代器
};
}
```

利用 C++ 栈对象的确定性生命周期，即便发生分支提前退出或异常栈展开，析构函数必定执行，确保强异常安全性。

#### 三大典型应用场景与作用域精细控制

1. **在 Module / Function 头部插入全局辅助声明**（如动态插入 `printf` 或全局常量）。
2. **构建嵌套控制流区域**（如创建 `scf.for` / `affine.for` 或 `if-else` 的内层 Body Block）。
3. **插入探针与临时调试指令**（如 Profile 测量插桩）。

> [!TIP]
> **局部花括号最佳实践**：可通过独立的 C++ 局部花括号 `{ PatternRewriter::InsertionGuard guard(rewriter); ... }` 精确限制 Guard 的生命周期，离开花括号后立即复位光标，无需等到整个函数结束。

### 5.4 OpBuilder 继承体系设计

#### MLIRContext 与 ModuleOp 的宿主从属关系

```
┌────────────────────────────────────────────────────────┐
│ MLIRContext (全局基础设施容器 / 上下文环境)              │
│ ├─ 已注册的 Dialects (Toy, LLVM, Affine, MemRef...)    │
│ ├─ 全局类型与属性单例池 (Types & Attributes Uniquing)   │
│ ├─ 多线程任务池 (ThreadPool)                            │
│ │                                                      │
│ ├── ModuleOp #1 (顶层 IR 树 1) ─── func @main          │
│ └── ModuleOp #2 (独立 IR 树 2) ─── func @helper        │
└────────────────────────────────────────────────────────┘
```

- **`MLIRContext`**：全局资源管理器与单例工厂，贯穿整个编译 Pass 流水线。
- **`ModuleOp`**：由 Context 创建并托管的 IR 顶层结构节点。`module.getContext()` 是向上查询其所属的全局上下文。
- **数量关系**：单文件对应 1 个顶层 Module；GPU/异构编译中支持 Module 嵌套；JIT 执行引擎中可存在多个平行的独立 Module 共享同一 Context。

#### 为什么 LLVM 与 MLIR 各自拥有独立的 IntegerType？

1. **语义表达能力差异（Signedness）**：
   - LLVM `llvm::IntegerType`：**无符号特质（Signless）**，整数只有位宽（如 `i32`），正负号语义押后至算子层面（`sdiv` vs `udiv`）。
   - MLIR `mlir::IntegerType`：**显式三态符号**（`Signless i32`, `Signed si32`, `Unsigned ui32`），用于高级语言类型检查与硬件描述建模，直至降级到 LLVM 方言时才转为 Signless。
2. **上下文与内存池隔离**：LLVM 类型属于 `llvm::LLVMContext`，MLIR 类型属于 `mlir::MLIRContext`。解耦保证 MLIR 可作为通用元框架独立使用，无需强行绑定庞大的 LLVMContext。
3. **下游目标不仅是 LLVM**：MLIR 的定位是 *Anything to Anything*，还可直接生成 SPIR-V、C/C++ 或 Verilog/FIRRTL（CIRCT）。

#### 不透明指针（Opaque Pointers）演进与 API 差异

- **旧版 MLIR (LLVM $\le$ 14)**：带类型指针（Typed Pointers），如 `!llvm.ptr<i8>`，API 为 `LLVMPointerType::get(Type elementType)`。
- **现代 MLIR (LLVM $\ge$ 15+)**：不透明指针（Opaque Pointers），如 `!llvm.ptr`，API 简化为 `LLVMPointerType::get(MLIRContext *context)`。彻底移除了多余的指针类型嵌套与 bitcast 开销。

#### OpBuilder 家族单继承树与多态转换

```
  ┌────────────────────────┐
  │       OpBuilder        │ <── 最基础的构建器，提供 create<Op>、setInsertionPoint、InsertionGuard
  └───────────┬────────────┘
              │ 继承
              ▼
  ┌────────────────────────┐
  │    PatternRewriter     │ <── 增加驱动监听（Listener）、replaceOp、eraseOp、undo 记录
  └───────────┬────────────┘
              │ 继承
              ▼
┌──────────────────────────────────────┐
│      ConversionPatternRewriter       │ <── 增加类型转换账本（TypeConverter）
└──────────────────────────────────────┘
```

- **`OpBuilder`（父类）**：命令式修改 IR，不感知 Pattern Driver 的调度。
- **`PatternRewriter`（子类）**：通过内部 Listener 将修改实时通知给 `GreedyRewriteDriver` 的工作队列（Worklist），确保模式迭代收敛。
- **多态兼容**：`InsertionGuard` 接收 `OpBuilder&`，因此在所有子类（`PatternRewriter`、`ConversionPatternRewriter`）中均可无缝直接使用。

### 5.5 模式集注册与上下文指针生命周期

在 Pass 的 `runOnOperation` 中，通常会看到如下模式集合初始化与批量注入代码：

```cpp
mlir::RewritePatternSet patterns(&getContext());

// 批量注入各类方言的官方标准降级规则
mlir::populateAffineToStdConversionPatterns(patterns, &getContext());
mlir::cf::populateSCFToControlFlowConversionPatterns(patterns, &getContext());
mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);
mlir::cf::populateControlFlowToLLVMConversionPatterns(patterns, &getContext());

// 注入用户自定义降级规则
patterns.add<PrintOpLowering>(&getContext());
```

#### populate* 批量注入范式

- 一个完整的 Dialect（如 Affine）内部包含几十种算子（`affine.for`, `affine.if`, `affine.load`, `affine.store` 等）。
- 若要求开发者逐一手动 `patterns.add<...>()`，会产生大量样板代码。
- 官方库将同一转换阶段的整套规则打包为 `populate...Patterns` 函数（Idiom），供 Pass 批量注册到 `patterns` 容器中。

#### 为什么传 MLIRContext 指针而非引用？

`getContext()` 返回的是当前全局上下文的引用 `MLIRContext&`，通过 `&getContext()` 取地址转为指针 `MLIRContext*` 传给容器和模式类。

1. **C++ 成员变量赋值与移动语义（最核心原因）**：
   - `RewritePatternSet` 和 `Pattern` 必须在内部保存上下文句柄。
   - 若在类内声明引用成员（`MLIRContext &context;`），C++ 会直接禁用默认的赋值运算符（`operator=`）与移动赋值，导致这些类无法放入 `std::vector` 等容器中进行动态扩容或重赋值。
   - 改为保存指针（`MLIRContext *context;`）后，类可以自由进行拷贝、移动和赋值。
2. **遵循 LLVM 编码规范**：
   - 值语义轻量句柄（如 `Type`, `Value`, `Attribute`）使用传值（By Value）或传引用（By Reference）。
   - 环境/全局大对象身份（如 `MLIRContext*`, `Operation*`, `TypeConverter*`）一律使用传指针（By Pointer），明确表达借用全局地址的意图。

## 6. 自定义类型解析与存储架构

### 6.1 文本解析短路状态机

MLIR 中所有的文本语法解析函数均返回 `ParseResult`。

```cpp
// NOTE: All MLIR parser functions return a ParseResult. This is a 
// specialization of LogicalResult that auto-converts to a `true` boolean 
// value on failure to allow for chaining...
```

#### ParseResult 的短路求值设计（失败为 true）

手写编译器的语法解析器充斥着大量顺序递进逻辑（必须依次解析类型、名称、等号、表达式，任何一步失败均需立即中断）。

MLIR 特意将 `ParseResult` 在隐式转换为 `bool` 时映射为：
- **解析失败（Failure） $\rightarrow$ 转换为 `true`**
- **解析成功（Success） $\rightarrow$ 转换为 `false`**

#### 单行链式检查 vs 传统多层嵌套

利用 C++ 逻辑或运算符 `||` 的**短路求值（Short-circuit evaluation）**特性（只要前面的表达式为 `true`，后续表达式直接跳过），繁琐的嵌套判断可优雅压缩为一行：

```cpp
// ✅ 极其优雅的单行链式短路写法
if (parseType() || parseName() || parseEqual() || parseExpression())
    return failure();
```

- 若 `parseType()` 成功（返回 `false`），`||` 继续求值 `parseName()`；
- 若 `parseName()` 失败（返回 `true`），C++ 立即短路，跳过后续的 `parseEqual` 和 `parseExpression`，直接进入 `if` 体内返回 `failure()`。

#### 空类型句柄（Null Type）哨兵值机制

当自定义类型解析函数的返回值声明为 `mlir::Type` 而非 `ParseResult` 时：

```cpp
Type parseStructType(AsmParser &parser) {
    if (parser.parseKeyword("struct") || parser.parseLess())
        return Type(); // 必须返回 Type()，利用空对象表达失败
        
    // ... 解析成功 ...
    return StructType::get(context, elementTypes);
}
```

- **`Type()` 的本质是空指针**：默认构造的 `Type()` 内部持有 `nullptr`，在布尔上下文中隐式转换为 `false`。
- 上游框架通过简单的指针判空 `if (!t) return failure();` 即可感知解析状态（经典 Null Object Pattern）。

### 6.2 TypeStorage 存储实体抽象

为了保证类型在整个编译期间**完全不可变（Immutable）且全局唯一（Uniqued）**，MLIR 将类型拆分为两层：

```
      ┌────────────────────────────────────────────────────────────┐
      │               class StructType (句柄 / Class)              │
      │────────────────────────────────────────────────────────────│
      │ - 仅持有一个指针 (sizeof(void*) = 8 字节)                   │
      │ - 按值传递 (Pass by Value)，支持自由复制                    │
      │ - 提供 getElementTypes() 等高层业务 API                    │
      └─────────────────────────────┬──────────────────────────────┘
                                    │ 内部封装指针指向
                                    ▼
      ┌────────────────────────────────────────────────────────────┐
      │         struct StructTypeStorage (实体 / Body / Storage)    │
      │────────────────────────────────────────────────────────────│
      │ - elementTypes : llvm::ArrayRef<mlir::Type>                │
      │ - 托管在 MLIRContext 的竞技场内存池 (Arena) 中               │
      │ - 全局单例、禁止拷贝、随 Context 批量释放                     │
      └────────────────────────────────────────────────────────────┘
```

#### StructTypeStorage 的五大核心结构

打开自定义类型的实现，其标准骨架包含五个关键组成部分：

```cpp
struct StructTypeStorage : public mlir::TypeStorage {
    // 1. 唯一化查找键 (Key)：由内部包含的子元素类型数组唯一标识
    using KeyTy = llvm::ArrayRef<mlir::Type>;

    // 2. 构造函数
    StructTypeStorage(llvm::ArrayRef<mlir::Type> elementTypes)
        : elementTypes(elementTypes) {}

    // 3. 比较逻辑：判断传入的 Key 与当前 Storage 是否完全相同
    bool operator==(const KeyTy &key) const {
        return key == elementTypes;
    }

    // 4. 哈希计算：用于 MLIRContext 内部哈希表定位
    static llvm::hash_code hashKey(const KeyTy &key) {
        return llvm::hash_combine_range(key.begin(), key.end());
    }

    // 5. 构造与深拷贝：在 Context 内存池中原地创建自身
    static StructTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                        const KeyTy &key) {
        // 将临时数组深拷贝到 Context 的 BumpPtrAllocator 内存区中
        llvm::ArrayRef<mlir::Type> elementTypes = allocator.copyInto(key);
        
        // 在内存池切出的裸内存上执行 Placement New 构造
        return new (allocator.allocate<StructTypeStorage>())
            StructTypeStorage(elementTypes);
    }

    llvm::ArrayRef<mlir::Type> elementTypes; // 真正保存的数据
};
```

#### StructType::get 唯一化生命周期全流程

```
调用 StructType::get({i32, f64})
         │
         ▼
封装为 KeyTy ({i32, f64})并在 MLIRContext 哈希表中查询
         │
         ├─►【哈希表中已存在】──► 直接返回现有 Storage 的轻量包装句柄
         │
         ▼【哈希表中不存在】
触发 Storage::construct:
  1. allocator.copyInto(key) 深拷贝元素数组到 Arena 池
  2. allocator.allocate 切出裸内存并 Placement New 构造
  3. 将新 Storage 指针注册进哈希表
         │
         ▼
返回 StructType 句柄
```

> [!TIP]
> **性能优势**：无论在代码中调用多少次 `StructType::get(context, {i32, f64})`，返回的底层 `StructTypeStorage*` 永远指向同一个内存地址。判断两个结构体类型是否完全相同只需执行简单的**指针地址对比**（`typeA == typeB`），时间复杂度为 $O(1)$。

### 6.3 BumpPtrAllocator 内存池模型

在大型编译任务中，IR 包含数以百万计的微小类型、属性和节点。传统的 `malloc/free` 或 `std::shared_ptr` 会导致大量内存碎片并产生高昂的单个析构/链表维护开销。

MLIR 采用了 **Arena 内存池 + BumpPtrAllocator + Placement New** 的协同内存管理架构：

```
[ Slab 连续大内存块 (例如 64KB) ]
┌──────────────────────────┬─────────────────────────────────┐
│ 已分配的 TypeStorage 内存  │     尚未使用的裸内存 (Raw Bytes)  │
└──────────────────────────┴─────────────────────────────────┘
                            ▲
                          CurPtr (每次分配仅向右“碰撞”平移 N 字节)
```

#### BumpPtrAllocator 碰撞指针分配器原理

1. **结构**：向操作系统申请若干个大型连续内存块（Slab，如 4KB 或 64KB），维护一个边界指针 `CurPtr`。
2. **分配逻辑**：
   - 检查当前 Slab 剩余空间是否足够 $N$ 字节；
   - 若足够，直接执行汇编级加法 `CurPtr += N`，并返回旧指针地址。
3. **性能**：分配时间复杂度严格为 $O(1)$，效率逼近在 C++ 函数栈上分配局部变量。

#### 基于裸内存的 Placement New 原地对象构造

`BumpPtrAllocator` 分配出的只是无类型的裸字节缓冲区（Uninitialized Raw Memory）。通过 C++ 的 `Placement New` 语法，在不调用系统 `malloc` 的前提下，直接在这块现有内存地址上显式触发构造函数：

```cpp
// 1. 碰撞指针平移切出裸内存
void *mem = allocator.allocate(sizeof(StructTypeStorage), alignof(StructTypeStorage));

// 2. 原地构造 C++ 对象
StructTypeStorage *storage = new (mem) StructTypeStorage(elementTypes);
```

#### Arena 生命周期：零单体析构与 O(1) 批量释放

- **运行期间零析构**：所有的 `TypeStorage` 全局唯一且只读，运行期间从不单独调用 `delete`，无需维护复杂的引用计数或空闲链表。
- **销毁时 $O(1)$ 批量归还**：当整个 `MLIRContext` 析构时，`BumpPtrAllocator` 将所有的 Slab 内存大块一次性整体交还给操作系统。**数以百万计对象的清理开销降为 $O(1)$**。

### 6.4 ODS 声明式类型代码生成

在现代 MLIR 中，无需手写上述繁琐的 `TypeStorage` 样板代码。通过 TableGen 的 ODS（Operation/Type Definition System）声明，`mlir-tblgen` 可自动生成完整的 C++ 类与存储逻辑。

#### 在 TableGen (.td) 文件中声明 TypeDef

```tablegen
include "mlir/IR/AttrTypeBase.td"

// 1. 定义方言的 Type 通用基类
class Toy_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Toy_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

// 2. 声明具体的 StructType
def Toy_StructType : Toy_Type<"Struct", "struct"> {
    let summary = "Toy struct type";
    let description = [{
        A struct type containing an array of element types.
        Syntax in IR: !toy.struct<i32, f64>
    }];

    // 核心：指定存储字段（自动生成 TypeStorage 成员与 KeyTy）
    let parameters = (ins ArrayRefParameter<"mlir::Type">: $elementTypes);

    // 声明式语法：自动生成 Parser 与 Printer 代码
    let assemblyFormat = "`<` $elementTypes `>`";
}
```

#### ODS 语法关键字段解析

- `TypeDef<Dialect, ClassName>`：指定所属方言及生成的 C++ 类名（如 `StructType`）。
- `mnemonic`：在 MLIR 文本 IR 中的类型前缀（如 `!toy.struct<...>` 中的 `"struct"`）。
- `parameters`：用 `(ins ...)` 描述成员变量。常用参数类型包括 `ArrayRefParameter<"mlir::Type">`、`StringRefParameter` 等。
- `assemblyFormat`：声明文本 IR 的打印与解析格式规则（反引号包裹字面量，`$elementTypes` 对应参数名）。

#### CMake 配置与 C++ 方言注册

```cmake
# CMake 配置：通过 mlir-tblgen 自动生成 .h.inc 和 .cpp.inc
set(LLVM_TARGET_DEFINITIONS ToyTypes.td)
mlir_tablegen(ToyTypes.h.inc -gen-type-def-decls)
mlir_tablegen(ToyTypes.cpp.inc -gen-type-def-defs)
add_public_tablegen_target(ToyTypesIncGen)
```

```cpp
// 在 C++ Dialect 初始化中批量注册生成的类型列表
#define GET_TYPEDEF_CLASSES
#include "ToyTypes.h.inc"

#define GET_TYPEDEF_DEFS
#include "ToyTypes.cpp.inc"

void ToyDialect::initialize() {
    addTypes<
#define GET_TYPEDEF_LIST
#include "ToyTypes.h.inc"
    >();
}
```

#### TableGen DSL 与生成的底层 C++ 映射对照表

| TableGen DSL 描述 | 自动生成的底层 C++ 代码层 |
| :--- | :--- |
| `def Toy_StructType` | 继承自 `mlir::Type::TypeBase` 的 `StructType` 导出类 |
| `parameters = (ins ...)` | 私有的 `detail::StructTypeStorage` 结构体、`KeyTy` 别名、`construct()` 分配逻辑与 `hashKey()` 计算 |
| `mnemonic` + `assemblyFormat` | 自动生成的 `StructType::parse()` 与 `StructType::print()` 解析/打印函数 |
| `(ins ...)` 中的字段名 | 自动生成的 C++ Getter 方法（如 `structType.getElementTypes()`）与静态 `get()` 工厂函数 |

