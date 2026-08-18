# Triton 结构化张量视图核心机制与架构推演

> 本指南深入剖析 Triton 编译器体系中 **`TensorView`（结构化张量视图）** 的设计哲学、TableGen ODS 建模、C++ 构造分层、短路验证机制、`TritonRaiseTensorView` 逆向仿射状态机推导及下游硬件 DMA 降级。

---

## 1. NPU 结构化张量视图演进

### 1.1 传统离散指针硬件失配

在经典 GPU 编程场景中，Triton Python 前端通过直观的张量化指针算术表达块级（Block-level）内存访问：

```python
@triton.jit
def vector_add_kernel(x_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    # 构造离散指针张量与边界布尔掩码
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    # 指针解引用与掩码加载
    x = tl.load(x_ptr + offsets, mask=mask)
```

上述代码经过 Python AST 降级后，在 Triton IR（TTIR）中生成经典的离散指针算术指令序列：

```mlir
// 初始 Pointer-style TTIR
%block_start = arith.muli %pid, %c256_i32 : i32
%starts = tt.splat %block_start : i32 -> tensor<256xi32>
%range = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
%offs = arith.addi %starts, %range : tensor<256xi32>

// 广播基地址指针并生成 256 个离散指针
%x_ptrs = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
%ptrs = tt.addptr %x_ptrs, %offs : tensor<256x!tt.ptr<f32>>, tensor<256xi32>

// 构造离散布尔掩码
%splat_n = tt.splat %n : i32 -> tensor<256xi32>
%mask = arith.cmpi slt, %offs, %splat_n : tensor<256xi32>

// 发起离散指针加载
%value = tt.load %ptrs, %mask, %zero : tensor<256x!tt.ptr<f32>>
```

在 GPU 体系下，这种以 `tensor<256x!tt.ptr<f32>>`（指针张量）和 `tensor<256xi1>`（布尔掩码）为载体的表达具有高度灵活性，能够自然映射到 GPU 的 SIMT 架构中——硬件上的每个线程各自持有一个独立指针，在掩码单元（Predicate Register）控制下发起内存事务。

**NPU 与专用张量加速器的硬件冲突**

```text
GPU (SIMT 模型)                            NPU / 专用张量加速器 (DMA / 向量模型)
┌───────────────────────────────────────┐  ┌───────────────────────────────────────────┐
│ Thread 0: Load ptr[0]  (if mask[0])   │  │ 2D DMA Engine:                           │
│ Thread 1: Load ptr[1]  (if mask[1])   │  │   - Base   = %x_ptr (单个物理基地址)      │
│ Thread 2: Load ptr[2]  (if mask[2])   │  │   - Shape  = [%n]   (硬件自动边界截断)    │
│ ...                                   │  │   - Stride = [1]    (跨步搬运步长)        │
│ Thread 255: Load ptr[255] (mask[255]) │  │   - Offset = [%pid * 256] (块起始坐标)   │
└───────────────────────────────────────┘  └───────────────────────────────────────────┘
 依赖：离散指针发散与掩码求值               依赖：高维连续/仿射结构描述符
```

- **硬件缺乏离散掩码执行单元**：专用 NPU 通常使用 **2D DMA 控制器** 或 **变长向量加载单元（如 RISC-V Vector / VLEN）** 执行高效的批量数据搬运。硬件在指令集层面要求输入结构化的**描述符参数**：物理基地址指针（`Base`）、父张量各维几何边界（`Shape`）、内存跨步（`Stride`）以及当前块在全局空间中的起始坐标（`Offset`）；
- **指针张量的信息丢失与逆向成本**：在传统的 `tt.addptr` 标量/向量算术数据流中，高层几何结构被彻底打散为底层的离散指针偏移序列，边界信息被编码进布尔张量 `%mask` 中。后端代码生成器若想调用硬件 DMA，必须通过极高成本的模式匹配尝试逆向还原几何结构。一旦匹配失败，编译器只能被迫退化为开销巨大的标量循环（Gather/Scatter 串行发射），导致硬件吞吐急剧恶化。

为了从根本上解决这一矛盾，Triton IR 引入了**结构化张量视图（`TensorView`）**。

---

### 1.2 寻址几何与读写解耦模型

`TensorView` 的核心设计哲学是将**寻址描述（Address Geometry）**与**实际内存读写（Memory Access Effect）**进行彻底解耦：

```text
                    ┌─────────────────────────────────────────────────────────┐
                    │               tt.make_tensor_view (纯视图)               │
                    │─────────────────────────────────────────────────────────│
                    │ - Base   : !tt.ptr<f32>   (物理首地址)                  │
                    │ - Shape  : [%n]           (父张量各维边界)              │
                    │ - Strides: [%stride]      (各维跨步)                    │
                    │ - Offsets: [%block_start] (当前 Tile 在父张量中的偏移)  │
                    │ - 契约   : [Pure] (无内存读写副作用，可自由外提)        │
                    └────────────────────────────┬────────────────────────────┘
                                                 │ 产生轻量句柄
                                                 │ Value : !tt.tensorview<tensor<256xf32>>
                                                 ▼
                    ┌─────────────────────────────────────────────────────────┐
                    │                 tt.load_view (物理读)                    │
                    │─────────────────────────────────────────────────────────│
                    │ - Operand      : %view                                  │
                    │ - Padding      : %zero (越界位置填充值)                 │
                    │ - Inherent Prop: boundaryCheck = array<i32: 0>          │
                    │ - 契约         : [MemRead<GlobalMemory>] (产生读副作用)  │
                    └─────────────────────────────────────────────────────────┘
```

传统指针模式与结构化视图模式在操作数组织、边界检查与优化能力上的全景对比如下：

| 维度 | 传统指针模式（`tt.load`） | 结构化视图模式（`tt.make_tensor_view` + `tt.load_view`） |
| :--- | :--- | :--- |
| **操作数组织** | 离散的指针张量 `tensor<256x!tt.ptr<T>>` + 布尔掩码 `tensor<256xi1>` | 单个物理指针 `!tt.ptr<T>` + 多维动态坐标数组（Shape, Strides, Offsets） |
| **几何与读写** | 寻址算术与内存读写耦合在单个 `tt.load` 算子中 | **解耦**：`make_tensor_view` 负责几何建模，`load_view` 负责读写 |
| **边界检查方式** | 显式计算每个元素的布尔值 `cmpi slt` | 算子固有属性 `boundaryCheck = array<i32: 0>` 声明需要检查的维度 |
| **硬件友好度** | 匹配 GPU SIMT 线程掩码执行 | 直接映射为 NPU 2D DMA 硬件描述符或向量长度寄存器 |
| **Pass 优化能力** | 指针计算与内存效果绑定，难以跨基本块移动 | `make_tensor_view` 具备 `Pure` 特性，可安全外提到循环外部（LICM） |

---

### 1.3 Generic 与 Pretty 语法视图

为了深入理解 `TensorView` 在 MLIR 内部的实际数据结构，我们需要对比其 **Generic Assembly（底层真实存储）** 与 **Pretty Syntax（声明式可读语法）**。

#### Generic Assembly 物理存储展平

在 MLIR 的通用底层对象模型中，所有的参数被平铺在单一的 Operands 列表与属性字典中：

```mlir
%stride1 = arith.constant 1 : i64

// 1. tt.make_tensor_view 的底层平铺展开
// Operands 顺序: base | shape[0] | strides[0] | offsets[0]
%view = "tt.make_tensor_view"(%x_ptr, %n, %stride1, %block_start)
  : (!tt.ptr<f32>, i32, i64, i32) -> !tt.tensorview<tensor<256xf32>>

// 2. tt.load_view 的底层展开
// <{...}> 为 Operation 固有属性存储（Properties），保存 boundaryCheck 与缓存策略
%value = "tt.load_view"(%view, %zero)
  <{boundaryCheck = array<i32: 0>,
    cache = 1 : i32,
    evict = 1 : i32,
    isVolatile = false}>
  : (!tt.tensorview<tensor<256xf32>>, tensor<256xf32>) -> tensor<256xf32>
```

#### Pretty Syntax 语义结构化

通过 ODS 中的 `assemblyFormat` 声明，文本呈现被重新组织为结构化的语义格式：

```mlir
%stride1 = arith.constant 1 : i64

// 结构化划分各维几何参数
%view = tt.make_tensor_view %x_ptr,
          [%n],            // Shape: 父张量各维大小
          [%stride1],      // Strides: 各维内存步长 (固定为 i64)
          [%block_start]   // Offsets: 当前 Block 起始全局坐标
          : <tensor<256xf32>>, // Result Type 中编码的静态 Block 类型
            [i32], [i32]       // 动态 Shape 与 Offsets 的实际整数类型

// 结构化表达读操作与越界保护
%value = tt.load_view %view, %zero
          {boundaryCheck = array<i32: 0>}
          : !tt.tensorview<tensor<256xf32>>
```

> [!NOTE]
> **方括号的分组实质**：Pretty Syntax 中的 `[%n]`、`[%stride1]` 和 `[%block_start]` 只是语法层面的显示分组。在底层 `Operation` 容器中，它们依然被组织为单一连续的 `SmallVector<Value>`，通过 ODS 生成的 Trait 算法还原区间边界。

---

### 1.4 MLIR 四维对象正交模型

在 `TensorView` 的设计中，不同维度的数据根据其生命周期、不可变性与所有权归属，被精确分配到了 MLIR 的四大对象子系统中：

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MLIR 对象子系统分工矩阵                                 │
├──────────────────────┬──────────────────────┬───────────────────────────────────┤
│ 数据维度             │ 承载载体             │ 典型字段与示例                    │
├──────────────────────┼──────────────────────┼───────────────────────────────────┤
│ 1. 动态数据流        │ SSA Operands         │ base, shape, strides, offsets     │
│ 2. 静态类型身份      │ Type Parameters      │ blockType (RankedTensor), order   │
│ 3. 算子固有状态      │ Inherent Properties  │ boundaryCheck, cache, evict       │
│ 4. 辅助分析提示      │ Discardable Attrs    │ tt.contiguity, tt.divisibility    │
└──────────────────────┴──────────────────────┴───────────────────────────────────┘
```

1. **SSA Operands（动态数据流）**：
   - 涵盖字段：`base`（基地址指针）、`shape`（动态父张量维度）、`strides`（动态步长）、`offsets`（动态块起点）、`padding`（可选填充值）；
   - 生命周期与存储：作为 `Operation` 的输入操作数，保存在连续的物理数组中，遵循严格的 SSA 数据流依赖关系与支配树（Dominance）约束。

2. **Type Parameters（静态类型单例）**：
   - 涵盖字段：`TensorViewType` 中的 `blockType`（如 `tensor<256xf32>`）与 `order`（维度排列，如 `[1, 0]`）；
   - 生命周期与存储：作为类型定义的不可变参数，进入 `MLIRContext` 内存池的 `TypeStorage` 中参与全局唯一化（Uniquing）。
   ```mlir
   !tt.tensorview<tensor<8x8xf16>>                 // 默认 order = [1, 0]
   !tt.tensorview<tensor<8x8xf16>, order = [0, 1]> // 另一个独立的 Uniqued 类型单例
   ```

3. **Inherent Properties（算子固有执行契约）**：
   - 涵盖字段：`load_view` 与 `store_view` 上的 `boundaryCheck`、`cache`（缓存修饰符）、`evict`（驱逐策略）、`isVolatile`；
   - 生命周期与存储：直接作为 C++ 结构体成员嵌入在 `Operation` 实例分配的 `Properties` 内存块中，随算子的创建、复制与析构而自动管理。

4. **Discardable Attributes（辅助分析优化提示）**：
   - 涵盖字段：`tt.contiguity`（连续性长度）、`tt.divisibility`（对齐整除性）、`tt.constancy`（常量性）；
   - 生命周期与存储：存放于通用的字典属性（`DictionaryAttr`）中。优化 Pass 即使在变换过程中丢弃这些属性，也不会破坏程序的功能正确性（仅使后续 Pass 采取保守假设）。

```mlir
// Generic IR 中属性与固有状态的边界定界：
%value = "tt.load_view"(%view)
  <{boundaryCheck = array<i32: 0>, cache = 1 : i32}> // Inherent Properties (固有状态)
  {tt.contiguity = 16 : i32}                          // Discardable Attribute (分析提示)
  : (!tt.tensorview<tensor<256xf32>>) -> tensor<256xf32>
```

> [!IMPORTANT]
> **设计契约定界**：
> 1. `shape`、`strides` 和 `offsets` 是**单次访问的动态几何参数**，因此必须作为 SSA Operands；
> 2. `blockType` 决定了加载产生的结果 Tensor 形状与元素类型，因此必须作为静态 **Type Parameter**；
> 3. `boundaryCheck` 描述了执行该次访存时硬件是否需要激活边界截断逻辑，属于该算子的**固有执行契约**，因此进入 `Properties`。

---

## 2. ODS 建模与底层契约推演

### 2.1 TensorViewType 编译期唯一化

在 `TritonTypes.td` 中，`TensorViewType` 作为 Triton Dialect 的核心类型定义如下：

```tablegen
def TT_TensorViewType : TritonTypeDef<"TensorView", "tensorview", []> {
  let summary = "Tensor view type in Triton IR type system";
  let description = [{
      A view over a tile (block) of a tensor residing in global memory, used to
      express block-based loads/stores.
  }];

  // 1. 核心类型参数：块静态类型与维度连续性排列
  let parameters = (ins
    "RankedTensorType":$blockType,
    ArrayRefParameter<"int32_t">:$order
  );

  // 2. 自定义打印/解析与校验声明
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;

  // 3. 上下文推导构建器
  let builders = [
    // 默认行主序构建器（由 blockType 推导 Context 与默认 order）
    TypeBuilderWithInferredContext<(ins "RankedTensorType":$blockType), [{
      return $_get(blockType.getContext(), blockType,
                   ::mlir::triton::TensorViewType::getDefaultOrder(blockType.getRank()));
    }]>,
    // 显式指定 order 构建器
    TypeBuilderWithInferredContext<(ins "RankedTensorType":$blockType,
                                        "ArrayRef<int32_t>":$order), [{
      return $_get(blockType.getContext(), blockType, order);
    }]>
  ];
}
```

#### order 维度排列的物理意义与内存深拷贝

- **`order` 语义契约**：按照**物理内存地址变化速度从快到慢**排列各维度索引（Fastest-changing dimension first）。例如对于 2D 行主序矩阵（Row-major，第 1 维列连续、步长为 1），其地址变化最快的维度是 Dim 1，因此 `order = [1, 0]`；对于 3D 行主序张量，其 `order = [2, 1, 0]`；
- **默认排序算法**：`getDefaultOrder(rank)` 固定生成递减序列 `[rank-1, ..., 0]`；
- **`ArrayRefParameter<"int32_t">` 的内存协议**：`order` 是一个动态长度的整数数组。在类型唯一化（Type Uniquing）构造阶段，MLIR 自动生成的 `TensorViewTypeStorage::construct` 会调用 `allocator.copyInto(order)` 将调用者传入的临时栈数组完整**深拷贝到 `MLIRContext` 托管的 BumpPtrAllocator 内存池中**，确保类型句柄在跨 Pass 传递时内部指针始终有效。

#### verify 对 order 的排列约束

`genVerifyDecl = 1` 使得构建器在将参数写入全局哈希表前，强制触发 C++ 端手写的校验逻辑（`TensorViewType::verify`）：

```cpp
LogicalResult TensorViewType::verify(
    function_ref<InFlightDiagnostic()> emitError,
    RankedTensorType blockType, ArrayRef<int32_t> order) {
  int64_t rank = blockType.getRank();
  
  // 1. 维度长度一致性校验
  if (static_cast<int64_t>(order.size()) != rank)
    return emitError() << "tensorview order size (" << order.size()
                       << ") does not match block rank (" << rank << ")";

  // 2. [0, rank) 全排列合法性校验（禁止越界与重复）
  SmallVector<bool> seen(rank, false);
  for (int32_t dim : order) {
    if (dim < 0 || dim >= rank || seen[dim]) {
      return emitError() << "tensorview order must be a permutation of "
                            "[0, rank), got an invalid or repeated entry: "
                         << dim;
    }
    seen[dim] = true;
  }
  return success();
}
```

---

### 2.2 MakeTensorViewOp 核心 Traits 深度剖析

```text
                        MakeTensorViewOp 核心 Trait 架构矩阵
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 1. Pure (AlwaysSpeculatable + NoMemoryEffect)                                   │
│    └─► 赋予优化器自由进行循环不变代码外提（Loop Hoist）与死代码删除（DCE）的能力    │
├─────────────────────────────────────────────────────────────────────────────────┤
│ 2. SameVariadicOperandSize (等长公式分段)                                       │
│    └─► 将底层的 flat operands 按照 (N-1)/3 精确还原为 shape, strides, offsets   │
├─────────────────────────────────────────────────────────────────────────────────┤
│ 3. TypesMatchWith (双向类型约束与推导)                                           │
│    ├─► 校验期：assert(base.type == ptr<result.elementType, AS>)                 │
│    └─► 解析期：自动从 result type 动态反推未显式打印的 base operand 类型        │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### Pure Trait 与 Loop Hoist 优化保证

`Pure` 是 `AlwaysSpeculatable`（永远允许投机执行）与 `NoMemoryEffect`（零内存副作用）的组合：

- **投机执行保证（Speculatability）**：`make_tensor_view` 仅在 SSA 寄存器层面组合几何参数，不解引用任何物理指针。即使位于从未执行的分支路径或循环内部，只要输入 SSA 可用，提前计算该 View 绝不会引发硬件异常；
- **循环外提收益（LICM）**：当循环内的 `shape`, `strides`, `offsets` 不随迭代变化时，编译器可安全将 `make_tensor_view` 提升至循环前置块（Preheader），避免在每次循环迭代中重复构造 View 描述符。

#### SameVariadicOperandSize 扁平操作数等长分段算法

底层 `mlir::Operation` 的操作数物理结构是一维扁平数组 `SmallVector<Value>`。`MakeTensorViewOp` 包含 1 个固定操作数（`base`）和 3 个变长操作数（`shape`, `strides`, `offsets`）。为了在不借助额外属性的前提下精确定位各字段在扁平数组中的起始位置与长度，ODS 采用了等长公式算法：

```text
静态字段：  base   │ shape          │ strides        │ offsets
物理分布：  0      │ 1 ... R        │ R+1 ... 2R     │ 2R+1 ... 3R
           └──────┴────────────────┴────────────────┴─────────────┘
           固定 1   │ 变长 R         │ 变长 R         │ 变长 R
```

由 `OpDefinitionsGen` 自动生成的索引分段计算逻辑如下：

```cpp
std::pair<unsigned, unsigned>
MakeTensorViewOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true, true, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    prevVariadicCount += isVariadic[i];

  // 1 个非 variadic 操作数，3 个等长 variadic 字段
  int variadicSize = (getOperation()->getNumOperands() - 1) / 3;

  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

ValueRange getShape()   { return getODSOperands(1); } // 对应区间 [1, 1+R)
ValueRange getStrides() { return getODSOperands(2); } // 对应区间 [1+R, 1+2R)
ValueRange getOffsets() { return getODSOperands(3); } // 对应区间 [1+2R, 1+3R)
```

**多组变长操作数分段机制对比**：

| 机制对比 | `AttrSizedOperandSegments` | `SameVariadicOperandSize`（TensorView 选型） |
| :--- | :--- | :--- |
| **元数据存储** | 需在算子上附加 `operandSegmentSizes` 属性字典 | **零属性开销**（纯算术推导，无任何额外内存负载） |
| **结构前提** | 允许各 variadic 操作数具有任意不同长度 | **强数学约束**：各 variadic 字段长度严格相等（均为张量 Rank $R$） |
| **操作数校验** | 依赖属性数组元素与操作数总数比对 | 严格约束操作数总数满足 $\text{NumOperands} = 1 + 3R$ |

> [!WARNING]
> **操作数总数硬契约**：该机制要求操作数总数必须严格满足公式 $\text{NumOperands} = 1 + 3R$。如果手动构造 IR 时传入了长度不一致的 `shape` 与 `strides`，整除截断会导致 getter 提取到错误的内存切片。

#### TypesMatchWith 双向类型约束与 Parser 推导

`TypesMatchWith` 建立起了 `result`（`TensorViewType`）与 `base`（`PointerType`）之间的强类型约束：

```tablegen
TypesMatchWith<"infer the base pointer type from the result tensorview type",
               "result", "base",
               "getPointerType(::mlir::cast<::mlir::triton::TensorViewType>($_self).getBlockType().getElementType(), getAddressSpace($_self))">
```

该 Trait 在编译器生成代码中承担双重使命：
1. **自动生成不变式校验（Invariant Verification）**：
   ```cpp
   Type expectedBaseType = getPointerType(
       cast<TensorViewType>(getResult().getType()).getBlockType().getElementType(),
       getAddressSpace(getResult().getType()));
       
   if (expectedBaseType != getBase().getType())
     return emitOpError("failed to verify that base pointer matches result element type");
   ```
2. **支持声明式语法解析器（Declarative Assembly Parser）**：在 Pretty Syntax 中，冒号后仅打印了结果类型 `type($result)`，省略了 `type($base)`。Parser 借助 `TypesMatchWith` 保存的变换表达式，在解析阶段直接推导出 `base` 的预期指针类型，完成 `parser.resolveOperands(baseOperands, expectedBaseType, ...)`。

---

### 2.3 固有属性与 LLVM 描述符 ABI 布局

#### LLVM 降级描述符结构体内存拓扑

当 `tt.tensorview` 经过方言转换（Dialect Conversion）最终降级到 LLVM IR 时，抽象的编译期视图类型会被物化为一个**C 兼容的扁平描述符结构体（TensorView Descriptor Struct）**：

```text
               TensorView Descriptor 64 位物理内存排布 (Rank = 2, 总计 56 字节)
 字节偏移 (Offset)
 0x00 ┌────────────────────────────────────────────────────────────┐
      │  void *base (8 字节物理内存首地址裸指针)                    │  8 Bytes
 0x08 ├────────────────────────────┬───────────────────────────────┤
      │  int64_t shape[0] (8 Bytes)│  int64_t shape[1] (8 Bytes)   │  16 Bytes
 0x18 ├────────────────────────────┼───────────────────────────────┤
      │  int64_t stride[0] (8B)    │  int64_t stride[1] (8 Bytes)  │  16 Bytes
 0x28 ├────────────────────────────┼───────────────────────────────┤
      │  int64_t offset[0] (8B)    │  int64_t offset[1] (8 Bytes)  │  16 Bytes
 0x38 └────────────────────────────────────────────────────────────┘ (sizeof == 56 Bytes)
```

该结构体在 C++ ABI 层面由寄存器组直接承载传递，完全消除了任何动态堆内存分配。

与纯视图构建不同，`LoadViewOp` 和 `StoreViewOp` 是真正产生内存交互的算子：

```tablegen
def TT_LoadViewOp : TT_Op<"load_view", [
  TypesMatchWith<"result matches the view block type", "src", "result",
                 "::mlir::cast<::mlir::triton::TensorViewType>($_self).getBlockType()">,
  TypesMatchWith<"padding matches the view block type", "src", "padding",
                 "::mlir::cast<::mlir::triton::TensorViewType>($_self).getBlockType()",
                 "($_op.getOperands().size() <= 1) || std::equal_to<>()">
]> {
    let arguments = (
      ins
      Arg<TT_TensorViewType, "", [MemRead<GlobalMemory>]>:$src,
      Optional<TT_Type>:$padding,
      DefaultValuedAttr<DenseI32ArrayAttr, "::llvm::ArrayRef<int32_t>{}">:$boundaryCheck,
      DefaultValuedAttr<TT_CacheModifierAttr, "::mlir::triton::CacheModifier::NONE">:$cache,
      DefaultValuedAttr<TT_EvictionPolicyAttr, "::mlir::triton::EvictionPolicy::NORMAL">:$evict,
      DefaultValuedAttr<BoolAttr, "false">:$isVolatile
    );
    let results = (outs TT_Type:$result);
}
```

#### Inherent Properties 结构体展开

`DefaultValuedAttr` 声明的字段会自动汇入算子的 `Properties` 存储中：

```cpp
// LoadViewOp::Properties 内部结构
struct Properties {
  DenseI32ArrayAttr boundaryCheck; // 声明需要边界检查的维度列表 (如 [0])
  CacheModifierAttr cache;         // 缓存策略 (NONE, CA, CG, CS...)
  EvictionPolicyAttr evict;        // 驱逐优先级 (NORMAL, EVICT_FIRST, EVICT_LAST)
  BoolAttr isVolatile;             // 是否为 Volatile 访问
};
```

#### 细粒度内存读写副作用建模

通过 `Arg<TT_TensorViewType, "", [MemRead<GlobalMemory>]>:$src` 标注，ODS 自动为算子实现了 `MemoryEffectOpInterface`：

```cpp
void LoadViewOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  // 精确声明：对输入操作数 0 (src view) 关联的全局内存发起 Read 操作
  effects.emplace_back(MemoryEffects::Read::get(),
                       &getOperation()->getOpOperand(0),
                       /*stage=*/0, /*symbol=*/false,
                       GlobalMemory::get());
}
```

- `MemRead<GlobalMemory>` 阻止了包含读写依赖的非法指令重排；
- 与 `MakeTensorViewOp` 的 `Pure` 形成鲜明对比，确立了“**几何构造可任意调度，内存读写严格受控**”的编译器优化边界。

---

## 3. C++ 构造分层与短路验证

### 3.1 四层 Builder 归一化链

在编写编译器 Pass 或前端 Lowering 时，开发者通常持有不同层级的信息（例如有时持有静态张量形状，有时已构建好结果类型）。`MakeTensorViewOp` 在 C++ 端实现了一套清晰的**四层 Builder 归一化流水线（Normalization Pipeline）**：

```text
                    MakeTensorViewOp 四层 Builder 归一化流水线
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Convenience A (默认 order)                                               │
│    build(builder, state, base, shape, strides, offsets, tensorShape)        │
│    └─► 补全默认 order = getDefaultOrder(tensorShape.size())，转发至 B        │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. Convenience B (显式 order)                                               │
│    build(builder, state, base, shape, strides, offsets, tensorShape, order) │
│    ├─► 1. 从 base 的 PointerType 获取 pointeeType                           │
│    ├─► 2. RankedTensorType::get(tensorShape, pointeeType) -> blockType      │
│    ├─► 3. TensorViewType::get(blockType, order) -> resultType               │
│    └─► 4. 将推导出的 resultType 转发至 Normalized C                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. Normalized C (ODS 强类型生成)                                            │
│    build(builder, state, resultType, base, shape, strides, offsets)        │
│    ├─► state.addOperands(base, shape, strides, offsets)                     │
│    └─► state.addTypes(resultType)                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ 4. Generic D (ODS 扁平物理写入)                                             │
│    build(builder, state, resultTypes, flatOperands, attributes)             │
│    └─► 直接将平铺的物理数组与属性写入 OperationState                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

四层 Builder 的具体实现与参数转发逻辑如下：

**第 1 层：Convenience A（单形状入口，补齐默认 Order）**

```cpp
void MakeTensorViewOp::build(
    OpBuilder &builder, OperationState &state,
    Value base, ValueRange shape, ValueRange strides,
    ValueRange offsets, ArrayRef<int32_t> tensorShape) {
  // 补全默认行主序 order，转发至第 2 层
  build(builder, state, base, shape, strides, offsets, tensorShape,
        TensorViewType::getDefaultOrder(tensorShape.size()));
}
```

**第 2 层：Convenience B（显式 Order 入口，推导结果类型）**

```cpp
void MakeTensorViewOp::build(
    OpBuilder &builder, OperationState &state,
    Value base, ValueRange shape, ValueRange strides,
    ValueRange offsets, ArrayRef<int32_t> tensorShape,
    ArrayRef<int32_t> order) {
  // 1. 从基地址指针提取元素类型
  auto pointerType = cast<PointerType>(base.getType());
  
  // 2. 构造块张量类型与视图类型
  auto blockType = RankedTensorType::get(
      SmallVector<int64_t>(tensorShape.begin(), tensorShape.end()),
      pointerType.getPointeeType());
  auto resultType = TensorViewType::get(blockType, order);

  // 3. 转发至第 3 层
  build(builder, state, resultType, base, shape, strides, offsets);
}
```

**第 3 层：Normalized C（强类型标准入口，写入 State）**

```cpp
// 由 ODS 自动生成：接收已确定的 resultType 与结构化命名参数
void MakeTensorViewOp::build(OpBuilder &, OperationState &state,
                             Type result, Value base,
                             ValueRange shape, ValueRange strides,
                             ValueRange offsets) {
  state.addOperands(base);
  state.addOperands(shape);
  state.addOperands(strides);
  state.addOperands(offsets);
  state.addTypes(result);
}
```

**第 4 层：Generic D（底层扁平入口）**

```cpp
// 接收完全展平的 flatOperands 与 resultTypes，供通用反序列化或 Bytecode 使用
void MakeTensorViewOp::build(
    OpBuilder &, OperationState &state, TypeRange resultTypes,
    ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() >= 1);
  state.addOperands(operands);
  state.addAttributes(attributes);
  state.addTypes(resultTypes);
}
```

---

### 3.2 OperationState 生命周期演变

在 C++ 层面构造算子时，`create` 静态工厂与 `build` 填充函数的分工与协作生命周期如下：

```text
调用 MakeTensorViewOp::create(builder, loc, base, shape, strides, offsets, tensorShape, order)
  │
  ├─► 1. 初始化中间状态：OperationState state(loc, "tt.make_tensor_view")
  │
  ├─► 2. 调度归一化链：build(builder, state, ...)  [逐层推导类型并填充 state]
  │
  ├─► 3. 底层实体分配：Operation *operation = builder.create(state)
  │       ├─ 在当前 insertion point 插入新 Operation 节点
  │       └─ 建立 SSA 链表引用关系
  │
  └─► 4. 包装视图句柄：return cast<MakeTensorViewOp>(operation)
```

| 阶段 / 对象 | 职责定位 | 可变性与内存 |
| :--- | :--- | :--- |
| **`OperationState`** | 算子实例化前的临时暂存容器（栈对象） | 可变（Mutable），随 `build` 填充逐步累加参数 |
| **`build(...)`** | 纯状态填充逻辑 | 不分配 `Operation` 内存，仅向 `OperationState` 追加操作数与类型 |
| **`builder.create(state)`** | 物理实体创建与 IR 插入 | 在堆/Arena 上分配 `Operation` 物理节点并插入当前基本块游标位置 |
| **`create(...)` 静态工厂** | 面向调用者的便捷门面（Facade） | 自动整合“状态初始化 $\rightarrow$ 调度 `build` $\rightarrow$ 实体分配 $\rightarrow$ 句柄转换” |

---

### 3.3 双重验证防线与短路验证机制

```text
                    MakeTensorViewOp::verifyInvariants()
                                     │
                                     ▼
        ┌────────────────────────────────────────────────────────────┐
        │ Step 1: verifyInvariantsImpl()  [ODS 自动生成的底层校验]     │
        │ - 检查 base/shape/strides/offsets 的类型约束                 │
        │ - 检查 result 的 TT_TensorViewType 约束                     │
        │ - 执行 TypesMatchWith 自动生成的 base/result 谓词校验        │
        └────────────────────────────┬───────────────────────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 │ 失败 (LogicalResult::failure)          │ 成功 (LogicalResult::success)
                 ▼                                       ▼
          [立即中断并报错]              ┌──────────────────────────────────────────────────┐
                                       │ Step 2: verify()  [C++ 手写的高阶业务语义校验]     │
                                       │ - 检查 getShape().size() == blockType.getRank()  │
                                       └──────────────────────────────────────────────────┘
```

```cpp
LogicalResult MakeTensorViewOp::verifyInvariants() {
  // 核心：短路逻辑运算符 &&
  if (succeeded(verifyInvariantsImpl()) && succeeded(verify()))
    return success();
  return failure();
}
```

采用 `&&` 顺序进行短路执行的核心原因包括：
1. **类型安全前提**：`verifyInvariantsImpl()` 首先确认了 `getResult().getType()` 100% 满足 `TT_TensorViewType` 约束，以及 `base` 与 `result` 元素类型的一致性；
2. **手写 Verifier 免除防御性判断**：在第 2 步的 `verify()` 实现中，开发者可以直接安全地进行 `cast<TensorViewType>` 强转，而无需使用耗时且冗余的 `dyn_cast` 判空：

```cpp
LogicalResult MakeTensorViewOp::verify() {
  // 前置不变式已通过，直接安全解引用 blockType
  int64_t rank = cast<TensorViewType>(getResult().getType())
                     .getBlockType().getRank();
                     
  // 检查变长操作数分段还原后的长度是否与张量 Rank 匹配
  if (static_cast<int64_t>(getShape().size()) != rank)
    return emitOpError()
           << "expected " << rank
           << " shape/strides/offsets operands to match result rank, got "
           << getShape().size();
           
  return success();
}
```

---

## 4. TritonRaiseTensorView 逆向提升

### 4.1 Pass 架构定位与硬契约管线

`TritonRaiseTensorView`（对应源码 `lib/Dialect/Triton/Transforms/RaiseTensorView.cpp`）是连接前端经典指针算术与后端结构化张量优化的核心编译桥梁：

```text
              TritonRaiseTensorView 在编译流水线中的定位
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. 前端 Python JIT 生成初始 Pointer-style TTIR                              │
│    (tt.make_range, tt.splat, tt.addi, tt.addptr, tt.load)                   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ 经历标准 Canonicalize 与 Inlining
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. TritonRaiseTensorView (本 Pass)                                          │
│    ├─► visitPtr 逆向追踪指针算术 ──► 提纯基地址与仿射多维步长 (PtrState)       │
│    ├─► parseMask 逆向追踪比较逻辑 ──► 提纯父张量边界与 BoundaryCheck 掩码    │
│    └─► 重构为结构化算子 ──► tt.make_tensor_view + tt.load_view / store_view │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ 下游承接硬件物化
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. TritonNPU / 专用加速器代码生成 (如 MaterializeChunkLoops / DMA Lowering) │
└─────────────────────────────────────────────────────────────────────────────┘
```

**硬契约与保守原则（No Legacy Fallback）**：
- **完全提阶或显式报错**：针对所有可以清晰解码为仿射多维块的访问，100% 提升为 `TensorView`；
- **拒绝静默残留**：若代码中存在不可解析的非结构化访存（如数据依赖的间接寻址 Gather/Scatter、取模回绕等），Pass 会抛出硬性编译错误，绝不静默放过未被提升的 legacy 指针指令，防止在后端代码生成阶段产生隐蔽的性能陷阱。

---

### 4.2 PtrState 仿射状态逆向解码

Pass 在遍历指针算术时，通过内部结构体 `PtrState` 维护当前 SSA 值的仿射几何描述：

$$\text{elem}[i_0, i_1, \dots] = \text{base} + \sum_{d} \left( \text{offsets}[d] + i_d \times \text{strides}[d] \right)$$

```cpp
struct PtrState {
  Value base;                  // 物理内存基地址指针 (!tt.ptr<T>)
  Type indexType;              // 索引计算的整数类型 (i32 / i64 / index)
  SmallVector<Value> offsets;  // 每维当前 Block 的起始偏移 SSA 变量
  SmallVector<Value> strides;  // 每维内存地址跨步 SSA 变量
  SmallVector<int64_t> sizes;  // 每维静态 Tile 大小 (如 [16, 32])
};
```

#### visitPtr 递归状态转移表

| 匹配到的算子（Defining Op） | 仿射状态转移逻辑（State Transition） | 物理几何意义 |
| :--- | :--- | :--- |
| **`tt.make_range {start, end}`** | $\text{offsets}[0] = \text{start}$, $\text{strides}[0] = 1$, $\text{sizes}[0] = \text{end} - \text{start}$ | 引入 1 维单位连续步长（Unit-stride）的基础序列 |
| **`tt.splat %val`** | $\text{offsets}[d] = \%\text{val}$, $\text{strides}[d] = 0$, $\text{sizes}[d] = \text{dimSize}$ | 引入广播标量，所有维度的步长初始化为 0（Uniform） |
| **`arith.addi %lhs, %rhs`** | $\text{offsets}[d] = \text{lhs.offsets}[d] + \text{rhs.offsets}[d]$, $\text{strides}[d] = \text{lhs.strides}[d] + \text{rhs.strides}[d]$ | 叠加两个仿射表达式（如块基地址 + 局部偏移） |
| **`arith.muli %lhs, %cst`** | $\text{offsets}[d] = \text{lhs.offsets}[d] \times \%\text{cst}$, $\text{strides}[d] = \text{lhs.strides}[d] \times \%\text{cst}$ | 线性缩放步长（如多维行主序展开中的乘外层维度 Stride） |
| **`tt.expand_dims %src {axis}`** | 在 $\text{offsets}, \text{strides}, \text{sizes}$ 的 `axis` 位置插入 1 维单例 $\text{size}=1, \text{stride}=0$ | 增加未广播的虚拟维度 |
| **`tt.broadcast %src`** | 将 $\text{sizes}[d] = 1$ 的维度扩展为目标形状，保持 $\text{strides}[d] = 0$ | 多维张量广播（如行向量广播为 2D 矩阵） |
| **`tt.addptr %base, %offset`** | 设置 $\text{state.base} = \%\text{base}$，将 `%offset` 的解码结果吸收至 `state` | 捕获物理指针基地址 |

#### 一维逆向回溯推导示例

```text
                                  visitPtr 逆向回溯推导示例 (一维)
  %ptrs = tt.addptr %base, %offs
              │
              ├─► 捕获 state.base = %x_ptr
              ▼
  %offs = arith.addi %starts, %range
              │
              ├─► %starts (来自 splat(%pid * 256)) ──► state.offsets[0] = %block_start
              │                                        state.strides[0] = 0
              ▼
  %range = tt.make_range {start=0, end=256}
              └──────────────────────────────────────► state.offsets[0] += 0
                                                       state.strides[0] += 1
                                                       state.sizes[0]   = 256
```

#### 二维仿射状态代数推演全景跟踪

在 2D Strided 访存（如 Tile $16 \times 32$）中，前端会分别构造行索引与列索引，并通过 `expand_dims` + `broadcast` 提升至 2D 网格，最后通过 `addi` 叠加。`visitPtr` 逆向推导的代数向量演化过程如下：

```text
1. 行路径 (Row Path: 16):
   tt.make_range(0, 16)               ──► offsets=[0],      strides=[1],           sizes=[16]
   tt.expand_dims(axis=1)             ──► offsets=[0, 0],   strides=[1, 0],        sizes=[16, 1]
   tt.broadcast(16x32)                ──► offsets=[0, 0],   strides=[1, 0],        sizes=[16, 32]
   arith.muli(%stride_h)              ──► offsets=[0, 0],   strides=[%stride_h, 0],sizes=[16, 32]

2. 列路径 (Col Path: 32):
   tt.make_range(0, 32)               ──► offsets=[0],      strides=[1],           sizes=[32]
   tt.expand_dims(axis=0)             ──► offsets=[0, 0],   strides=[0, 1],        sizes=[1, 32]
   tt.broadcast(16x32)                ──► offsets=[0, 0],   strides=[0, 1],        sizes=[16, 32]

3. 汇合叠加 (arith.addi %row, %col):
   offsets = [0 + 0, 0 + 0]           ──► offsets = [0, 0]
   strides = [%stride_h + 0, 0 + 1]   ──► strides = [%stride_h, 1]
   sizes   = [16, 32]

4. 块基地址合并 (arith.addi %block_offset, %local_offset):
   offsets = [%block_off_h, %block_off_w]
   strides = [%stride_h, 1]

5. 物理指针结合 (tt.addptr %base, %offs):
   base    = %ptr
```

---

### 4.3 边界掩码仿射解析与组装

在传统 TTIR 中，边界控制是通过 `arith.cmpi slt, %offs, %limit` 生成的布尔张量表达的。Pass 必须将该数据流反解为两项结构化信息：
1. **父张量的实际维度边界（Parent Shape）**；
2. **需要激活硬件边界保护的维度索引（BoundaryCheck Dims）**。

```cpp
// parseMask 逆向推导伪代码
LogicalResult parseMask(Value mask, SmallVectorImpl<Value> &dimBound) {
  // 1. 匹配 cmpi slt 或 sle 算子
  auto cmpi = mask.getDefiningOp<arith::CmpIOp>();
  if (!cmpi || cmpi.getPredicate() != arith::CmpIPredicate::slt)
    return failure();

  Value lhs = cmpi.getLhs(); // 应当为 offs 数据流
  Value rhs = cmpi.getRhs(); // 应当为 splat(limit)

  // 2. 提取 limit 标量作为该维度的父张量边界
  auto splat = rhs.getDefiningOp<triton::SplatOp>();
  dimBound[dim] = splat.getSrc(); // 提取出原始标量 %n
  return success();
}
```

在收集完 `PtrState` 与 `dimBound` 后，Pass 在汇合点执行最终组装：

```cpp
SmallVector<Value> shapeVals, strideVals, offsetVals;
SmallVector<int32_t> boundaryCheck;

for (int64_t d = 0; d < rank; ++d) {
  strideVals.push_back(state.strides[d]);
  offsetVals.push_back(state.offsets[d]);

  if (dimBound[d]) {
    shapeVals.push_back(dimBound[d]); // 填入 mask 中解码出的父张量大小
    boundaryCheck.push_back(d);       // 声明该维需要边界检查
  } else {
    // 若该维无 mask 限制，使用静态块大小作为占位 Shape，且不加入 boundaryCheck
    shapeVals.push_back(constantI32(state.sizes[d]));
  }
}
```

---

### 4.4 端到端 IR 提升时空对照

以一个二维 Strided 矩阵块读取（Tile $16 \times 32$，带二维边界保护）为例，对比提升前后的 IR 结构：

#### 提升前 Pointer-style TTIR

```mlir
// 初始 Pointer-style TTIR
%base_ptr = tt.splat %ptr : !tt.ptr<f32> -> tensor<16x32x!tt.ptr<f32>>

// 行索引生成与展开
%r_h = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
%r_h_2d = tt.expand_dims %r_h {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
%off_h = tt.broadcast %r_h_2d : tensor<16x1xi32> -> tensor<16x32xi32>
%stride_h_splat = tt.splat %stride_h : i32 -> tensor<16x32xi32>
%idx_h = arith.muli %off_h, %stride_h_splat : tensor<16x32xi32>

// 列索引生成与展开
%r_w = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
%r_w_2d = tt.expand_dims %r_w {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
%off_w = tt.broadcast %r_w_2d : tensor<1x32xi32> -> tensor<16x32xi32>

// 扁平地址累加与掩码
%offs_2d = arith.addi %idx_h, %off_w : tensor<16x32xi32>
%ptrs = tt.addptr %base_ptr, %offs_2d : tensor<16x32x!tt.ptr<f32>>, tensor<16x32xi32>
%mask_h = arith.cmpi slt, %off_h, %splat_H : tensor<16x32xi32>
%mask_w = arith.cmpi slt, %off_w, %splat_W : tensor<16x32xi32>
%mask = arith.andi %mask_h, %mask_w : tensor<16x32xi1>

// 物理加载
%val = tt.load %ptrs, %mask, %zero : tensor<16x32x!tt.ptr<f32>>
```

#### 提升后 结构化 TensorView IR

```mlir
// 经过 TritonRaiseTensorView 提升后的精炼 IR
%view = tt.make_tensor_view %ptr,
          [%H, %W],                      // 恢复的父张量完整几何 Shape
          [%stride_h, %c1_i64],          // 恢复的各维物理跨步
          [%block_off_h, %block_off_w]   // 恢复的当前 Block 块起点
          : <tensor<16x32xf32>, order = [1, 0]>,
            [i32, i32], [i32, i32]

%val = tt.load_view %view, %zero
          {boundaryCheck = array<i32: 0, 1>}
          : !tt.tensorview<tensor<16x32xf32>, order = [1, 0]>
```

#### 提升前后算子与操作数映射对照表

| 提升前 Pointer-style 算子序列 | 逆向提取的几何语义 | 映射至 `tt.make_tensor_view` / `load_view` 目标字段 |
| :--- | :--- | :--- |
| `%base_ptr = tt.splat %ptr` | 物理内存首地址 | `base = %ptr` |
| `%r_h` + `%r_h_2d` + `%stride_h_splat` + `%idx_h` | 行向连续索引与物理步长 | `shape[0] = %H`, `strides[0] = %stride_h`, `offsets[0] = %block_off_h` |
| `%r_w` + `%r_w_2d` + `%off_w` | 列向连续索引（单位步长） | `shape[1] = %W`, `strides[1] = 1`, `offsets[1] = %block_off_w` |
| `%mask_h` 与 `%mask_w`（`cmpi slt`） | 声明两维均存在父张量越界保护需求 | `boundaryCheck = array<i32: 0, 1>` |
| `%val = tt.load %ptrs, %mask, %zero` | 实际物理读操作与越界填充值 | `tt.load_view %view, %zero` |

---

## 5. 硬件降级物化与 DMA 驱动

### 5.1 结构化视图块循环物化

在实际 NPU 硬件上，片上向量寄存器或 SRAM 缓冲区的物理容量通常受限：前端定义的算法分块可能为 $128 \times 256$，但底层向量执行单元的单指令向量长度可能仅为 64（Chunk Size = 64）。结构化 `TensorView` 在进入底层代码生成前，经历**物理分块循环物化（Chunk Loop Materialization）**：

#### 读取视图循环物化

```text
tt.load_view %view (128x256)
  │
  ├─► Pass: triton-materialize-chunk-loops
  ▼
scf.for %i = 0 to 128 step 1 {          // 外层维度循环
  scf.for %j = 0 to 256 step 64 {       // 沿 fastest 维度按 Chunk 步长循环
    // 动态更新 View 的 offsets 局部切片
    %sub_view = tt.make_tensor_view %base, [%H, %W], [%stride, 1], [%off_i, %j]
    // 发射单条硬件向量加载
    %chunk_val = tt.load_view %sub_view : !tt.tensorview<tensor<64xf32>>
    // ... 向量计算与寄存器累加 ...
  }
}
```

#### 写入视图循环物化

写操作（`tt.store_view`）具有严格的对称物化逻辑：在嵌套循环内部，待写入的完整张量 `%computed_tensor` 同样通过 `tensor.extract_slice` 切出当前 Chunk 的数据切片，并与对应偏移的 `%sub_view` 结合发射写入：

```mlir
scf.for %i = 0 to 128 step 1 {
  scf.for %j = 0 to 256 step 64 {
    // 1. 提取当前物理 Chunk 的写入数据切片
    %val_slice = tensor.extract_slice %computed_tensor[%i, %j] [1, 64] [1, 1]
                   : tensor<128x256xf32> to tensor<64xf32>
    // 2. 构造局部 View
    %sub_view = tt.make_tensor_view %dst_ptr, [%H, %W], [%stride, 1], [%i, %j]
    // 3. 执行单条硬件向量存储
    tt.store_view %sub_view, %val_slice {boundaryCheck = array<i32: 0, 1>}
  }
}
```

#### 循环物化的核心优势

相比于直接对离散指针做循环展开，基于 `TensorView` 进行循环物化具有不可替代的架构优势：
1. **边界条件解析简化**：尾部不完整 Chunk 的边界处理无需在每条指令上重新计算 `cmpi` 掩码，而是通过简单的算术截断（`affine.min` 计算剩余元素数量）直接更新硬件向量长度寄存器（`vsetvl`）；
2. **多维地址自动线性化**：利用 `order` 信息，Pass 总是优先沿着地址最连续的物理维度（Fastest-changing axis）进行步长迭代，保证向量加载始终命中连续内存行。

---

### 5.2 2D DMA 硬件描述符降级

在将方言最终转换至 LLVM 方言（`TritonReexenNPUToLLVM`）阶段，`tt.load_view` 和 `tt.store_view` 会直接映射为**硬件 DMA 传输调用**或**底层内联汇编指令**。

对于二维的 `load_view` 操作，Lowering Pass 在栈上分配并填充标准的硬件 DMA 描述符（DMA Descriptor）：

```cpp
// LoadStoreOpToLLVM 中的 DMA 描述符填充逻辑
struct Dma2DDescriptor {
  void *src_addr;            // 源地址：base + sum(offsets[d] * strides[d])
  void *dst_addr;            // 目的地址：片上 SRAM / 向量寄存器堆首地址
  uint32_t width_bytes;      // 连续传输宽度：sizes[1] * sizeof(elem)
  uint32_t height_lines;     // 传输行数：sizes[0]
  uint32_t src_stride_bytes; // 源内存行跨步：strides[0] * sizeof(elem)
  uint32_t dst_stride_bytes; // 目的内存行跨步：width_bytes (连续紧凑排列)
};
```

```mlir
// 降级后的 LLVM Dialect 代码形态
%desc = llvm.alloca %c1 x !llvm.struct<"dma_desc", (ptr, ptr, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// 依次将 base, offsets, strides, sizes 写入描述符字段...
llvm.call @reexen_dma_submit_2d(%desc) : (!llvm.ptr) -> ()
llvm.call @reexen_dma_wait(%desc) : (!llvm.ptr) -> ()
```

---

### 5.3 结构化访存编译流向与设计总结

回顾整个 Triton `TensorView` 的生命周期，其编译流向构成了一条“**由散到整、再由整到硬**”的清晰推导链路：

```text
                              Triton 结构化访存生命周期全景
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Python 源码表达                                                          │
│    offs = pid * BLOCK + tl.arange(0, BLOCK)                                 │
│    values = tl.load(x_ptr + offs, mask=offs < n)                            │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ AST Lowering
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. Pointer-style TTIR (离散指针与掩码)                                       │
│    %ptrs = tt.addptr %base, %offs                                           │
│    %val  = tt.load %ptrs, %mask                                             │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ TritonRaiseTensorView (仿射状态机推导)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. TensorView TTIR (结构化几何与内存解耦)                                    │
│    %view = tt.make_tensor_view %base, [%shape], [%strides], [%offsets]       │
│    %val  = tt.load_view %view {boundaryCheck = [0]}                         │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ MaterializeChunkLoops (硬件 VLEN 切分)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. Vector / Chunk-level TTIR                                                │
│    scf.for 沿 fastest 维度按 VLEN 步长迭代，动态发射向量加载                     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ TritonToLLVM Lowering
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. 目标机器码 / 硬件 DMA 指令                                                │
│    发射 2D DMA 硬件描述符传输 / RISC-V 变长向量加载 (vsetvl + vle.v)         │
└─────────────────────────────────────────────────────────────────────────────┘
```

**三大核心设计总结**：
1. **寻址与副作用分离**：`tt.make_tensor_view`（纯几何、`Pure`、可外提）与 `tt.load_view`（物理读、细粒度副作用）明确了编译优化的安全边界；
2. **声明式 ODS 契约保障**：`SameVariadicOperandSize` 解决了多组可变参数的物理扁平化存储；`TypesMatchWith` 确保了类型系统在验证与语法解析中的双向自洽；
3. **逆向仿射提阶的鲁棒性**：`TritonRaiseTensorView` 使得 Triton 能够继续保持 Python 前端指针算术的表达灵活性，同时在编译器后端为专有硬件还原出完整的 DMA 结构化参数。
