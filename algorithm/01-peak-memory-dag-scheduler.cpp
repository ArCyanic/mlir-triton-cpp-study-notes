/**
 * @file 01-peak-memory-dag-scheduler.cpp
 * @brief 编译器 DAG 静态显存峰值最小化调度算法 (MLIR Use-Def + XOR-Sum O(1) 优化版)
 *
 * =========================================================================================
 * 💡 架构设计与顶级算法考点复刻 (MLIR + DOD + Math Trick):
 * =========================================================================================
 * 1. 算子 (Operation) 与值 (Value) 的解耦 (MLIR 官方范式):
 *    - 【mlir::Operation (算子节点)】: 仅维护 getOperands() 与
 * getResults()，不存任何下游算子指针。
 *    - 【mlir::Value (SSA 值 / Tensor 边)】: 维护 getDefiningOp() (Def) 与
 * getUsers() (Use-Def 链)。
 *    -  出度挂在 Tensor 上，用于释放内存；入度挂载 Op
 * 上，用于拓扑排序中判断是否就绪。
 *
 * 2. 数据驱动 (Data-Oriented Design, DOD) 与状态可重入:
 *    - 图的 IR 是不可变静态只读数据，使用连续扁平数组 (Flat std::vector) 存储。
 *    - 调度算法运行在独立的 `ScheduleContext`
 * 瞬态上下文上，无任何哈希表与动态堆分配。
 *
 * 3. 🌟 XOR 异或累加器黑魔法 (O(1) Single-Cycle Last-Sibling Discovery):
 *    - 当某个 Value 被多个消费者共享时，使用位运算 `active_user_xor_sum ^=
 * op_id` 维护剩余消费者。
 *    - 当 `remaining_users == 1` 时，`active_user_xor_sum`
 * 中的值直接就是最后一名同伴算子的 ID！
 *    - 将大 Fan-out 场景下的同伴定位从 O(U) 线性循环直接压缩至 1 个 CPU
 * 时钟周期的 O(1)！
 *
 * 4. 惰性动态权值优先队列 (Lazy Versioning Min-Heap):
 *    - 配合版本号校验，自适应动态响应显存释放机会，以 O(log K)
 * 实现全局峰值最小化。
 * =========================================================================================
 */

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

// =========================================================================================
// 1. MLIR 风格的核心 IR 抽象: Value (SSA 边) 与 Operation (图节点)
// =========================================================================================

/**
 * @brief 复刻 mlir::Value (表示张量 SSA 数据流边)
 */
class Value {
public:
  // 💡 采用现代 C++ 惯用法: 传值 (Pass-by-value) + std::move
  // 兼顾左值拷贝与右值临时变量的高效移动构造 (0 堆内存分配)
  Value(int id, std::string name, size_t size_mb, int defining_op_id)
      : id_(id), name_(std::move(name)), size_mb_(size_mb),
        defining_op_id_(defining_op_id) {}

  int getId() const { return id_; }
  const std::string &getName() const { return name_; }
  size_t getSizeMb() const { return size_mb_; }

  // Def: 谁生成了该 Value (-1 表示图的外部输入)
  int getDefiningOp() const { return defining_op_id_; }
  void setDefiningOp(int op_id) { defining_op_id_ = op_id; }

  // Use: 谁消费了该 Value (Use-Def 链)
  const std::vector<int> &getUsers() const { return users_; }
  int getNumUses() const { return static_cast<int>(users_.size()); }
  bool use_empty() const { return users_.empty(); }

  void addUser(int consumer_op_id) { users_.push_back(consumer_op_id); }

private:
  int id_;
  std::string name_;
  size_t size_mb_;
  int defining_op_id_;     // Def
  std::vector<int> users_; // Use-Def Chain
};

/**
 * @brief 复刻 mlir::Operation (表示计算算子节点)
 */
class Operation {
public:
  // 💡 现代 C++ 移动语义设计 (Pass-by-value and std::move):
  // 1. 触发的构造函数: operands_(std::move(operands)) 与
  // results_(std::move(results))
  //    精准匹配 std::vector 的【移动构造函数】: vector(vector&& other)
  //    noexcept;
  // 2. 底层物理机制 (指针所有权转移 / Pointer Stealing):
  //    - 仅需拷贝 24 字节的内部指针 (start, finish, end_of_storage)。
  //    - 0 次堆内存重新分配 (0 malloc)、0 次元素逐个深拷贝！
  // 3. 适用场景:
  //    - 当调用方传入右值临时变量 (如 `{v_out, t_out}`)
  //    时，实现全生命周期零开销移入。
  Operation(int id, std::string name, std::vector<int> operands,
            std::vector<int> results)
      : id_(id), name_(std::move(name)), operands_(std::move(operands)),
        results_(std::move(results)) {}

  int getId() const { return id_; }
  const std::string &getName() const { return name_; }

  // 输入操作数 (Operands)
  const std::vector<int> &getOperands() const { return operands_; }
  int getNumOperands() const { return static_cast<int>(operands_.size()); }
  int getOperand(int idx) const { return operands_[idx]; }

  // 输出结果 (Results)
  const std::vector<int> &getResults() const { return results_; }
  int getNumResults() const { return static_cast<int>(results_.size()); }
  int getResult(int idx) const { return results_[idx]; }

private:
  int id_;
  std::string name_;
  std::vector<int> operands_; // 输入 Value IDs
  std::vector<int> results_;  // 输出 Value IDs
};

// =========================================================================================
// 2. 调度瞬态上下文 (ScheduleContext - 纯数组 DOD + 图论出入度统一建模)
// =========================================================================================

struct ScheduleContext {
  std::vector<int>
      tensor_out_degree; // 🌟 下标: value_id -> 张量剩余未消费的出度
                         // (归零则彻底释放显存)
  std::vector<int>
      active_user_xor_sum; // 🌟 下标: value_id -> 剩余未发射消费者的 ID 异或和
                           // (O(1) 定位最后一人)
  std::vector<int> operation_in_degree; // 🌟 下标: op_id    ->
                                        // 算子剩余未完成的入度 (归零则算子就绪)
  std::vector<int> op_versions; // 下标: op_id    -> 算子当前优先级版本号

  // 💡 状态标志设计考量 (std::vector<uint8_t> vs. std::vector<bool> vs.
  // llvm::BitVector):
  // 1. 为什么用 uint8_t 而非 std::vector<bool>?
  //    - 规避 std::vector<bool> 的位压缩代理对象 (Proxy Reference) 与额外的位移
  //    (Shift/Mask) 开销。
  //    - 原生 1 字节寻址 (movzx)，同时一条 64B L1 Cache Line 可装载 64
  //    个算子状态 (比 int 节省 4 倍缓存)。
  // 2. 进阶拓展 (llvm::BitVector / 硬件位图加速):
  //    - 在 LLVM/MLIR 数据流分析 (如 Liveness) 中，常采用类似 llvm::BitVector
  //    (uint64_t 数组) 实现字级并行集合运算。
  //    - 当算子数量 <= 64 时，可直接用单个 `uint64_t is_ready_mask`，配合
  //    `__builtin_ctzll` 硬件单指令 0 循环跳跃查找下一个就绪算子。
  std::vector<uint8_t> is_ready;     // 下标: op_id    -> 是否已就绪
  std::vector<uint8_t> is_scheduled; // 下标: op_id    -> 是否已发射执行

  ScheduleContext(const std::vector<Value> &values,
                  const std::vector<Operation> &ops) {
    size_t num_values = values.size();
    size_t num_ops = ops.size();

    // 1. 初始化各 Tensor 的动态出度与初始 XOR 异或和
    tensor_out_degree.resize(num_values);
    active_user_xor_sum.assign(num_values, 0);

    for (size_t i = 0; i < num_values; ++i) {
      tensor_out_degree[i] = values[i].getNumUses();
      int xor_sum = 0;
      for (int consumer_id : values[i].getUsers()) {
        xor_sum ^= consumer_id; // 累加所有消费者的 ID
      }
      active_user_xor_sum[i] = xor_sum;
    }

    // 2. 基于 Use-Def 链自动推导各 Operation 的初始入度
    operation_in_degree.assign(num_ops, 0);
    for (size_t i = 0; i < num_ops; ++i) {
      int in_deg = 0;
      for (int operand_val_id : ops[i].getOperands()) {
        if (values[operand_val_id].getDefiningOp() != -1) {
          in_deg++;
        }
      }
      operation_in_degree[i] = in_deg;
    }

    op_versions.assign(num_ops, 0);
    is_ready.assign(num_ops, 0);
    is_scheduled.assign(num_ops, 0);
  }
};

// 堆元素定义
struct ReadyOpItem {
  int64_t net_delta; // 净显存变化量 (MB, 越小越优先)
  int op_id;
  int version;

  bool operator>(const ReadyOpItem &other) const {
    if (net_delta != other.net_delta)
      return net_delta > other.net_delta;
    return op_id > other.op_id;
  }
};

// =========================================================================================
// 3. MLIR 风格的计算图容器与调度算法实现
// =========================================================================================

class ComputationGraph {
public:
  int createValue(std::string name, size_t size_mb, int defining_op_id = -1) {
    int vid = static_cast<int>(values_.size());
    val_name_to_id_[name] = vid;
    values_.emplace_back(vid, std::move(name), size_mb, defining_op_id);
    return vid;
  }

  int createOperation(std::string name, std::vector<int> operands,
                      std::vector<int> results) {
    int op_id = static_cast<int>(ops_.size());

    // 1. 自动构建 Use-Def 链 (在移走容器前先读取其内部 ID)
    for (int op_val_id : operands) {
      values_[op_val_id].addUser(op_id);
    }
    for (int res_val_id : results) {
      values_[res_val_id].setDefiningOp(op_id);
    }

    op_name_to_id_[name] = op_id;

    // 2. 触发 Operation 移动构造函数 (Zero Heap Allocations)
    ops_.emplace_back(op_id, std::move(name), std::move(operands),
                      std::move(results));
    return op_id;
  }

  struct SimulationResult {
    std::vector<int> scheduled_op_ids;
    std::vector<size_t> memory_timeline_mb;
    size_t peak_memory_mb = 0;
  };

  // --- 调度算法 A: 朴素 BFS 拓扑排序 (FIFO) ---

  SimulationResult schedule_bfs() const {
    ScheduleContext ctx(values_, ops_);
    std::queue<int> q;

    for (const auto &op : ops_) {
      if (ctx.operation_in_degree[op.getId()] == 0) {
        q.push(op.getId());
      }
    }

    SimulationResult result;
    size_t current_mem = 0;

    while (!q.empty()) {
      int op_id = q.front();
      q.pop();
      result.scheduled_op_ids.push_back(op_id);

      const auto &op = ops_[op_id];

      // 1. 产生 Results 显存
      for (int res_vid : op.getResults()) {
        current_mem += values_[res_vid].getSizeMb();
      }

      // 2. 消费 Operands 显存 (张量出度归零则释放)
      for (int op_vid : op.getOperands()) {
        ctx.tensor_out_degree[op_vid]--;
        if (ctx.tensor_out_degree[op_vid] == 0) {
          current_mem -= values_[op_vid].getSizeMb();
        }
      }

      result.memory_timeline_mb.push_back(current_mem);
      result.peak_memory_mb = std::max(result.peak_memory_mb, current_mem);

      // 3. 沿 Use-Def 链消减下游算子入度
      for (int res_vid : op.getResults()) {
        for (int consumer_op_id : values_[res_vid].getUsers()) {
          ctx.operation_in_degree[consumer_op_id]--;
          if (ctx.operation_in_degree[consumer_op_id] == 0) {
            q.push(consumer_op_id);
          }
        }
      }
    }
    return result;
  }

  // --- 调度算法 B: MLIR Use-Def + XOR O(1) + 惰性优先队列显存贪心调度 ---

  SimulationResult schedule_lazy_pq_greedy() const {
    ScheduleContext ctx(values_, ops_);
    std::priority_queue<ReadyOpItem, std::vector<ReadyOpItem>,
                        std::greater<ReadyOpItem>>
        pq;

    auto eval_net_delta = [&](int op_id) -> int64_t {
      const auto &op = ops_[op_id];
      int64_t out_size = 0;
      for (int res_vid : op.getResults()) {
        out_size += values_[res_vid].getSizeMb();
      }

      int64_t freed_size = 0;
      for (int op_vid : op.getOperands()) {
        if (ctx.tensor_out_degree[op_vid] == 1) { // 剩余出度为 1 (最后一次消费)
          freed_size += values_[op_vid].getSizeMb();
        }
      }
      return out_size - freed_size;
    };

    auto push_ready_op = [&](int op_id) {
      ctx.is_ready[op_id] = 1;
      ctx.op_versions[op_id]++;
      int64_t delta = eval_net_delta(op_id);
      pq.push({delta, op_id, ctx.op_versions[op_id]});
    };

    // 初始就绪算子入堆 (入度为 0)
    for (const auto &op : ops_) {
      if (ctx.operation_in_degree[op.getId()] == 0) {
        push_ready_op(op.getId());
      }
    }

    SimulationResult result;
    size_t current_mem = 0;

    while (!pq.empty()) {
      auto top = pq.top();
      pq.pop();

      int op_id = top.op_id;

      // 惰性版本检查
      if (top.version != ctx.op_versions[op_id] || ctx.is_scheduled[op_id]) {
        continue;
      }

      ctx.is_scheduled[op_id] = 1;
      result.scheduled_op_ids.push_back(op_id);

      const auto &op = ops_[op_id];

      // 1. 产出 Results
      for (int res_vid : op.getResults()) {
        current_mem += values_[res_vid].getSizeMb();
      }

      // 2. 消费 Operands 并通过 XOR 异或和 O(1) 触发同伴算子的优先级跃升
      for (int op_vid : op.getOperands()) {
        ctx.tensor_out_degree[op_vid]--;
        ctx.active_user_xor_sum[op_vid] ^=
            op_id; // 🌟 异或消除当前算子自身 (A ^ A = 0)

        if (ctx.tensor_out_degree[op_vid] == 0) {
          current_mem -= values_[op_vid].getSizeMb(); // 出度归零，张量显存彻底释放
        } else if (ctx.tensor_out_degree[op_vid] == 1) {
          // ===================================================================
          // 🌟 核心优化机制：同伴算子显存红利跃升 (Priority Promotion)
          // -------------------------------------------------------------------
          // 1. 触发机理：
          //    当张量出度降为 1 时，意味着该张量只剩下最后 1 个活跃消费者 (last_sibling_op_id)。
          //    此时该同伴算子一旦执行，便能将该张量出度降为 0 并彻底释放其占用的显存空间！
          //    因此该同伴算子的净显存增量 eval_net_delta 会大幅下降（获得负向显存释放红利），
          //    具有极高的调度紧迫性，需要立即将其推入优先队列以刷新其优先级。
          //
          // 2. O(1) 极速定位：
          //    无需遍历消费者列表，当前 active_user_xor_sum[op_vid] 的值即为最后唯一的同伴算子 ID。
          //
          // 3. 为什么必须检查 is_ready？
          //    同伴算子虽然在此张量上获得了释放红利，但它的其它输入操作数可能尚未由前序算子计算完毕
          //    (即其 operation_in_degree 尚未归零)。若直接入堆会导致其可能在依赖未就绪时被提前发射，
          //    击穿 DAG 拓扑因果序。
          //    - 若已就绪 (is_ready == 1)：已在堆中等待，此时通过 push_ready_op 递增版本并赋予更优的 delta，
          //      实现 O(log N) 跃升到堆顶优先调度；
          //    - 若未就绪 (is_ready == 0)：放弃本次提前入队。未来当其最后一个前置依赖完成、入度归零时，
          //      Step 3 会触发入队，届时读取到的 tensor_out_degree 同样为 1，依然能享受该显存释放红利。
          //
          // 4. 为什么无需检查 !is_scheduled？
          //    已执行的算子在发射时已从 active_user_xor_sum 中异或剔除，留存在异或和中的算子必然尚未被调度。
          // ===================================================================
          int last_sibling_op_id = ctx.active_user_xor_sum[op_vid];
          if (ctx.is_ready[last_sibling_op_id]) {
            push_ready_op(last_sibling_op_id);
          }
        }
      }

      result.memory_timeline_mb.push_back(current_mem);
      result.peak_memory_mb = std::max(result.peak_memory_mb, current_mem);

      // 3. 通过输出 Value 的 Use-Def 链消减下游算子入度
      for (int res_vid : op.getResults()) {
        for (int consumer_op_id : values_[res_vid].getUsers()) {
          ctx.operation_in_degree[consumer_op_id]--;
          if (ctx.operation_in_degree[consumer_op_id] == 0) {
            push_ready_op(consumer_op_id);
          }
        }
      }
    }

    return result;
  }

  const Operation &getOp(int id) const { return ops_[id]; }
  const Value &getValue(int id) const { return values_[id]; }

private:
  std::vector<Value> values_;
  std::vector<Operation> ops_;
  std::unordered_map<std::string, int> val_name_to_id_;
  std::unordered_map<std::string, int> op_name_to_id_;
};

// =========================================================================================
// 4. 打印与测试主程序
// =========================================================================================

void print_timeline(const std::string &title, const ComputationGraph &g,
                    const ComputationGraph::SimulationResult &res) {
  std::cout << "\n================ " << title
            << " ================" << std::endl;
  std::cout << "Peak Memory Footprint: " << res.peak_memory_mb << " MB"
            << std::endl;
  std::cout << "Step-by-step Schedule & Memory Timeline:" << std::endl;
  for (size_t i = 0; i < res.scheduled_op_ids.size(); ++i) {
    int op_id = res.scheduled_op_ids[i];
    std::cout << "  Step " << std::setw(2) << i + 1 << ": Op [" << std::left
              << std::setw(22) << g.getOp(op_id).getName() << "] -> "
              << "Live Mem: " << std::right << std::setw(4)
              << res.memory_timeline_mb[i] << " MB  ";

    int bars = res.memory_timeline_mb[i] / 50;
    std::cout << std::string(bars, '#') << std::endl;
  }
}

int main() {
  ComputationGraph g;

  // =========================================================================
  // 构造工业级复杂多分支计算图：多模态跨注意力特征融合层 (Multi-Modal
  // Cross-Attention Block) 涵盖视觉(Vision)、文本(Text)、音频(Audio)
  // 三路异构流，每路流均有中间高维膨胀与压缩。
  //
  //              [Vision_In 200M]      [Text_In 200M]      [Audio_In 200M]
  //                     │                    │                    │
  //              [V_Up 600M]           [T_Up 700M]          [A_Up 800M]    <--
  //              💥 中间激活特征爆炸
  //                     │                    │                    │
  //              [V_Down 100M]         [T_Down 100M]        [A_Down 100M]  <--
  //              📉 压缩降维并释放高维特征
  //                     \                    │                    /
  //                      ────────────────────┼────────────────────
  //                                          ▼
  //                             [Cross_Modal_Fusion 300M]
  // =========================================================================

  // 1. 创建 SSA Values (Tensor 边)
  int v_in = g.createValue("Vision_In", 200);
  int t_in = g.createValue("Text_In", 200);
  int a_in = g.createValue("Audio_In", 200);

  int v_hid = g.createValue("V_Hidden_600M", 600);
  int v_out = g.createValue("V_Compressed_100M", 100);

  int t_hid = g.createValue("T_Hidden_700M", 700);
  int t_out = g.createValue("T_Compressed_100M", 100);

  int a_hid = g.createValue("A_Hidden_800M", 800);
  int a_out = g.createValue("A_Compressed_100M", 100);

  int f_out = g.createValue("Fusion_Out", 300);

  // 2. 创建 Operations (算子节点) - 拓扑全由 Value 的 Use-Def 链自动建立！
  g.createOperation("Op0_Init_Vision", {}, {v_in});
  g.createOperation("Op1_Init_Text", {}, {t_in});
  g.createOperation("Op2_Init_Audio", {}, {a_in});

  // 视觉流
  g.createOperation("Op3_Vision_Expand", {v_in}, {v_hid});
  g.createOperation("Op4_Vision_Compress", {v_hid}, {v_out});

  // 文本流
  g.createOperation("Op5_Text_Expand", {t_in}, {t_hid});
  g.createOperation("Op6_Text_Compress", {t_hid}, {t_out});

  // 音频流
  g.createOperation("Op7_Audio_Expand", {a_in}, {a_hid});
  g.createOperation("Op8_Audio_Compress", {a_hid}, {a_out});

  // 跨模态多路聚合
  g.createOperation("Op9_Cross_Fusion", {v_out, t_out, a_out}, {f_out});

  // 3. 运行仿真对比
  auto res_bfs = g.schedule_bfs();
  auto res_lazy_pq = g.schedule_lazy_pq_greedy();

  print_timeline("Strategy 1: Naive BFS Topological Sort (FIFO Queue)", g,
                 res_bfs);
  print_timeline(
      "Strategy 2: MLIR Use-Def + XOR O(1) + Lazy PQ Greedy Scheduler", g,
      res_lazy_pq);

  std::cout << "\n------------------ Benchmark Summary ------------------"
            << std::endl;
  std::cout << "1. Naive BFS Peak Memory:     " << std::setw(5)
            << res_bfs.peak_memory_mb << " MB" << std::endl;
  std::cout << "2. MLIR XOR-PQ Peak:          " << std::setw(5)
            << res_lazy_pq.peak_memory_mb << " MB" << std::endl;
  std::cout << "3. Peak Memory Saved:         " << std::fixed
            << std::setprecision(2)
            << (100.0 * (res_bfs.peak_memory_mb - res_lazy_pq.peak_memory_mb) /
                res_bfs.peak_memory_mb)
            << " %" << std::endl;

  return 0;
}
