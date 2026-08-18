/**
 * @file 02-linalg-tile-and-fuse-slice-propagation.cpp
 * @brief MLIR Linalg 算子平铺与生产者逆向切片推导 (Producer-Consumer Tiling & Slice Propagation)
 *
 * =========================================================================================
 * 💡 架构设计与编译器考点背景 (MLIR Linalg / Polyhedral Model / Loop Tiling):
 * =========================================================================================
 * 1. 为什么算子融合不能只看全局 Tensor，而必须下沉到 Tile 级别？
 *    - 传统图融合 (如 Conv + Bias + ReLU) 只能融合逐元素算子；但面对大尺寸特征图时，中间结果无法放入 GPU Shared Memory / SRAM。
 *    - MLIR `linalg.generic` 引入 **Tile and Fuse** 机制 (如 `linalg::fuseProducerOfTensor`):
 *      下游 Consumer 算子仅切分出一个局部小块 (Tile, 如 16x16)，编译器依据 Affine Map 逆向推导出
 *      上游 Producer 必须计算的对应切片 (Input Slice)，将两者放入同一个循环体内，彻底消除全局显存 (HBM) 往返。
 *
 * 2. 核心数学与算法模型：
 *    - **仿射映射 (Affine Map)**: 将迭代空间坐标 (d0, d1, ...) 映射为张量索引坐标 (expr0, expr1, ...)。
 *    - **区间逆向求交与包围盒扩展 (Bounding Box Interval Propagation)**:
 *      若下游请求输出区间 [offset, offset + size)，对于滑动窗口/卷积算子 (Kernel K, Stride S, Dilation D)，
 *      上游所需的输入区间闭包为：
 *          Input_Start = offset * S
 *          Input_End   = (offset + size - 1) * S + (K - 1) * D + 1
 *          Input_Size  = Input_End - Input_Start = (size - 1) * S + (K - 1) * D + 1
 *    - **光环效应与重算开销分析 (Halo Region & Recomputation Trade-off)**:
 *      相邻 Tile 在重叠卷积区域会产生 Halo 冗余数据。编译器需精确量化：
 *      Tile 分块带来的 SRAM Cache 收益 vs Halo 重复计算/搬运代价。
 * =========================================================================================
 */

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// =========================================================================================
// 1. 基础几何与仿射切片数据结构
// =========================================================================================

/// 一维离散区间切片 [offset, offset + size)，步长为 stride
struct Slice1D {
  int64_t offset = 0; // 起始下标
  int64_t size = 0;   // 元素数量
  int64_t stride = 1; // 步长 (用于 Strided Tile)

  int64_t end() const { return offset + size; }

  bool operator==(const Slice1D &other) const {
    return offset == other.offset && size == other.size &&
           stride == other.stride;
  }
};

/// 多维张量切片 (Bounding Box)
struct TensorSlice {
  std::vector<Slice1D> dims;

  size_t rank() const { return dims.size(); }

  int64_t num_elements() const {
    if (dims.empty())
      return 0;
    int64_t total = 1;
    for (const auto &d : dims)
      total *= d.size;
    return total;
  }

  void dump(const std::string &prefix = "") const {
    if (!prefix.empty())
      std::cout << prefix << ": ";
    std::cout << "[";
    for (size_t i = 0; i < dims.size(); ++i) {
      std::cout << dims[i].offset << ":" << dims[i].end() << " (len="
                << dims[i].size << ")";
      if (i + 1 < dims.size())
        std::cout << ", ";
    }
    std::cout << "] (Total: " << num_elements() << " elems)" << std::endl;
  }
};

// =========================================================================================
// 2. MLIR 风格的仿射算子与推导引擎建模
// =========================================================================================

enum class OpKind {
  Elementwise, // 逐元素恒等映射 (Point-to-Point)
  Transpose,   // 维度转置 (Dim Permute)
  Conv2D,      // 2D 空间卷积/滑动窗口 (Sliding Window Stencil)
  Pooling2D    // 2D 池化
};

struct StencilConfig {
  int64_t kernel_h = 3;
  int64_t kernel_w = 3;
  int64_t stride_h = 1;
  int64_t stride_w = 1;
  int64_t dilation_h = 1;
  int64_t dilation_w = 1;
  int64_t pad_h = 0;
  int64_t pad_w = 0;
};

class LinalgOpModel {
public:
  std::string name;
  OpKind kind;
  std::vector<int64_t> input_shape;
  std::vector<int64_t> output_shape;
  StencilConfig stencil;

  LinalgOpModel(std::string name, OpKind kind,
                std::vector<int64_t> in_shape,
                std::vector<int64_t> out_shape,
                StencilConfig cfg = {})
      : name(std::move(name)), kind(kind), input_shape(std::move(in_shape)),
        output_shape(std::move(out_shape)), stencil(cfg) {}

  /**
   * @brief 核心推导：根据输出张量切片 (Output Slice)，逆向推导该算子所需的输入切片 (Input Slice)
   *
   * 数学推导原理：
   * 1. Elementwise: 输入区间与输出区间严格 1:1 恒等对齐：
   *    In_Slice[i] = Out_Slice[i]
   *
   * 2. Transpose (以 2D 为例): 维度互换：
   *    In_Slice[0] = Out_Slice[1], In_Slice[1] = Out_Slice[0]
   *
   * 3. Conv2D / Stencil (滑动窗口逆向包围盒推导):
   *    设输出在 H 维上的切片为 [out_h, out_h + size_h)
   *    该切片中第 0 个输出点依赖输入: out_h * S
   *    该切片中最后 1 个输出点 (out_h + size_h - 1) 依赖输入的右边界为:
   *        (out_h + size_h - 1) * S + (K - 1) * D
   *    因此覆盖整个输出切片所需的输入连续闭包为:
   *        In_Start = out_h * S
   *        In_Len   = (size_h - 1) * S + (K - 1) * D + 1
   */
  TensorSlice infer_input_slice(const TensorSlice &out_slice) const {
    TensorSlice in_slice;

    switch (kind) {
    case OpKind::Elementwise: {
      in_slice = out_slice;
      break;
    }

    case OpKind::Transpose: {
      in_slice.dims.resize(out_slice.rank());
      if (out_slice.rank() == 2) {
        in_slice.dims[0] = out_slice.dims[1];
        in_slice.dims[1] = out_slice.dims[0];
      }
      break;
    }

    case OpKind::Conv2D:
    case OpKind::Pooling2D: {
      // 假设 shape 为 2D [H, W]
      in_slice.dims.resize(2);

      // H 维度推导
      int64_t out_h_off = out_slice.dims[0].offset;
      int64_t out_h_len = out_slice.dims[0].size;
      int64_t in_h_start = out_h_off * stencil.stride_h;
      int64_t in_h_len = (out_h_len - 1) * stencil.stride_h +
                         (stencil.kernel_h - 1) * stencil.dilation_h + 1;

      // 边界截断保护 (Clamping to Input Bounds)
      in_h_start = std::max<int64_t>(0, in_h_start);
      if (in_h_start + in_h_len > input_shape[0]) {
        in_h_len = input_shape[0] - in_h_start;
      }
      in_slice.dims[0] = {in_h_start, in_h_len, 1};

      // W 维度推导
      int64_t out_w_off = out_slice.dims[1].offset;
      int64_t out_w_len = out_slice.dims[1].size;
      int64_t in_w_start = out_w_off * stencil.stride_w;
      int64_t in_w_len = (out_w_len - 1) * stencil.stride_w +
                         (stencil.kernel_w - 1) * stencil.dilation_w + 1;

      in_w_start = std::max<int64_t>(0, in_w_start);
      if (in_w_start + in_w_len > input_shape[1]) {
        in_w_len = input_shape[1] - in_w_start;
      }
      in_slice.dims[1] = {in_w_start, in_w_len, 1};
      break;
    }
    }

    return in_slice;
  }
};

// =========================================================================================
// 3. Tile & Fuse 链式切片传播与显存/光环开销仿真器
// =========================================================================================

struct FusionPipeline {
  LinalgOpModel producer;
  LinalgOpModel consumer;

  struct GridTilingResult {
    int64_t num_tiles = 0;
    int64_t total_consumer_output_elems = 0;
    int64_t total_producer_computed_elems = 0;
    int64_t total_source_input_fetched_elems = 0;
    int64_t un_fused_global_mem_traffic_elems = 0; // 不融合时的全局内存访存量
    int64_t fused_global_mem_traffic_elems = 0;    // 融合平铺后的全局内存访存量
    double memory_traffic_reduction_ratio = 0.0;
    double halo_recompute_overhead_ratio = 0.0;
  };

  /**
   * @brief 全网格平铺仿真：将 Consumer 输出切分成 tile_h x tile_w 的网格，
   *        并逆向逐级推导整个流水线的数据足迹。
   */
  GridTilingResult simulate_grid_tiling(int64_t tile_h, int64_t tile_w,
                                        bool verbose = true) const {
    GridTilingResult res;

    int64_t out_h = consumer.output_shape[0];
    int64_t out_w = consumer.output_shape[1];

    int64_t total_source_fetches = 0;
    int64_t total_producer_output_generated = 0;

    if (verbose) {
      std::cout << "\n=========================================================="
                << std::endl;
      std::cout << "🚀 执行 Tile and Fuse 逆向切片传播仿真" << std::endl;
      std::cout << "流水线拓扑: [Source Input " << producer.input_shape[0] << "x"
                << producer.input_shape[1] << "] --> Producer ["
                << producer.name << "] --> [Intermediate "
                << producer.output_shape[0] << "x" << producer.output_shape[1]
                << "] --> Consumer [" << consumer.name
                << "] --> [Final Output " << out_h << "x" << out_w << "]"
                << std::endl;
      std::cout << "Consumer 平铺尺寸 (Tile Size): " << tile_h << " x " << tile_w
                << std::endl;
      std::cout << "=========================================================="
                << std::endl;
    }

    int tile_id = 0;
    for (int64_t h = 0; h < out_h; h += tile_h) {
      int64_t cur_tile_h = std::min(tile_h, out_h - h);

      for (int64_t w = 0; w < out_w; w += tile_w) {
        int64_t cur_tile_w = std::min(tile_w, out_w - w);
        tile_id++;

        // 1. 确定当前 Consumer 输出 Tile
        TensorSlice consumer_out_slice;
        consumer_out_slice.dims = {{h, cur_tile_h, 1}, {w, cur_tile_w, 1}};

        // 2. 逆向推导 Producer 输出切片 (即 Consumer 的输入)
        TensorSlice producer_out_slice =
            consumer.infer_input_slice(consumer_out_slice);

        // 3. 再次逆向推导 Producer 所需的原始 Source Input 切片
        TensorSlice source_in_slice =
            producer.infer_input_slice(producer_out_slice);

        total_producer_output_generated += producer_out_slice.num_elements();
        total_source_fetches += source_in_slice.num_elements();

        if (verbose && (tile_id <= 3 || tile_id == 4)) {
          std::cout << "\n--- [Tile #" << tile_id << " @ Coord (" << h << ", "
                    << w << ")] ---" << std::endl;
          consumer_out_slice.dump("  Step 1. Consumer Target Output Slice ");
          producer_out_slice.dump("  Step 2. 逆向推导 Producer Output Slice");
          source_in_slice.dump("  Step 3. 逆向推导 Source Input Slice   ");

          int64_t halo_elems =
              source_in_slice.num_elements() - consumer_out_slice.num_elements();
          std::cout << "  💡 Halo 光环冗余边界点: " << halo_elems
                    << " elems (用于滑动窗口边界扩展)" << std::endl;
        }
      }
    }

    res.num_tiles = tile_id;
    res.total_consumer_output_elems = out_h * out_w;
    res.total_producer_computed_elems = total_producer_output_generated;
    res.total_source_input_fetched_elems = total_source_fetches;

    // --- 显存访存量对比模型 (Memory Traffic Analysis) ---
    // 1. 不融合基线 (Un-fused):
    //    - Producer: 读取 Source Input (1 次) + 写入 Intermediate Buffer 到全局显存 (1 次)
    //    - Consumer: 从全局显存读取 Intermediate Buffer (1 次) + 写入 Final Output (1 次)
    int64_t src_size = producer.input_shape[0] * producer.input_shape[1];
    int64_t inter_size = producer.output_shape[0] * producer.output_shape[1];
    int64_t final_size = consumer.output_shape[0] * consumer.output_shape[1];

    res.un_fused_global_mem_traffic_elems =
        src_size + inter_size + inter_size + final_size;

    // 2. 融合平铺 (Tile & Fuse in SRAM / Registers):
    //    - 中间结果直接在片上 SRAM / 寄存器完成消费，不产生任何全局显存写入与读取！
    //    - 全局访存仅包括: 读取 Source Input (含 Halo 重叠读取) + 写入 Final Output
    res.fused_global_mem_traffic_elems = total_source_fetches + final_size;

    res.memory_traffic_reduction_ratio =
        1.0 - (double)res.fused_global_mem_traffic_elems /
                  res.un_fused_global_mem_traffic_elems;

    res.halo_recompute_overhead_ratio =
        (double)(total_source_fetches - src_size) / src_size;

    return res;
  }
};

// =========================================================================================
// 4. 打印报告与主验证程序
// =========================================================================================

void print_benchmark_report(const FusionPipeline::GridTilingResult &res) {
  std::cout << "\n================ 📊 编译器 Tile & Fuse 收益与代价评估报告 ================"
            << std::endl;
  std::cout << "1. 网格分块总数 (Total Tiles):           " << std::setw(10)
            << res.num_tiles << std::endl;
  std::cout << "2. 最终产出数据量 (Final Output Elems):    " << std::setw(10)
            << res.total_consumer_output_elems << " elems" << std::endl;
  std::cout << "3. 不融合全局显存访存量 (Un-fused HBM):   " << std::setw(10)
            << res.un_fused_global_mem_traffic_elems << " elems" << std::endl;
  std::cout << "4. 融合平铺全局显存访存量 (Fused HBM):     " << std::setw(10)
            << res.fused_global_mem_traffic_elems << " elems" << std::endl;
  std::cout << "5. 全局显存带宽节省比例 (HBM Bandwidth):  " << std::setw(9)
            << std::fixed << std::setprecision(2)
            << res.memory_traffic_reduction_ratio * 100.0 << " % 🚀"
            << std::endl;
  std::cout << "6. Halo 边界重叠搬运冗余比 (Halo Overhead):" << std::setw(9)
            << std::fixed << std::setprecision(2)
            << res.halo_recompute_overhead_ratio * 100.0 << " %"
            << std::endl;
  std::cout << "========================================================================"
            << std::endl;
}

int main() {
  // ---------------------------------------------------------------------------------------
  // 案例 1: 经典 Conv2D 3x3 (Producer) + ReLU/BiasAdd (Consumer)
  // ---------------------------------------------------------------------------------------
  // Source Input: 66 x 66
  // Producer: 3x3 卷积, Stride=1, Pad=0 -> 产出 64 x 64 中间特征图
  // Consumer: 逐元素激活算子 -> 产出 64 x 64 最终结果
  StencilConfig conv3x3_cfg{/*kernel_h=*/3, /*kernel_w=*/3,
                           /*stride_h=*/1, /*stride_w=*/1,
                           /*dilation_h=*/1, /*dilation_w=*/1};

  LinalgOpModel conv_producer("Conv2D_3x3", OpKind::Conv2D, {66, 66}, {64, 64},
                              conv3x3_cfg);
  LinalgOpModel relu_consumer("ReLU_Activation", OpKind::Elementwise, {64, 64},
                              {64, 64});

  FusionPipeline pipeline1{conv_producer, relu_consumer};

  // 以 16 x 16 Tile 尺寸进行平铺推导
  auto res1 = pipeline1.simulate_grid_tiling(16, 16, /*verbose=*/true);
  print_benchmark_report(res1);

  // ---------------------------------------------------------------------------------------
  // 案例 2: 不同 Tile 尺寸对 Halo 光环开销的敏感度分析 (Tuning Tile Size)
  // ---------------------------------------------------------------------------------------
  std::cout << "\n>>> [Tile Size Sensitivity Analysis] 不同分块尺寸对显存带宽与 Halo 开销的影响:"
            << std::endl;
  std::cout << std::left << std::setw(12) << "Tile Size" << std::setw(15)
            << "Num Tiles" << std::setw(20) << "HBM Saved (%)" << std::setw(20)
            << "Halo Overhead (%)" << std::endl;

  std::vector<int64_t> test_tile_sizes = {8, 16, 32, 64};
  for (int64_t ts : test_tile_sizes) {
    auto r = pipeline1.simulate_grid_tiling(ts, ts, /*verbose=*/false);
    std::cout << std::left << std::setw(12)
              << (std::to_string(ts) + "x" + std::to_string(ts))
              << std::setw(15) << r.num_tiles << std::setw(20)
              << (std::to_string(r.memory_traffic_reduction_ratio * 100.0)
                      .substr(0, 5) +
                  " %")
              << std::setw(20)
              << (std::to_string(r.halo_recompute_overhead_ratio * 100.0)
                      .substr(0, 5) +
                  " %")
              << std::endl;
  }

  return 0;
}
