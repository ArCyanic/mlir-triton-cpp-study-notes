# 编译流水线、链接模型与 ELF

> 本文系统性解构从 C++ 源码到可执行文件的端到端**编译与链接流水线**，涵盖**作用域、存储期与链接属性（Linkage）体系**、**ELF 二进制文件格式的 Section 与 Segment 双重视图**、**C++ 名字改编（Name Mangling）与 `extern "C"`**、全局对象 `.init_array` 构造顺序以及**静态库链接顺序敏感性**的底层机理。

---

## 目录

- [1. 编译流水线与阶段划分](#1-编译流水线与阶段划分)
- [2. 作用域、存储期与链接属性](#2-作用域存储期与链接属性)
  - [2.1 作用域 vs 存储期](#21-作用域-vs-存储期)
  - [2.2 链接属性（Linkage）](#22-链接属性linkage)
  - [2.3 匿名命名空间与 static](#23-匿名命名空间与-static)
- [3. ELF 格式与双重视图](#3-elf-格式与双重视图)
  - [3.1 Section 节区分布](#31-section-节区分布)
  - [3.2 `.bss` 节分配机制](#32-bss-节分配机制)
- [4. Name Mangling 与 ABI 互操作](#4-name-mangling-与-abi-互操作)
  - [4.1 Itanium ABI 编码规则](#41-itanium-abi-编码规则)
  - [4.2 `extern "C"` 桥接机制](#42-extern-c-桥接机制)
- [5. 全局对象生命周期与 `.init_array`](#5-全局对象生命周期与-init_array)
  - [5.1 `main` 前执行链路](#51-main-前执行链路)
  - [5.2 静态初始化顺序与 Meyers' Singleton](#52-静态初始化顺序与-meyers-singleton)
- [6. 静态链接与动态链接](#6-静态链接与动态链接)
  - [6.1 静态库单遍扫描算法](#61-静态库单遍扫描算法)
  - [6.2 动态链接与延迟绑定（PLT/GOT）](#62-动态链接与延迟绑定pltgot)

---

## 1. 编译流水线与阶段划分

从一个 `hello.cpp` 文本文件到操作系统中可运行的二进制程序，经历了四个标准阶段：

```text
源码 (main.cpp)
      │
      ├─► 【1. 预处理 (Preprocessing)】 ──► gcc -E main.cpp -o main.i
      │    • 展开 #include 头文件 (纯文本替换)
      │    • 处理 #define 宏替换与 #ifdef 条件编译
      │    • 剔除所有代码注释
      │
      ├─► 【2. 编译 (Compilation)】      ──► gcc -S main.i -o main.s
      │    • 词法/语法分析 (AST) -> 语义分析 (Type Check)
      │    • 中间代码生成与优化 (LLVM IR / Middle-End Optimization)
      │    • 生成目标平台的汇编语言代码 (.s)
      │
      ├─► 【3. 汇编 (Assembly)】         ──► gcc -c main.s -o main.o
      │    • 将汇编指令翻译为机器码
      │    • 生成 ELF 可重定位目标文件 (Relocatable Object File, .o)
      │
      └─► 【4. 链接 (Linking)】          ──► gcc main.o -o main
           • 符号决议 (Symbol Resolution)：将未决符号与具体函数/变量绑定
           • 重定位 (Relocation)：修补代码段中的内存绝对/相对偏移地址
           • 合并生成最终的可执行文件 (Executable) 或动态共享库 (.so)
```

---

## 2. 作用域、存储期与链接属性

在 C++ 类型与符号系统中，一个变量或函数由四个正交维度定义：

### 2.1 作用域 vs 存储期

* **作用域（Scope，编译期视角）**：源码中能合法访问该标识符名称的文本区域（块作用域、类作用域、命名空间作用域）。
* **存储期（Storage Duration，运行期视角）**：对象占据内存空间的时间跨度：
  1. **自动存储期（Automatic）**：函数栈帧变量，进入作用域分配，退出作用域自动析构。
  2. **静态存储期（Static）**：全局变量、`static` 变量，程序启动时分配，程序终止时释放。
  3. **动态存储期（Dynamic）**：通过 `new`/`malloc` 在堆上开辟，由程序员显式调用 `delete`/`free` 或智能指针释放。
  4. **线程存储期（Thread-local）**：`thread_local` 声明的变量，随线程创建与销毁。

---

### 2.2 链接属性（Linkage）

链接属性决定了**不同翻译单元（Translation Unit, `.cpp` 编译产物）之间能否通过符号名共享同一个实体**：

| 链接属性 | 含义 | 声明方式 | 符号表可见性 |
| :--- | :--- | :--- | :--- |
| **外部链接 (External Linkage)** | 跨翻译单元全局共享，所有 `.cpp` 看到的都是同一个实体 | 普通非 `static` 全局变量/函数、类声明、`extern` 变量 | 在 `.symtab` 中标记为 `GLOBAL` |
| **内部链接 (Internal Linkage)** | 仅在当前当前 `.cpp` 内部可见，不同 `.cpp` 中同名实体互不干扰 | 全局 `static` 变量/函数、匿名命名空间（`namespace { ... }`） | 在 `.symtab` 中标记为 `LOCAL` |
| **无链接 (No Linkage)** | 纯局部实体，外界完全无法引用 | 函数局部变量、局部 `typedef` / `using` | 不进入全局链接符号表 |

---

### 2.3 匿名命名空间与 static

* `static void foo()`：将函数声明为内部链接，仅能作用于函数和变量，无法用于类型（`class`/`struct`）。
* `namespace { class MyHelper {}; }`：**匿名命名空间**为其内部的**所有变量、函数以及类类型**赋予唯一的内部链接属性，是现代 C++ 替代全局 `static` 的推荐做法。

---

## 3. ELF 格式与双重视图

在 Linux 环境下，`.o` 目标文件、`.so` 共享库和可执行文件均采用 **ELF（Executable and Linkable Format）** 格式。

```text
               ELF 文件结构双重视图 (Dual View)
               
      【链接器视角：Section Header Table】
┌─────────────────────────────────────────────────────────┐
│ .text       : 机器指令代码 (Executable Code)              │
│ .rodata     : 只读数据 (字符串常量、虚表 vtable、const 常量)│
│ .data       : 已初始化的全局 / 静态变量                   │
│ .bss        : 未初始化或初始化为 0 的全局变量 (仅记录大小)  │
│ .symtab     : 符号表 (记录所有全局与局部符号的偏移与类型)   │
│ .rela.text  : 代码重定位表 (记录哪些指令地址需要链接时修补) │
│ .init_array : 全局构造函数指针数组                        │
└────────────────────────────┬────────────────────────────┘
                             │ (链接器将多个 Section 合并成 Segment)
                             ▼
      【装载器/内核视角：Program Header Table】
┌─────────────────────────────────────────────────────────┐
│ Segment 1: PT_LOAD [R-X] ──► 映射到只读代码内存页 (.text, .rodata)│
│ Segment 2: PT_LOAD [RW-] ──► 映射到可读写数据页 (.data, .bss)   │
└─────────────────────────────────────────────────────────┘
```

---

### 3.1 Section 节区分布

* **`.text`**：编译生成的纯机器指令。
* **`.rodata`（Read-Only Data）**：只读常量区。字面量字符串（`"Hello World"`）、浮点常量、**类的虚函数表（vtable）** 均存放在此。
* **`.data`**：存放初始值不为 0 的全局变量和 `static` 变量，其初始值直接打包在磁盘文件中。
* **`.bss`（Block Started by Symbol）**：存放未初始化或初始值为 0 的全局变量。

---

### 3.2 `.bss` 节分配机制

* **原理解密**：未初始化的全局变量（如 `int big_array[1000000];`）其内容全是 0。如果直接存入文件，会白白浪费数兆磁盘空间。
* **ELF 设计**：`.bss` 节在 ELF 头部中**仅仅记录了它在运行时所需的总字节大小（Size）**，磁盘文件中**不分配任何实际数据存储**。
* **装载时分配**：当操作系统内核 `execve` 加载程序时，根据 `.bss` 记录的大小在内存中直接映射出一批清零的物理页（Page），高效且节省磁盘。

---

## 4. Name Mangling 与 ABI 互操作

### 4.1 Itanium ABI 编码规则

C 语言不支持函数重载，函数名与汇编符号名一一对应（如 `foo()` $\to$ `foo`）。  
C++ 支持 **函数重载（Overloading）、命名空间（Namespace）、类成员函数与模板**。为了让链接器能够区分同名重载函数，编译器前端必须对符号名进行编码改编：

```cpp
namespace mlir {
    void process(int a, float b);
}
```

在 GCC/Clang（遵循 Itanium C++ ABI）下，该函数被编译改编为符号：
```text
_ZN4mlir7processEif
 │ │ │    │      │└─ 参数 2: float (f)
 │ │ │    │      └── 参数 1: int (i)
 │ │ │    └───────── 函数名: process (长度 7)
 │ │ └────────────── 命名空间: mlir (长度 4)
 │ └──────────────── 嵌套名称前缀 (Nested Name)
 └────────────────── C++ 符号标准前缀 (_Z)
```

---

### 4.2 `extern "C"` 桥接机制

```cpp
#ifdef __cplusplus
extern "C" {
#endif

// 强制编译器使用 C 语言的链接规则生成未被改编的纯符号名 "launch_kernel"
void launch_kernel(void* stream, int grid_dim);

#ifdef __cplusplus
}
#endif
```

* **核心作用**：
  1. 告诉 C++ 编译器：**不要对该函数进行 Name Mangling**，符号表中直接记录为裸名字 `launch_kernel`；
  2. 使得该共享库（`.so`）可以被纯 C 程序链接，或者被 Python（`ctypes` / `cffi`）、Rust 等外部语言通过 `dlsym` 动态查找调用。

---

## 5. 全局对象生命周期与 `.init_array`

### 5.1 `main` 前执行链路

```cpp
struct GlobalInitializer {
    GlobalInitializer() {
        std::cout << "1. Execute before main!\n";
    }
};

GlobalInitializer g_init; // 全局对象

int main() {
    std::cout << "2. Enter main()\n";
    return 0;
}
```

#### 底层执行链路：
```text
操作系统内核 execve() ──► 动态链接器 ld-linux.so ──► 程序的入口点 _start
                                                            │
                                                            ▼
                                                     __libc_start_main()
                                                            │
                                                            ▼
                                        遍历 ELF 的 .init_array 节中的所有函数指针并依次调用！
                                        (GlobalInitializer 的构造函数在此刻执行！)
                                                            │
                                                            ▼
                                                       调用 main()
                                                            │
                                                            ▼
                                                       main() 退出
                                                            │
                                                            ▼
                                        遍历 ELF 的 .fini_array 依次调用全局对象的析构函数！
```

---

### 5.2 静态初始化顺序与 Meyers' Singleton

* **危险陷阱**：C++ 标准规定，**同一个 `.cpp` 内部的全局对象按定义顺序依次构造；但不同 `.cpp` 之间的全局对象构造顺序完全是未定义的（Undefined）！**
* 如果 `fileA.cpp` 中的全局对象在构造时依赖 `fileB.cpp` 中的全局对象，极易引发访问未初始化内存的崩溃。

#### 工业级解决方案：Meyers' Singleton（单例局域静态化）
```cpp
class CompilerContext {
public:
    static CompilerContext& getInstance() {
        // 局部静态变量：C++11 保证其在第一次调用时线程安全地初始化 (Magic Static)
        static CompilerContext instance;
        return instance;
    }
};
```
* **原理解析**：将全局变量转化为函数内部的局部 `static` 变量，将初始化时机从 `main` 前的不确定阶段**推迟到第一次函数调用时（On-first-use）**，彻底消除跨编译单元初始化死锁。

---

## 6. 静态链接与动态链接

### 6.1 静态库单遍扫描算法

静态库（Archive，`.a`）实质上是**多个 `.o` 文件的无序压缩包（通过 `ar` 工具打包）**。

#### 链接器的单遍扫描算法（Single-pass Scanning）：
链接器在解析命令行时，从左向右扫描，维护三个集合：
1. **$E$（Executable Object Set）**：最终要合并的目标文件集合；
2. **$U$（Undefined Symbol Set）**：当前已被引用但尚未找到定义的符号集合；
3. **$D$（Defined Symbol Set）**：当前已经找到定义的符号集合。

#### 命令行顺序陷阱（Order Sensitivity）：
假设 `app.o` 调用了 `libA.a` 中的函数，而 `libA.a` 调用了 `libB.a` 中的函数：

```bash
# ❌ 错误写法 (链接报错 undefined reference)：
g++ libB.a libA.a app.o -o app
# 原因：扫描到 libB.a 时，U 集合为空，libB.a 中的所有 .o 文件直接被全部丢弃！
# 随后扫到 app.o 产生未决符号，但 libB 已经错过扫描，无法回溯！

# ✔️ 正确写法 (被依赖者永远放在右侧)：
g++ app.o -lA -lB -o app
```

---

### 6.2 动态链接与延迟绑定（PLT/GOT）

动态库在程序启动或运行时（`dlopen`）才加载，代码段需要被多个进程共享，因此必须编译为**位置无关代码（PIC, Position Independent Code，即 `-fPIC`）**。

* **GOT（Global Offset Table，全局偏移表）**：位于可读写的数据段（`.got`），存放外部函数或全局变量的真实运行时物理绝对地址；
* **PLT（Procedure Linkage Table，过程链接表）**：位于代码段（`.plt`），由一系列跳转指令组成。
* **延迟绑定（Lazy Binding）**：第一次调用外部函数时，通过 PLT 跳转触发动态链接器（`ld-linux.so`）解析符号并填入 GOT 表；后续调用直接从 GOT 表获取绝对地址跳转，避免了启动时解析所有未用符号的开销。
