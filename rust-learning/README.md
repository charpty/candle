# 从 C++ 到 Rust：跟着 Candle 学推理开发

这是一份面向有丰富 C++ 经验开发者的渐进式 Rust 学习笔记。我们不会先把 Rust
语法完整学一遍，而是沿着 Candle 的真实推理链路，在需要某个语言特性时再学习它。

主线源码是：

- [LLaMA 推理入口](../candle-examples/examples/llama/main.rs)
- [LLaMA 模型实现](../candle-transformers/src/models/llama.rs)
- [生成与采样](../candle-transformers/src/generation/mod.rs)
- [Tensor 实现](../candle-core/src/tensor.rs)
- [Storage 与后端分发](../candle-core/src/storage.rs)
- [Backend trait](../candle-core/src/backend.rs)
- [Layout 实现](../candle-core/src/layout.rs)
- [Device 实现](../candle-core/src/device.rs)
- [Module trait](../candle-core/src/lib.rs#L149)
- [VarBuilder 权重加载](../candle-nn/src/var_builder.rs)
- [Linear 实现](../candle-nn/src/linear.rs)
- [Embedding 实现](../candle-nn/src/embedding.rs)

## 学习包索引

- [Rust + Candle 推理链路讲解站](./html/index.html)：可直接打开的 HTML 学习站，把 CLI、
  Tensor、VarBuilder、Attention、KV Cache、Sampling、EOS 和后端分发串成一张交互链路图；
  另有 [项目架构全景图](./html/architecture.html)、[课程大纲与交付标准](./html/course-syllabus.html) 和
  [一次生成全链路拆解](./html/trace-walkthrough.html) 可作为对外交付页。
- [Candle 项目级架构讲义](./candle-project-architecture.md)：从 workspace、crate 边界、Tensor
  内部结构、后端分发、模型加载到一次 token 生成，讲清整个项目的主干。
- [Candle bug hunt 报告](./bug-hunt-report.md)：记录 138 个真实边界问题的发现、复现、修复、
  UT 和审计方法，覆盖通用 ops、conv groups、模型加载、KV cache 状态原子性、loss、BatchNorm、
  Mimi transformer、ViT 配置校验，Gemma4 vision/text/audio/multimodal 配置和空输入边界，
  以及 Qwen3-VL/PaddleOCR-VL vision/text 配置、vision grid、空输入、M-RoPE 和 glue 层调用合同。
- [本文件](./README.md)：按课程讲 Rust 和 Candle 推理主线，解释完整知识框架。
- [Rust + Candle 典型用法精讲](./typical-patterns.md)：把本文和 tour 代码里反复出现的
  `Result`、`Option`、借用、Tensor layout、`Module`、`VarBuilder`、KV Cache、Cargo feature、
  Serde、unsafe、tracing 等用法拆开讲。
- [Rust + Candle 深度专题：从类型系统到推理系统不变量](./deep-inference-systems.md)：更深入地
  串起所有权、shape、layout、Device、KV Cache、后端分发、unsafe 和服务化边界。
- [Rust + Candle 动手实验手册](./rust-candle-labs.md)：按命令输出、可恢复的小破坏、编译器错误、
  测试和源码阅读记录，把概念真正练到手上。
- [Candle 源码阅读路线图](./source-reading-playbook.md)：按一次推理请求的生命周期说明每条
  调用链该看哪些文件、追到哪里停、要验证什么。
- [Rust + Candle 学习自检表](./checkpoints.md)：按课程列出必须能解释的点、要跑的命令、
  小改动练习、常见误解和达标标准。
- [Rust + Candle 水平测评题](./rust-candle-exam.md)：100 分闭卷/半开卷考题，用来检验
  Rust 所有权、Tensor layout、模型加载、KV Cache、工程化和 unsafe 边界掌握情况。
- [Rust + Candle 水平测评题评分参考](./rust-candle-exam-grading-guide.md)：考后对照用的
  答案要点、扣分点和补课路线。

## 配套可执行源码导览

如果你更喜欢直接沿代码学习，从下面五个文件按顺序看：

1. [main.rs](../candle-examples/examples/rust-candle-tour/main.rs)：架构图、主流程、
   prefill/decode 生成循环，以及沿途出现的 Rust 所有权和借用。
2. [model.rs](../candle-examples/examples/rust-candle-tour/model.rs)：Embedding、Attention、
   Tensor shape、KV Cache、VarBuilder 和 lm_head。
3. [sampling.rs](../candle-examples/examples/rust-candle-tour/sampling.rs)：`Option`、enum、
   `match`、参数校验与单元测试。
4. [backend_walkthrough.rs](../candle-examples/examples/rust-candle-tour/backend_walkthrough.rs)：
   `Tensor::matmul` 如何经过 Storage 分发到 CPU/CUDA/Metal。
5. [layout_walkthrough.rs](../candle-examples/examples/rust-candle-tour/layout_walkthrough.rs)：
   `transpose`、`narrow` 和 `contiguous()` 如何改变 Tensor layout。

动手练习按 [Rust + Candle 动手实验手册](./rust-candle-labs.md) 走：每个实验都有运行命令、
观察点、小破坏、修复方式和达标问题。

示例使用本地生成的小权重，不需要下载模型：

```bash
cd candle
cargo run -p candle-examples \
  --example rust-candle-tour \
  -- \
  --cpu \
  --show-backend-path
```

如果想把 prefill/decode 中每个关键 Tensor 的 shape 和 dtype 打出来：

```bash
cargo run -p candle-examples \
  --example rust-candle-tour \
  -- \
  --cpu \
  --trace-shapes \
  -n 2
```

如果想观察 Tensor layout、stride 和 `contiguous()`：

```bash
cargo run -p candle-examples \
  --example rust-candle-tour \
  -- \
  --cpu \
  --show-layout-path \
  -n 0
```

如果想看 demo 模型通过 `VarBuilder` 期望加载哪些权重路径：

```bash
cargo run -p candle-examples \
  --example rust-candle-tour \
  -- \
  --cpu \
  --show-weight-paths \
  -n 0
```

如果想比较 KV cache 解码和每步全量重算：

```bash
cargo run -p candle-examples \
  --example rust-candle-tour \
  -- \
  --cpu \
  --compare-cache \
  -n 4
```

如果想观察 EOS 停止逻辑：

```bash
cargo run -p candle-examples \
  --example rust-candle-tour \
  -- \
  --cpu \
  --eos-token 22 \
  -n 8
```

这份长文档解释“为什么”，配套代码展示“实际怎么写”。后续学习优先沿代码走，遇到概念时再
回到对应课程。

## 学习方式

每一课分成四部分：

1. 从一段能运行的 Candle 代码开始。
2. 用 C++ 概念建立第一层映射。
3. 解释 Rust 真正不同的语义，特别是所有权和借用。
4. 完成一个很小的代码修改，用编译器反馈巩固概念。

不要急着记住所有语法。现阶段更重要的是看到一段 Rust 代码时，能回答三个问题：

1. 谁拥有这块数据？
2. 当前代码是在移动、共享还是借用数据？
3. 失败时错误会在哪里被处理？

## 课程进度

“文档状态”表示讲义是否已经写好，“学习状态”由我们每次学习后更新。

| 课程 | 内容 | 文档状态 | 学习状态 |
| --- | --- | --- | --- |
| 第 0 课 | Candle 工程地图与推理全景 | 已编写 | 进行中 |
| 第 1 课 | 从 LLaMA 入口学习变量、`Option`、`Result`、`match` 和借用 | 已编写 | 进行中 |
| 第 2 课 | 生成与采样：函数、表达式、enum 和随机数状态 | 已编写 | 未开始 |
| 第 3 课 | Tensor 执行系统：所有权、Storage、Layout 与后端分发 | 已编写 | 未开始 |
| 第 4 课 | 神经网络抽象：struct、`impl`、trait、泛型与动态分发 | 已编写 | 未开始 |
| 第 5 课 | 模型加载：Serde、SafeTensors、mmap 与 VarBuilder | 已编写 | 未开始 |
| 第 6 课 | LLaMA 推理：Attention、RoPE、GQA、KV Cache 与 shape | 已编写 | 未开始 |
| 第 7 课 | 工程化：Cargo feature、测试、日志、性能分析与并发 | 已编写 | 未开始 |
| 第 8 课 | 系统边界：`unsafe`、backend、CUDA/Metal 与 FFI | 已编写 | 未开始 |

---

# 第 0 课：先看懂 Candle 的工程地图

## 0.1 Cargo workspace

Candle 根目录的 [Cargo.toml](../Cargo.toml) 定义了一个 Cargo workspace。可以先把它理解成
CMake 管理的多 target 工程：

| Candle crate | 职责 | C++ 类比 |
| --- | --- | --- |
| `candle-core` | Tensor、Device、Storage、算子与自动微分 | 核心 runtime/library |
| `candle-nn` | Linear、Embedding、LayerNorm 等神经网络组件 | layer library |
| `candle-transformers` | LLaMA、Qwen、BERT 等模型实现 | model library |
| `candle-examples` | CLI、模型加载和推理样例 | examples/tools |

Rust 中一个可独立编译、发布和依赖的包叫 **crate**。`Cargo.toml` 同时承担了一部分
`CMakeLists.txt`、包管理清单和 feature 配置的职责。

我们暂时只沿着这一条调用链阅读：

```text
candle-examples/examples/llama/main.rs
        │
        ├── tokenizer / safetensors / 命令行参数
        │
        ▼
candle-transformers/src/models/llama.rs
        │
        ├── Attention / MLP / KV Cache
        │
        ▼
candle-nn
        │
        ├── Linear / Embedding / VarBuilder
        │
        ▼
candle-core
        └── Tensor / Device / Storage / CPU、CUDA、Metal 后端
```

## 0.2 第一阶段目标

第一阶段不要求你能从零实现 LLaMA。目标是：

- 能顺着 `main` 函数解释一次生成循环。
- 能看懂常见的 `Option<T>`、`Result<T>` 和 `match`。
- 能判断常见表达式发生了 move、borrow 还是 clone。
- 能完成小范围重构，并看懂 `cargo check` 的错误。

## 0.3 Candle 到底负责什么

Candle 是一个 Rust 机器学习框架。对于 LLM 推理，它主要负责：

1. 表示 Tensor 的 shape、dtype、layout、device 和底层存储。
2. 提供矩阵乘、归一化、索引、拼接、激活函数等算子。
3. 把同一个 Tensor API 分发到 CPU、CUDA 或 Metal 后端。
4. 提供 Linear、Embedding、RMSNorm 等神经网络组件。
5. 从 SafeTensors 等权重格式构建模型。
6. 实现 LLaMA、Qwen、BERT 等模型的 forward。
7. 提供 temperature、top-k、top-p 等生成策略。

它不只是一个“模型示例集合”。可以把 Candle 看成下面几层：

```text
┌─────────────────────────────────────────────────────────────┐
│ 应用层：CLI、HTTP 服务、Tokenizer、流式输出                  │
│ candle-examples                                              │
├─────────────────────────────────────────────────────────────┤
│ 模型层：LLaMA、Qwen、BERT、Whisper                           │
│ candle-transformers                                          │
├─────────────────────────────────────────────────────────────┤
│ 神经网络层：Linear、Embedding、RMSNorm、VarBuilder            │
│ candle-nn                                                    │
├─────────────────────────────────────────────────────────────┤
│ Tensor 层：Tensor、Shape、Layout、Storage、Autograd           │
│ candle-core                                                  │
├─────────────────────────────────────────────────────────────┤
│ 执行后端：CPU / CUDA / Metal / 自定义 kernel                  │
└─────────────────────────────────────────────────────────────┘
```

C++ 工程中常把这些层分别实现为 runtime、operator library、model library 和 server。
Candle 用多个 crate 保持边界，但通过 Cargo workspace 一起开发。

## 0.4 一次 LLaMA 推理的完整调用链

先不要陷入每个 API 的细节。一次推理可以分为三阶段。

### 阶段一：初始化

```text
Args::parse()
    │
    ├── 选择 Device 和 DType
    ├── 下载/定位 config.json、tokenizer.json、*.safetensors
    ├── 反序列化 LlamaConfig
    ├── 创建 Cache（RoPE cos/sin、每层 KV 槽位、mask cache）
    ├── VarBuilder::from_mmaped_safetensors(...)
    └── Llama::load(vb, &config)
```

对应代码：

- [入口初始化](../candle-examples/examples/llama/main.rs#L125)
- [Cache::new](../candle-transformers/src/models/llama.rs#L163)
- [VarBuilder mmap 构造](../candle-nn/src/var_builder.rs#L636)
- [Llama::load](../candle-transformers/src/models/llama.rs#L515)

`VarBuilder` 不是模型本身，也不是简单的文件读取器。它更像一个“带路径前缀、dtype、
device 和权重来源的参数查询器”。模型组件按名称和 shape 向它索取 Tensor。

例如第 3 层 Attention 的 Q 权重最终会形成类似：

```text
model.layers.3.self_attn.q_proj.weight
```

### 阶段二：prefill

prompt 经 tokenizer 变成 token id：

```text
"hello" -> [token_0, token_1, ...]
```

随后：

```text
Vec<u32>
    │ Tensor::new + unsqueeze
    ▼
[batch=1, seq_len]
    │ Embedding
    ▼
[1, seq_len, hidden_size]
    │ N 个 Transformer Block
    ▼
[1, seq_len, hidden_size]
    │ 只选择最后一个位置
    ▼
[1, hidden_size]
    │ lm_head
    ▼
[1, vocab_size] logits
```

第一次 forward 会处理完整 prompt，并为每个 Transformer Block 保存 K/V。这一步通常叫
**prefill**。它包含大规模矩阵乘，容易充分利用 GPU。

### 阶段三：逐 token decode

生成第一个 token 后，循环只把最新 token 送进模型：

```text
最新 token
    │ embedding
    │ 与每层已有 KV Cache 做 attention
    ▼
下一个 token 的 logits
    │ repeat penalty
    │ temperature / top-k / top-p
    ▼
采样一个 token
    │ tokenizer 增量解码
    ▼
打印文本并进入下一轮
```

这一步叫 **decode**。每轮输入序列长度是 1，但要读取之前所有 K/V。decode 往往更容易受
显存带宽、kernel launch、KV Cache 布局和 batch size 影响。

## 0.5 一个 Tensor 算子如何跑到硬件

以：

```rust
let c = a.matmul(&b)?;
```

为例，调用链是：

```text
Tensor::matmul
    │ 检查 rank、m/k/n、batch shape
    ▼
Storage::matmul
    │ 检查 dtype 和 device 一致
    │ match (Cpu, Cuda, Metal)
    ▼
CpuStorage::matmul / CudaStorage::matmul / MetalStorage::matmul
    │
    ▼
创建新的 Storage、Layout 和 Tensor
```

关键源码：

- [Tensor::matmul](../candle-core/src/tensor.rs#L1487)
- [Storage::matmul](../candle-core/src/storage.rs#L769)
- [BackendStorage trait](../candle-core/src/backend.rs#L6)

Candle 这里采用的是 eager execution：`matmul` 调用时就执行后端计算，并返回包含结果的
Tensor。推理时不是先构建完整静态图再一次执行。

Tensor 中仍会记录 `BackpropOp`，供训练时反向传播使用；这不等同于把 forward 变成一个
延迟执行图。

## 0.6 Tensor 不等于一段内存

阅读 Candle 时，先把 Tensor 拆成五个概念：

| 概念 | 作用 |
| --- | --- |
| `Storage` | 真正的数据，可能位于 CPU、CUDA 或 Metal |
| `Shape` | 各维大小，例如 `[1, 32, 128, 128]` |
| `Layout` | shape、stride、起始 offset，决定如何解释 Storage |
| `DType` | F16、BF16、F32、I64 等元素类型 |
| `Device` | 数据所在设备及后端上下文 |

因此 `transpose`、`narrow`、某些 `reshape` 可以只创建新 Layout 并共享 Storage；而
`contiguous()` 在 stride 不满足连续布局时会触发真实复制。

这与 C++ Tensor runtime 中的“buffer + tensor descriptor/view”非常接近。不同之处在于
Rust 会继续检查描述对象、共享所有权和可变访问是否安全。

## 0.7 编译期能力和运行时设备是两件事

下面两个概念不要混淆。

Cargo feature 决定二进制里是否编译某种能力：

```bash
cargo run -p candle-examples --example llama --release --features metal
cargo run -p candle-examples --example llama --release --features cuda
```

运行时 `Device` 决定本次 Tensor 实际放在哪：

```rust
Device::Cpu
Device::new_cuda(0)?
Device::new_metal(0)?
```

如果没有编译 `cuda` feature，运行时就没有真实 CUDA backend 可用。反过来，即使二进制
包含 CUDA 支持，也仍然可以通过 `--cpu` 选择 CPU。

设备选择逻辑见 [candle-examples/src/lib.rs](../candle-examples/src/lib.rs#L11)。

## 0.8 推荐阅读顺序

不要从 `candle-core` 第一行开始通读。按一次请求的生命周期阅读：

1. `candle-examples/examples/llama/main.rs`
2. `candle-transformers/src/models/llama.rs` 中的 `Llama::forward`
3. `Block::forward`
4. `CausalSelfAttention::forward`
5. `candle-nn/src/linear.rs`
6. `candle-core/src/tensor.rs` 中的 `matmul`
7. `candle-core/src/storage.rs` 中的后端分发
8. 最后再看具体 CPU/CUDA/Metal kernel

每次只向下追一层。看到 trait 或宏时先确认它在当前调用链中解决什么问题，再研究语法。

---

# 第 1 课：从 LLaMA 推理入口认识 Rust

本课阅读范围：

- [命令行参数定义](../candle-examples/examples/llama/main.rs#L60)
- [设备与 dtype 选择](../candle-examples/examples/llama/main.rs#L139)
- [prompt 与 tokenizer](../candle-examples/examples/llama/main.rs#L210)
- [token 生成循环](../candle-examples/examples/llama/main.rs#L241)

## 1.1 `let` 默认不可变

```rust
let args = Args::parse();
let device = candle_examples::device(args.cpu)?;
let mut tokens = tokenizer
    .encode(prompt, true)
    .map_err(E::msg)?
    .get_ids()
    .to_vec();
```

Rust 的局部变量默认不可变。只有需要修改绑定指向的值时才写 `mut`：

```rust
let count = 0;      // 不能再次给 count 赋值
let mut count = 0;  // 可以修改
count += 1;
```

这里的“不可变”比 C++ 的 `const` 更接近默认规则：

```cpp
const auto args = Args::parse();  // Rust 的默认感觉更接近这一行
auto count = 0;                   // C++ 默认可变
```

注意两件事：

1. `let mut tokens` 表示我们稍后会修改这个 `Vec`，例如 `tokens.push(next_token)`。
2. 绑定不可变不一定意味着对象内部永远不能变化；第 3 课会结合
   `Arc<RwLock<Storage>>` 讨论内部可变性。

## 1.2 `Option<T>`：把“可能不存在”放进类型

命令行参数中有：

```rust
top_p: Option<f64>,
top_k: Option<usize>,
prompt: Option<String>,
```

`Option<T>` 只有两个状态：

```rust
enum Option<T> {
    None,
    Some(T),
}
```

可以先类比为：

```cpp
std::optional<double> top_p;
std::optional<std::size_t> top_k;
std::optional<std::string> prompt;
```

Rust 通常不使用空指针表达“没有值”，而是让函数签名明确告诉调用者这个值可能不存在。

### 从 `Option<String>` 借用 `&str`

推理入口中：

```rust
let prompt = args.prompt
    .as_ref()
    .map_or(DEFAULT_PROMPT, |p| p.as_str());
```

类型变化如下：

```text
args.prompt             Option<String>
args.prompt.as_ref()    Option<&String>
p.as_str()              &str
prompt                  &str
```

`String` 拥有一段 UTF-8 字符数据，`&str` 只是借用其中的一段。可以用下面的 C++ 概念建立
初步映射：

| Rust | C++ 近似概念 |
| --- | --- |
| `String` | `std::string` |
| `&String` | `const std::string&` |
| `&str` | `std::string_view` |

这个类比并不完全等价。关键区别是 Rust 编译器会检查 `&str` 的生命周期，防止它指向已经
销毁或重新分配的数据。

这里调用 `as_ref()` 是因为代码只想借用 prompt。如果直接从 `Option<String>` 中取出
`String`，通常会移动所有权，之后就不能再把原来的 `Option` 当作仍然持有该字符串来使用。

闭包：

```rust
|p| p.as_str()
```

可以先理解为 C++ lambda：

```cpp
[](const std::string& p) -> std::string_view {
    return p;
}
```

## 1.3 `match`：带解构能力的穷尽分支

dtype 选择代码：

```rust
let dtype = match args.dtype.as_deref() {
    Some("f16") => DType::F16,
    Some("bf16") => DType::BF16,
    Some("f32") => DType::F32,
    Some(dtype) => bail!("Unsupported dtype {dtype}"),
    None => DType::F16,
};
```

`as_deref()` 把：

```text
Option<String> -> Option<&str>
```

这样匹配时不需要复制或重新分配字符串。

这里的 `match` 同时完成了四件事：

1. 检查 `Option` 是 `Some` 还是 `None`。
2. 在 `Some` 内继续匹配字符串内容。
3. 用 `Some(dtype)` 把未知字符串绑定到局部变量 `dtype`。
4. 产生一个 `DType`，赋值给外层变量。

Rust 的 `match` 是表达式：

```rust
let value = match condition {
    true => 1,
    false => 2,
};
```

分支末尾没有分号，表示该表达式是这个分支的值。所有能正常返回的分支必须产生兼容的类型。

Rust 还要求模式匹配是穷尽的。如果少写 `None`，编译器会直接报错。这比
`switch` 漏掉一个枚举值后依赖运行时行为更可靠。

## 1.4 `Result<T, E>` 与 `?`

入口函数签名是：

```rust
fn main() -> Result<()> {
```

这里的 `Result<()>` 来自 `anyhow`。成功时返回 `Ok(())`，失败时返回一个错误。

```rust
let device = candle_examples::device(args.cpu)?;
```

`?` 可以先展开理解为：

```rust
let device = match candle_examples::device(args.cpu) {
    Ok(value) => value,
    Err(error) => return Err(error.into()),
};
```

因此：

- 成功时，`?` 取出 `Ok` 内部的值。
- 失败时，`?` 立即从当前函数返回错误。
- 如果错误类型不同，编译器会尝试通过 `From`/`Into` 做类型转换。

与 C++ 异常相比，Rust 的错误路径会出现在函数返回类型里；与手写错误码相比，`?` 减少了
大量重复判断。

`bail!`：

```rust
Some(dtype) => bail!("Unsupported dtype {dtype}"),
```

等价于构造错误并立即 `return Err(...)`。

## 1.5 借用、切片和 non-lexical lifetime

生成循环中：

```rust
let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;
let logits = llama.forward(&input, context_index, &mut cache)?;
// ...
index_pos += ctxt.len();
// ...
tokens.push(next_token);
```

类型关系：

```text
tokens    Vec<u32>   拥有 token 缓冲区
ctxt      &[u32]     借用 tokens 中的一段连续区域
&device   &Device    不可变借用
&input    &Tensor    不可变借用
&mut cache
          &mut Cache 独占的可变借用
```

`&[u32]` 可以先类比 `std::span<const uint32_t>`：它不拥有数据，但携带起始位置和长度。

`ctxt` 最后一次使用是：

```rust
index_pos += ctxt.len();
```

之后编译器就可以结束这次借用，所以稍后允许：

```rust
tokens.push(next_token);
```

`push` 可能导致 `Vec` 重新分配。如果 `ctxt` 在 `push` 后还会使用，它就可能成为悬空切片，
Rust 编译器会拒绝这样的代码。

借用在“最后一次使用”处结束，而不是机械地持续到花括号结尾，这叫
**non-lexical lifetime（NLL）**。

### `&T` 与 `&mut T`

可以暂时使用下面的规则：

- 同一时间可以有多个 `&T`。
- 同一时间只能有一个 `&mut T`。
- `&mut T` 存在时，不能同时通过其他引用访问同一数据。

这不是单纯为了防止“写错代码”，而是 Rust 无数据竞争并发模型的基础。

`llama.forward` 对 cache 使用 `&mut cache`，表达了一个很重要的 API 事实：forward 会更新
KV Cache，而且调用期间需要独占访问它。

## 1.6 `enum`：比 C++ enum 更接近受检查的 tagged union

EOS 判断代码：

```rust
match eos_token_id {
    Some(model::LlamaEosToks::Single(eos_tok_id))
        if next_token == eos_tok_id => {
            break;
        }
    Some(model::LlamaEosToks::Multiple(ref eos_ids))
        if eos_ids.contains(&next_token) => {
            break;
        }
    _ => (),
}
```

它处理的是“可能没有 EOS 配置；如果有，配置又可能是单个或多个 token id”。

可以粗略类比为：

```cpp
std::optional<std::variant<SingleEos, MultipleEos>>
```

但 Rust 可以用一个模式直接完成：

- 解构外层 `Option`。
- 判断内层 enum variant。
- 取出 variant 携带的数据。
- 使用 `if` guard 增加条件。

`ref eos_ids` 表示借用 variant 里的集合，而不是把集合移动出来。后续我们会专门比较：

```rust
value
&value
ref binding
```

## 1.7 Candle Tensor 中的所有权预告

核心定义：

```rust
#[derive(Clone)]
pub struct Tensor(Arc<Tensor_>);
```

`Tensor_` 内部包含：

```rust
storage: Arc<RwLock<Storage>>,
layout: Layout,
op: BackpropOp,
is_variable: bool,
dtype: DType,
device: Device,
```

第一层 C++ 映射：

| Rust | C++ 近似概念 |
| --- | --- |
| `Tensor` | 持有实现对象的 value wrapper |
| `Arc<T>` | 原子引用计数的 `std::shared_ptr<T>` |
| `RwLock<T>` | `std::shared_mutex` 与被保护对象的组合 |
| `Clone` | 显式复制；具体是深拷贝还是共享取决于类型 |
| `enum Device` | `std::variant<Cpu, Cuda, Metal>` |
| `trait Module` | interface、template constraint 与 duck typing 的组合 |

因此：

```rust
let tensor2 = tensor1.clone();
```

对 Candle `Tensor` 而言通常只是增加 `Arc` 引用计数，不会复制整块 Tensor 数据。但不能把
“Rust 的 `clone` 都很便宜”当成通用规律；是否昂贵由具体类型的 `Clone` 实现决定。

我们在第 3 课会直接阅读 [Tensor 实现](../candle-core/src/tensor.rs)，讨论为什么 Candle
同时给 Tensor 和 Storage 使用引用计数，以及这对 view、reshape 和计算图有什么影响。

## 1.8 本课练习

暂时不要大改推理代码。先回答下面五个问题：

1. 为什么 `prompt` 的类型适合用 `&str`，而不是再创建一个 `String`？
2. `args.prompt.as_ref()` 如果去掉，可能会发生什么？
3. `&tokens[a..]` 是否复制了 token 数据？
4. 为什么 `llama.forward` 接收 `&mut cache` 而不是 `&cache`？
5. `Tensor::clone()` 和 `Vec::clone()` 的成本为什么可能完全不同？

### 编程练习：抽取采样策略

把 [采样策略构造代码](../candle-examples/examples/llama/main.rs#L226) 抽成一个函数：

```rust
fn build_sampling(
    temperature: f64,
    top_k: Option<usize>,
    top_p: Option<f64>,
) -> Sampling {
    todo!()
}
```

要求：

- `temperature <= 0.0` 时返回 `Sampling::ArgMax`。
- 其他情况用 `match (top_k, top_p)` 选择策略。
- 函数最后不写显式 `return`。
- 不使用 `clone`。
- 先自己写；编译器报错也是练习材料。

完成后可以运行：

```bash
cd candle
cargo check -p candle-examples --example llama
```

### 本课完成标准

如果你能不看答案解释下面这段代码，本课就算完成：

```rust
let sampling = match (top_k, top_p) {
    (None, None) => Sampling::All { temperature },
    (Some(k), None) => Sampling::TopK { k, temperature },
    (None, Some(p)) => Sampling::TopP { p, temperature },
    (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
};
```

需要说明：

- `top_k` 和 `top_p` 的类型。
- `match` 的输入为什么是元组。
- `Some(k)` 做了什么。
- 为什么四个分支缺少任何一个都会有风险。
- 为什么每个分支末尾都没有分号。

---

# 第 2 课：生成与采样

本课把 Rust 的函数、表达式、enum、所有权和可变状态放进真实生成逻辑里。

阅读范围：

- [Sampling 与 LogitsProcessor](../candle-transformers/src/generation/mod.rs)
- [LLaMA 示例中的采样配置](../candle-examples/examples/llama/main.rs#L226)
- [生成循环](../candle-examples/examples/llama/main.rs#L241)

## 2.1 从 logits 到 token

模型最后输出：

```text
logits: [batch, vocab_size]
```

示例中的 batch 是 1，随后 `squeeze(0)` 得到：

```text
logits: [vocab_size]
```

采样过程可以概括为：

```text
logits
  │ 可选 repeat penalty
  │ 除以 temperature
  │ softmax
  │ 可选 top-k / top-p 截断
  │ 带权随机采样或 argmax
  ▼
next_token: u32
```

`Sampling` 用 enum 表达互斥策略：

```rust
pub enum Sampling {
    ArgMax,
    All { temperature: f64 },
    TopK { k: usize, temperature: f64 },
    TopP { p: f64, temperature: f64 },
    TopKThenTopP { k: usize, p: f64, temperature: f64 },
    GumbelSoftmax { temperature: f64 },
}
```

这比用多个 bool 和 nullable 字段更不容易产生非法状态。例如 `ArgMax` 根本不携带
temperature；`TopK` 必须携带 `k`。

C++ 中可以用 `std::variant` 表达同样设计，但 Rust enum、模式匹配和 exhaustiveness 是
语言的一等能力。

## 2.2 抽取 `build_sampling`

我们要把 `main` 中的策略选择抽成：

```rust
fn build_sampling(
    temperature: f64,
    top_k: Option<usize>,
    top_p: Option<f64>,
) -> Sampling {
    if temperature <= 0.0 {
        Sampling::ArgMax
    } else {
        match (top_k, top_p) {
            (None, None) => Sampling::All { temperature },
            (Some(k), None) => Sampling::TopK { k, temperature },
            (None, Some(p)) => Sampling::TopP { p, temperature },
            (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
        }
    }
}
```

这段代码有几个 Rust 要点。

### 函数最后一个表达式就是返回值

```rust
fn answer() -> i32 {
    42
}
```

`42` 后面没有分号，因此它是函数返回值。写成 `42;` 后类型会变成 `()`，类似 C++ 的
`void`。

`if` 和 `match` 都是表达式，所以可以直接作为返回值。正常返回的每条分支必须得到同一类
型，这里都是 `Sampling`。

### 元组模式同时处理两个 Option

```rust
match (top_k, top_p)
```

输入类型是：

```text
(Option<usize>, Option<f64>)
```

两个 Option 一共有四种组合，所以四个分支刚好穷尽所有情况。`Some(k)` 和 `Some(p)`
在匹配的同时取出内部值。

### 这里的 Option 会不会被 move

会。`match (top_k, top_p)` 按值消费这两个变量。

但 `usize` 和 `f64` 实现了 `Copy`，因此 `Option<usize>` 和 `Option<f64>` 也实现了
`Copy`。对这类值，“move”会退化成按位复制，原变量之后仍可使用。

换成 `Option<String>` 后就不同：

```rust
let name: Option<String> = Some("llama".to_string());
match name {
    Some(value) => println!("{value}"),
    None => {}
}
// 再使用 name 通常会报 use of moved value
```

如果只想借用，可以匹配：

```rust
match &name {
    Some(value) => println!("{value}"),
    None => {}
}
```

## 2.3 `LogitsProcessor` 为什么需要 `&mut self`

定义：

```rust
pub struct LogitsProcessor {
    rng: rand::rngs::StdRng,
    sampling: Sampling,
}
```

采样会推进随机数生成器内部状态，所以：

```rust
pub fn sample(&mut self, logits: &Tensor) -> Result<u32>
```

必须接收 `&mut self`。这和 KV Cache 一样，函数签名直接告诉调用者：“调用会修改持久
状态，并且需要独占访问”。

C++ 方法可能写成：

```cpp
uint32_t sample(const Tensor& logits); // rng 可能被 mutable 或隐式修改
```

Rust 不允许这种不透明的可变性，除非显式使用 `Mutex`、`RefCell` 等内部可变性容器。

## 2.4 采样发生在哪个设备

非 ArgMax 策略中，代码会：

```rust
let prs = candle_nn::ops::softmax_last_dim(&logits)?;
let mut prs = prs.to_vec1()?;
```

softmax 是 Tensor 算子，可以在模型所在设备执行；`to_vec1()` 则把结果转成 Rust
`Vec<f32>`，供 `rand` 在 CPU 上完成 weighted sampling。

这意味着逐 token decode 中可能存在一次 device-to-host 数据传输和同步。学习框架时要区分：

- API 看起来只返回一个 `Vec`。
- 系统层面可能包含 GPU 同步与传输。

ArgMax 路径也最终需要把一个标量 token id 读回 host，但传输的数据量更小。

这正是为什么学习 Candle 不能只学 Rust 语法：所有权正确不等于性能最优。

## 2.5 闭包 `FnOnce`

`sample_f` 接收：

```rust
f: impl FnOnce(&mut [f32])
```

它允许调用者在真正采样前原地修改概率数组，而且承诺最多调用一次。

Rust 的闭包 trait 有三类：

| trait | 闭包如何使用捕获值 | 调用次数直觉 |
| --- | --- | --- |
| `Fn` | 只借用 | 可重复 |
| `FnMut` | 可变借用 | 可重复但需要可变闭包 |
| `FnOnce` | 可能消费捕获值 | 至少支持调用一次 |

这里 `FnOnce` 给调用者最大自由度，同时实现只需要调用一次。

## 2.6 本课练习

### 练习 A：完成重构

把 `build_sampling` 加到 LLaMA 示例，并替换 `main` 中原有分支。检查：

```bash
cd candle
cargo check -p candle-examples --example llama
```

### 练习 B：验证边界

回答：

1. `temperature == 0.0` 为什么选择 ArgMax？
2. `top_p <= 0.0` 或 `top_p >= 1.0` 时当前实现如何处理？
3. `top_k >= vocab_size` 时是否会崩溃？
4. 相同 seed 和相同 logits 是否应该得到可复现结果？

### 练习 C：设计合法参数

尝试增加：

```rust
fn validate_sampling(
    temperature: f64,
    top_k: Option<usize>,
    top_p: Option<f64>,
) -> anyhow::Result<()>
```

要求拒绝：

- 负 temperature。
- `top_k == Some(0)`。
- 不在 `(0, 1]` 内的 top-p。

这一练习会继续训练 `Option`、`Result`、`bail!` 和范围判断。

---

# 第 3 课：Tensor 执行系统

本课同时回答两个问题：

1. Candle 中一个 Tensor 在内存里是什么？
2. 调用一个算子时，代码如何走到 CPU/CUDA/Metal？

阅读范围：

- [Tensor 定义](../candle-core/src/tensor.rs#L23)
- [Layout](../candle-core/src/layout.rs)
- [Storage](../candle-core/src/storage.rs)
- [Backend trait](../candle-core/src/backend.rs)
- [Tensor::matmul](../candle-core/src/tensor.rs#L1487)

## 3.1 `Tensor` 是共享句柄

核心定义：

```rust
pub struct Tensor(Arc<Tensor_>);

pub struct Tensor_ {
    id: TensorId,
    storage: Arc<RwLock<Storage>>,
    layout: Layout,
    op: BackpropOp,
    is_variable: bool,
    dtype: DType,
    device: Device,
}
```

外层 `Tensor` 是 tuple struct，只包含一个 `Arc<Tensor_>`。可以类比：

```cpp
class Tensor {
    std::shared_ptr<TensorImpl> impl_;
};
```

但是 `Arc` 只是“原子引用计数共享所有权”，不是“自动允许任何线程使用”。类型能否跨线程还
受 `Send` 和 `Sync` 约束。

`#[derive(Clone)]` 为 Tensor 生成 `Clone` 实现。克隆 Tensor 时主要增加外层 `Arc` 的
引用计数。

## 3.2 为什么 Storage 又有一个 Arc

Tensor 和 Storage 分别引用计数，使多个不同的 Tensor view 可以共享同一块数据：

```text
Tensor A ── Layout [2, 3], stride [3, 1] ─┐
                                          ├── same Storage
Tensor B ── Layout [3, 2], stride [1, 3] ─┘
```

这让 `transpose`、`narrow`、broadcast 和部分 reshape 不必复制数据。

如果只有 `Arc<Tensor_>` 而 Storage 不单独共享，那么每个不同 Layout 都很难作为独立 Tensor
存在并共享底层 buffer。

## 3.3 `Layout` 决定数据如何被观察

`Layout` 保存：

```rust
pub struct Layout {
    shape: Shape,
    stride: Vec<usize>,
    start_offset: usize,
}
```

例如连续的 `[2, 3]`：

```text
shape        [2, 3]
stride       [3, 1]
start_offset 0
```

转置后可能变成：

```text
shape        [3, 2]
stride       [1, 3]
start_offset 0
```

数据没有移动，只是索引解释发生变化。

### `reshape` 是否复制

[Tensor::reshape](../candle-core/src/tensor.rs#L2499) 的行为：

- 输入 contiguous：共享 Storage，只创建新 Layout。
- 输入不 contiguous：分配新 Storage 并复制成 contiguous。

因此不能只看 API 名判断成本。相同的 `reshape` 调用可能是 O(1)，也可能是 O(n)。

### `contiguous()` 是否复制

[Tensor::contiguous](../candle-core/src/tensor.rs#L2464)：

- 已连续：返回 `self.clone()`，基本只是共享句柄。
- 不连续：分配新 Storage，并按原 stride 复制。

LLaMA Attention 中多次出现 `.contiguous()?`，通常是因为后续 matmul/kernel 对布局有要求。

## 3.4 `RwLock<Storage>` 与内部可变性

Rust 通常要求修改对象时持有 `&mut T`。但 Tensor 的底层 Storage 被多个 view 共享，有些
内部操作又需要受控修改，因此 Candle 使用：

```rust
Arc<RwLock<Storage>>
```

`RwLock` 在运行时执行读写互斥：

- 多个 reader 可以同时存在。
- writer 必须独占。

这是内部可变性的一种形式。外层拿到 `&Tensor`，内部仍可在锁的保护下修改 Storage。

不要把 `RwLock` 理解成“绕过借用检查”。它把一部分检查从编译期移动到运行时，并增加锁
开销、阻塞和潜在死锁等工程问题。

## 3.5 从 `Tensor::matmul` 到具体后端

`Tensor::matmul` 先处理框架级逻辑：

1. 检查两个 Tensor rank 相同且至少为 2。
2. 提取 batch、m、n、k。
3. 检查 shape 是否兼容。
4. 调用 `Storage::matmul`。
5. 创建结果 Tensor，并记录 `BackpropOp::Matmul`。

`Storage` 是：

```rust
pub enum Storage {
    Cpu(CpuStorage),
    Cuda(CudaStorage),
    Metal(MetalStorage),
}
```

`Storage::matmul` 先检查 device/dtype，然后模式匹配：

```rust
match (self, rhs) {
    (Self::Cpu(lhs), Self::Cpu(rhs)) => { /* CPU */ }
    (Self::Cuda(lhs), Self::Cuda(rhs)) => { /* CUDA */ }
    (Self::Metal(lhs), Self::Metal(rhs)) => { /* Metal */ }
    _ => { /* device mismatch */ }
}
```

这是运行时分发。被编译进二进制的后端集合则由 Cargo feature 在编译期决定。

## 3.6 `BackendStorage` 与关联类型

trait 的简化形式：

```rust
pub trait BackendStorage: Sized {
    type Device: BackendDevice;

    fn dtype(&self) -> DType;
    fn device(&self) -> &Self::Device;
    fn matmul(...) -> Result<Self>;
}
```

`type Device` 是关联类型，表示每种 Storage 实现都有一个确定的 Device 类型：

```text
CpuStorage   -> CpuDevice
CudaStorage  -> CudaDevice
MetalStorage -> MetalDevice
```

C++ 可以用抽象基类、CRTP 或 concept 表达相似约束。Rust trait 同时支持：

- 泛型静态分发。
- `dyn Trait` 动态分发。
- 关联类型描述实现间的类型关系。

## 3.7 宏在 Tensor 算子中的作用

Candle 用宏生成大量结构相同的一元/二元算子：

```rust
macro_rules! unary_op {
    ($fn_name:ident, $op_name:ident) => {
        pub fn $fn_name(&self) -> Result<Self> {
            // 后端执行、记录 op、构造 Tensor
        }
    };
}
```

Rust 宏不是 C++ 预处理器的纯文本替换。`macro_rules!` 基于 token tree 和语法模式工作，
仍然经过 Rust 解析和类型检查。

初学时不需要马上会写复杂宏。先学会：

1. 看到一个找不到函数体的方法时，搜索宏调用。
2. 展开思考它生成的方法签名。
3. 把宏参数代入模板阅读。

## 3.8 `Deref` 为什么让 Tensor 像 Tensor_

Tensor 实现：

```rust
impl std::ops::Deref for Tensor {
    type Target = Tensor_;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}
```

所以调用 `tensor.shape()` 时，方法查找可以从 `Tensor` 自动解引用到 `Tensor_`。

它有点像 C++ 智能指针的 `operator->`，但 Rust 的 deref coercion 还会参与引用类型转换。

不要为所有 wrapper 随意实现 `Deref`；它适合表达“这个类型透明地表现得像目标类型”的
关系。

## 3.9 本课练习

### 练习 A：手动跟踪 matmul

从：

```rust
let a = Tensor::new(&[[1f32, 2.], [3., 4.]], &Device::Cpu)?;
let b = Tensor::new(&[[5f32, 6.], [7., 8.]], &Device::Cpu)?;
let c = a.matmul(&b)?;
```

依次回答：

1. `a` 的 Shape、Layout、DType 和 Device 是什么？
2. `Tensor::matmul` 算出的 `(batching, m, n, k)` 是什么？
3. `Storage::matmul` 会进入哪个 match 分支？
4. `c` 是否与 `a` 共享 Storage？

### 练习 B：判断是否复制

判断下列操作一定复制、可能复制还是通常只创建 view：

```rust
let b = a.clone();
let b = a.transpose(0, 1)?;
let b = a.reshape((1, 4))?;
let b = a.transpose(0, 1)?.contiguous()?;
let b = a.to_dtype(DType::F16)?;
```

### 练习 C：观察 stride

运行配套示例：

```bash
cargo run -p candle-examples \
  --example rust-candle-tour \
  -- \
  --cpu \
  --show-layout-path \
  -n 0
```

观察：

- `base` 的 stride 是标准 row-major stride。
- `transpose(1,2)` 改变 dims 和 stride，通常不复制 Storage。
- `narrow(dim=1)` 改变 start offset，view 可能不再是 contiguous。
- `contiguous()` 把非连续 view 打包成新的 row-major layout。

验证数据不复制也能得到不同的二维观察方式。

---

# 第 4 课：神经网络抽象

本课学习 Candle 如何把 Tensor 算子组装成 Layer，再组装成完整模型。

阅读范围：

- [Module trait](../candle-core/src/lib.rs#L149)
- [Linear](../candle-nn/src/linear.rs)
- [Embedding](../candle-nn/src/embedding.rs)
- [LLaMA Block 与模型结构](../candle-transformers/src/models/llama.rs#L395)

## 4.1 `struct` 保存状态，`impl` 定义行为

Linear：

```rust
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }
}
```

Rust 没有 C++ class 关键字。数据放在 `struct`，固有方法写在 `impl Type` 中。

字段默认是模块私有的。外部代码通过 `weight()`、`bias()` 等方法访问。这类似 C++ private
字段加 getter，但可见性以 module/crate 为边界，而不只是 class。

## 4.2 `Module` trait

```rust
pub trait Module {
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}
```

Linear 实现它：

```rust
impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x @ weight.T + bias
    }
}
```

trait 定义能力，不要求继承数据布局。它同时接近：

- C++ 纯虚接口。
- C++20 concept。
- 一组可被泛型约束的方法。

区别取决于使用方式：

```rust
fn run<M: Module>(m: &M, x: &Tensor)        // 静态分发，类似 template
fn run(m: &dyn Module, x: &Tensor)          // 动态分发，类似虚函数
```

## 4.3 Candle 为什么让闭包也实现 Module

源码中：

```rust
impl<T: Fn(&Tensor) -> Result<Tensor>> Module for T {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self(xs)
    }
}
```

任何满足签名的闭包都能当作 Module：

```rust
let double = |x: &Tensor| x.affine(2.0, 0.0);
let y = double.forward(&x)?;
```

这体现了 trait 的组合能力：不需要让闭包继承某个基类。

## 4.4 泛型 Shape API

常见签名：

```rust
pub fn reshape<S: ShapeWithOneHole>(&self, s: S) -> Result<Tensor>
pub fn zeros<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self>
```

所以调用者可以传：

```rust
(2, 3)
[2, 3]
vec![2, 3]
```

只要类型实现相应 trait 即可。

C++ 可用 template 和 concept 表达，但 Rust 的 trait bound 会出现在统一的类型系统和错误
信息里。阅读时：

```rust
S: Into<Shape>
```

读作“任意能够转换成 Shape 的类型 S”。

## 4.5 Linear 如何处理不同 rank

[Linear::forward](../candle-nn/src/linear.rs#L42) 不只是机械调用 matmul。它检查输入 shape：

- 4D contiguous：先展平 batch 维，走普通 matmul，再 reshape 回去。
- 3D contiguous：同样展平，避免 broadcasted matmul。
- 其他情况：直接 matmul。
- 有 bias：使用 `broadcast_add`。

原因是普通 matmul 在 CPU/CUDA backend 上通常比广播 matmul 更快。

这里可以看到框架层优化的典型方式：

```text
语义等价变换
    +
根据 Layout 选择更适合后端的算子路径
```

## 4.6 Embedding 本质是 index_select

Embedding forward：

```rust
let mut final_dims = indexes.dims().to_vec();
final_dims.push(self.hidden_size);
let indexes = indexes.flatten_all()?;
let values = self.embeddings.index_select(&indexes, 0)?;
values.reshape(final_dims)
```

输入：

```text
[batch, seq]
```

权重：

```text
[vocab_size, hidden_size]
```

输出：

```text
[batch, seq, hidden_size]
```

Embedding 并不是神秘算子：它按 token id 从词表矩阵选择行，再恢复 batch/sequence shape。

## 4.7 LLaMA 如何用组合代替继承

模型结构：

```rust
pub struct Llama {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: Linear,
}

struct Block {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    mlp: Mlp,
}
```

这是一棵值组合树，没有 `BaseLayer` 继承体系。每个组件持有自己的权重 Tensor，并在
`forward` 中显式组合。

推理代码的结构几乎直接反映数学结构：

```rust
let residual = x;
let x = self.rms_1.forward(x)?;
let x = (self.attn.forward(&x, index_pos, block_idx, cache)? + residual)?;
let residual = &x;
let x = (self.mlp.forward(&self.rms_2.forward(&x)?)? + residual)?;
```

## 4.8 动态分发实例：VarBuilder backend

`VarBuilder` 的常用别名：

```rust
pub type VarBuilder<'a> =
    VarBuilderArgs<'a, Box<dyn SimpleBackend + 'a>>;
```

这里 `dyn SimpleBackend` 是 trait object，通过 vtable 动态调用不同权重来源：

- mmap SafeTensors
- 内存中的 HashMap
- NPZ
- PyTorch PTH
- 训练用 VarMap

模型加载代码不需要知道权重究竟来自哪种文件。

这和 C++ 的 `std::unique_ptr<IWeightProvider>` 很像。`Box` 提供独占堆所有权，`dyn Trait`
提供动态分发。

## 4.9 生命周期 `'a` 在这里解决什么

```rust
pub type VarBuilder<'a> =
    VarBuilderArgs<'a, Box<dyn SimpleBackend + 'a>>;
```

`'a` 表示 backend 可能借用了生命周期为 `'a` 的外部数据。例如从字节 slice 构造
SafeTensors 时，builder 不能活得比原始 slice 更久。

从 mmap 构造时能得到拥有映射资源的 backend，因此常见返回值可以不依赖调用栈上的临时
buffer。

初学阶段看到生命周期先问：“返回对象里是否保存了某个引用？”不要一上来尝试手算所有
lifetime。

## 4.10 本课练习

### 练习 A：实现一个 Module

实现一个无参数模块：

```rust
#[derive(Debug, Clone, Copy)]
struct Scale {
    factor: f64,
}

impl candle::Module for Scale {
    fn forward(&self, x: &Tensor) -> candle::Result<Tensor> {
        todo!()
    }
}
```

调用 Tensor 的标量乘法或 `affine` 完成实现。

### 练习 B：组合模块

实现：

```text
Linear -> SiLU -> Linear
```

要求：

- 用 struct 持有两个 Linear。
- `forward` 只借用输入。
- 用 `?` 传播每一步错误。
- 不复制输入 Tensor 数据。

### 练习 C：静态还是动态分发

分别解释为什么：

- `Linear::forward` 适合静态分发。
- VarBuilder 的权重来源适合动态分发。

提示：考虑类型是否在编译期已知、是否需要异构集合、代码体积和间接调用成本。

---

# 第 5 课：模型与权重加载

本课解释一个 Hugging Face checkpoint 如何变成 Candle 中可执行的 LLaMA 对象。

阅读范围：

- [示例中的文件加载](../candle-examples/examples/llama/main.rs#L147)
- [LlamaConfig](../candle-transformers/src/models/llama.rs#L38)
- [VarBuilder](../candle-nn/src/var_builder.rs)
- [Llama::load](../candle-transformers/src/models/llama.rs#L515)

## 5.1 模型初始化需要哪些文件

典型模型仓库至少包含：

```text
config.json
tokenizer.json
model.safetensors
```

大模型可能把权重拆成多个分片：

```text
model.safetensors.index.json
model-00001-of-00004.safetensors
model-00002-of-00004.safetensors
...
```

各文件职责：

| 文件 | 内容 |
| --- | --- |
| `config.json` | hidden size、层数、head 数、RoPE 参数、EOS 等 |
| `tokenizer.json` | 字符串与 token id 之间的规则 |
| `*.safetensors` | 按名称保存的参数 Tensor |
| index JSON | 参数名到权重分片文件的路由 |

模型结构代码和权重文件必须在名称、shape 和 dtype 上一致。

## 5.2 Serde：从 JSON 到强类型配置

Llama 配置：

```rust
#[derive(Debug, Clone, serde::Deserialize)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    // ...
}
```

读取：

```rust
let config: LlamaConfig =
    serde_json::from_slice(&std::fs::read(config_filename)?)?;
```

`#[derive(Deserialize)]` 让 Serde 根据字段生成反序列化实现。与手写 JSON parser 相比：

- 缺失的必需字段会产生错误。
- 字段类型不匹配会产生错误。
- `Option<T>` 可以表达可选字段。
- `#[serde(default = "...")]` 可以声明默认值。
- `#[serde(untagged)]` 可以让 EOS 同时接受单值或数组。

## 5.3 `into_config(self)` 表示消费

```rust
pub fn into_config(self, use_flash_attn: bool) -> Config
```

方法接收 `self`，而不是 `&self`，表示它消费原始 `LlamaConfig` 并把字段移动到内部 `Config`。

命名习惯：

| 命名 | 常见语义 |
| --- | --- |
| `as_*` | 借用到另一种视图 |
| `to_*` | 通常创建/复制一个新值 |
| `into_*` | 通常消费 self 并转换 |

这不是编译器强制的命名规则，但 Rust 生态广泛遵循。

这里将 JSON 兼容层与运行时配置层分开：

- `LlamaConfig` 忠实表达外部 JSON，字段可能是 `Option`。
- `Config` 填好默认值，forward 时不必重复处理缺省情况。

## 5.4 mmap SafeTensors

入口使用：

```rust
let vb = unsafe {
    VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)?
};
```

mmap 的目标是把文件映射进进程地址空间，避免先把整个大权重文件读进一个额外 buffer。
但这不表示 GPU 权重“零拷贝”：

- mmap 提供的是 host 虚拟地址映射。
- 真正创建 CUDA/Metal Tensor 时通常仍需要传输到设备。
- dtype 转换也可能分配新存储。

`unsafe` 来自文件映射的系统级约束：映射存活期间，底层文件不能被外部不安全地截断或修改，
否则 Rust 无法仅靠类型系统保证内存访问有效。

`unsafe` 块的意义是：“程序员在这里证明额外不变量成立”，不是关闭所有安全检查。

## 5.5 VarBuilder 的职责

`VarBuilder` 保存：

```text
backend       权重从哪里来
path          当前参数名前缀
dtype         目标 dtype
device        目标设备
```

核心 API：

```rust
vb.pp("model")
vb.pp("layers")
vb.get((out_dim, in_dim), "weight")
```

`pp` 是 `push_prefix` 的缩写，语义类似进入命名目录，但它不修改原 builder，而是返回一个
新 builder：

```rust
pub fn push_prefix<S: ToString>(&self, s: S) -> Self {
    let mut path = self.path.clone();
    path.push(s.to_string());
    Self {
        data: self.data.clone(),
        path,
        // ...
    }
}
```

backend 使用 `Arc` 共享，path 被复制。这种设计让父 builder 可继续使用，也避免可变的全局
“当前路径”。

## 5.6 一条权重名称如何生成

LLaMA 加载第 `i` 层：

```rust
Block::load(vb.pp(format!("model.layers.{i}")), cfg)
```

Block 再加载 Attention：

```rust
CausalSelfAttention::load(vb.pp("self_attn"), cfg)
```

Attention 加载 Q projection：

```rust
linear(size_in, size_q, vb.pp("q_proj"))
```

Linear 最终读取：

```rust
vb.get((out_dim, in_dim), "weight")
```

拼接得到：

```text
model.layers.{i}.self_attn.q_proj.weight
```

同时 VarBuilder 会检查权重 shape。名称存在但 shape 不符，不会静默继续，而会返回
`UnexpectedShape`。

## 5.7 用 iterator 构造所有 Block

模型代码：

```rust
let blocks: Vec<_> = (0..cfg.num_hidden_layers)
    .map(|i| Block::load(vb.pp(format!("model.layers.{i}")), cfg).unwrap())
    .collect();
```

这里涉及：

- `0..n`：Range iterator。
- `map`：把层号转换成 Block。
- `|i| ...`：闭包。
- `collect::<Vec<_>>()`：收集成 Vec，元素类型由编译器推导。

值得注意的是当前代码在闭包中用了 `unwrap()`。权重缺失时它会 panic，而不是把错误优雅返回。
工程代码中更稳妥的形态通常是：

```rust
let blocks = (0..cfg.num_hidden_layers)
    .map(|i| Block::load(vb.pp(format!("model.layers.{i}")), cfg))
    .collect::<Result<Vec<_>>>()?;
```

`Result` 也实现了 `FromIterator`：所有项成功就得到 `Ok(Vec<_>)`，遇到第一个错误就返回
`Err`。

这是一个很好的 Rust 工程化练习。

## 5.8 tied embedding 为什么使用 clone

当 `tie_word_embeddings` 开启时：

```rust
Linear::from_weights(wte.embeddings().clone(), None)
```

输入 embedding 和输出 lm_head 共享同一份权重。在 Candle 中 clone Tensor 共享底层
引用计数资源，正适合表达 tied weights。

如果这里调用的是某种物理深拷贝 API，就失去了 tied weight 的内存优势。

## 5.9 权重加载的错误边界

初始化可能失败的地方很多：

- 网络/缓存中找不到文件。
- JSON 格式或字段错误。
- SafeTensors 损坏。
- 权重名称不匹配。
- shape 不匹配。
- dtype 不受设备支持。
- GPU 显存不足。

示例使用 `anyhow::Result`，适合应用入口汇总不同库的错误。框架内部则使用
`candle::Result<T>` 和结构化 `candle::Error`。

一般原则：

- 库层尽量返回具体、可匹配的错误类型。
- 应用层可以用 `anyhow` 添加上下文并统一向上传播。
- 可恢复错误不要 `unwrap()`。
- 真正违反内部不变量时才考虑 panic。

## 5.10 本课练习

### 练习 A：移除加载路径中的 `unwrap`

把 Block 收集改为：

```rust
collect::<Result<Vec<_>>>()?
```

观察编译器如何从 iterator item `Result<Block>` 推导最终类型。

### 练习 B：手动画参数路径

写出这些参数的完整名称：

1. 第 0 层 Attention 的 K projection。
2. 第 7 层 MLP 的 down projection。
3. 模型最终 RMSNorm。
4. token embedding。

### 练习 C：比较三种 builder

解释适用场景：

```rust
VarBuilder::from_mmaped_safetensors(...)
VarBuilder::from_buffered_safetensors(...)
VarBuilder::from_varmap(...)
```

提示：考虑推理、内存 buffer、训练初始化和资源生命周期。

---

# 第 6 课：LLaMA 推理全流程

本课把模型数学、Tensor shape、KV Cache 和 Rust 借用放在同一张图里。

阅读范围：

- [Cache](../candle-transformers/src/models/llama.rs#L145)
- [Attention forward](../candle-transformers/src/models/llama.rs#L270)
- [Block forward](../candle-transformers/src/models/llama.rs#L435)
- [Llama forward](../candle-transformers/src/models/llama.rs#L503)
- [示例生成循环](../candle-examples/examples/llama/main.rs#L241)

## 6.1 先统一符号

| 符号 | 含义 |
| --- | --- |
| `B` | batch size |
| `S` | 当前输入序列长度 |
| `T` | 包含历史 cache 后的总序列长度 |
| `H` | hidden size |
| `Nq` | query attention head 数 |
| `Nkv` | key/value head 数 |
| `D` | head dim，通常 `H / Nq` |
| `V` | vocabulary size |

LLaMA 示例通常 `B = 1`。

## 6.2 从 token id 到 embedding

入口构造：

```rust
let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;
```

shape：

```text
ctxt                         [S]
Tensor::new(ctxt)            [S]
unsqueeze(0)                 [1, S]
```

Embedding：

```text
token ids                    [B, S]
embedding weight             [V, H]
index_select + reshape
output                       [B, S, H]
```

这里 token Tensor 的 dtype 是由 `u32` slice 推导出来的整数类型；Embedding 输出 dtype
来自模型权重。

## 6.3 一个 Transformer Block

Block 的计算：

```text
x
│
├────────────── residual 1 ──────────────┐
│ RMSNorm                                │
│ Self Attention                        │
└────────────────────────────────────── add
                                          │
                 ┌──── residual 2 ────────┤
                 │ RMSNorm                │
                 │ MLP: SiLU(gate) * up   │
                 │ down projection        │
                 └────────────────────── add
                                          │
                                          ▼
                                       output
```

代码：

```rust
let residual = x;
let x = self.rms_1.forward(x)?;
let x = (self.attn.forward(&x, index_pos, block_idx, cache)? + residual)?;
let residual = &x;
let x = (self.mlp.forward(&self.rms_2.forward(&x)?)? + residual)?;
```

第一个 `residual` 的类型是 `&Tensor`，只是复制引用；第二个 `residual = &x` 借用当前局部
Tensor。两者都没有复制 Tensor 数据。

## 6.4 Q、K、V shape

Attention 输入：

```text
x                            [B, S, H]
```

线性投影后：

```text
q_proj(x)                    [B, S, Nq * D]
k_proj(x)                    [B, S, Nkv * D]
v_proj(x)                    [B, S, Nkv * D]
```

reshape + transpose：

```text
Q                            [B, Nq,  S, D]
K                            [B, Nkv, S, D]
V                            [B, Nkv, S, D]
```

为什么 K/V head 数可以少于 Q？这是 GQA（Grouped Query Attention）。多个 query head
共享一组 K/V head，降低 KV Cache 容量和带宽。

`repeat_kv` 会把 K/V 扩展到与 query head 数兼容：

```text
[B, Nkv, T, D] -> [B, Nq, T, D]
```

注意这一步的具体实现可能使用 expand/reshape 等布局技巧；理解语义时先看 shape，分析性能时
再确认是否物理复制。

## 6.5 RoPE

Cache 初始化时预计算：

```text
cos[position, dim/2]
sin[position, dim/2]
```

每次 forward 根据：

```rust
index_pos
seq_len
```

从 cos/sin 中 `narrow` 当前区间，对 Q 和 K 应用旋转位置编码。

RoPE 不像传统 position embedding 那样直接加到 hidden state；它对 Q/K 的二维分量做旋转，
让 attention score 携带相对位置信息。

`index_pos` 必须正确累计，否则 token 会使用错误的位置角度。

## 6.6 Attention 数学与 shape

非 FlashAttention 路径：

```text
Q                            [B, Nq, S, D]
Kᵀ                           [B, Nq, D, T]
QKᵀ                          [B, Nq, S, T]
scale + causal mask
softmax
V                            [B, Nq, T, D]
attention * V                [B, Nq, S, D]
transpose + reshape          [B, S, H]
o_proj                       [B, S, H]
```

代码会暂时把 Q/K/V 转为 F32 计算 attention，提高数值稳定性，最后转回原 dtype。

FlashAttention 路径使用融合 kernel，避免显式物化完整 `[S, T]` attention matrix，并改变
Tensor layout 以适配 kernel 期望。

## 6.7 causal mask 为什么 decode 时可以省略

prefill 时 `S > 1`。第 i 个 token 不能看到未来 token，因此需要上三角 causal mask。

decode 时 `S == 1`，query 只有“最新位置”，历史 KV 全部是允许访问的位置，没有当前输入里的
未来 token，所以这条路径可以不构造 causal mask。

这不是说自回归模型不再 causal，而是 `S == 1` 时 causality 已由 cache 的构造方式保证。

## 6.8 KV Cache 保存什么

每层保存：

```rust
Option<(Tensor, Tensor)> // (K, V)
```

整体是：

```rust
Vec<Option<(Tensor, Tensor)>>
```

长度等于 Transformer 层数。

第一个 prefill：

```text
cache[layer] = K(prompt), V(prompt)
```

下一次 decode：

```text
K_total = concat(K_cache, K_new, seq_dim)
V_total = concat(V_cache, V_new, seq_dim)
cache[layer] = K_total, V_total
```

因此模型不必为每个新 token 重新计算所有历史 token 的 K/V。

缓存规模近似：

```text
2 * layers * batch * sequence * Nkv * D * bytes_per_element
```

GQA 减小 `Nkv`，因此能显著降低 KV Cache 内存。

## 6.9 prefill 与 decode 在入口中的切换

生成循环：

```rust
let (context_size, context_index) =
    if cache.use_kv_cache && index > 0 {
        (1, index_pos)
    } else {
        (tokens.len(), 0)
    };
```

第一次：

```text
index = 0
context_size = prompt token 数
context_index = 0
```

之后：

```text
context_size = 1
context_index = 已处理 token 数
```

`index_pos += ctxt.len()` 保证下一轮 RoPE 和 cache 位置继续前进。

关闭 KV Cache 后，每轮都把全部 tokens 重新 forward，计算复杂度和吞吐会明显恶化，但这是
验证 cache 正确性的有用对照。

## 6.10 为什么只取最后一个位置

Llama forward 最后：

```rust
let x = x.i((.., seq_len - 1, ..))?.contiguous()?;
let logits = self.lm_head.forward(&x)?;
```

生成下一个 token 只需要当前输入最后位置的 logits，所以不必对所有位置都执行 lm_head。

shape：

```text
Transformer output           [B, S, H]
选择最后位置                 [B, H]
lm_head                      [B, V]
```

训练时通常需要所有位置的 logits 来计算 next-token loss，推理接口和训练接口的输出策略会
不同。

## 6.11 生成循环中的 Rust 所有权

几个关键借用：

```rust
let ctxt = &tokens[...];              // 借用 Vec 的 slice
let input = Tensor::new(ctxt, ...)?;   // 使用 slice 创建 Tensor
let logits = llama.forward(
    &input,                            // 共享借用输入
    context_index,
    &mut cache,                        // 独占修改 KV Cache
)?;
tokens.push(next_token);               // ctxt 最后使用后才能修改 Vec
```

模型本身用 `&self`，说明 forward 不修改权重；Cache 用 `&mut`，说明状态变化被从模型参数中
显式分离。

这是一种很干净的服务设计：

```text
共享、只读 Model
    +
每个请求独立、可变 Cache
```

多请求服务可以共享模型权重，但每个会话必须维护自己的生成状态。

## 6.12 当前简单 Cache 的系统意义

这份 LLaMA 实现用 `Tensor::cat` 把历史 K/V 与新 K/V 拼接起来。语义直观，适合教学，但每轮
可能产生新分配和历史数据复制。

高吞吐推理引擎通常进一步使用：

- 预分配连续 KV Cache。
- 原地写入当前位置。
- Paged Attention。
- 多请求 continuous batching。
- block/page allocator。

因此 Candle 模型代码适合理解算法和 Rust 表达方式，但它不自动等于 vLLM 级服务调度器。

## 6.13 本课练习

### 练习 A：手算 shape

假设：

```text
B=1, S=16, H=4096, Nq=32, Nkv=8
```

回答：

1. `D` 是多少？
2. Q reshape 后是什么 shape？
3. K/V reshape 后是什么 shape？
4. cache 中已有 100 个 token 时，decode Attention score 是什么 shape？

### 练习 B：观测 prefill/decode

运行配套示例：

```bash
cargo run -p candle-examples \
  --example rust-candle-tour \
  -- \
  --cpu \
  --trace-shapes \
  --prompt 1,5,9,2 \
  -n 2
```

观察第一次 prefill 与后续 decode 的区别：

- `context_size` 第一次等于 prompt 长度，后续等于 1。
- Q 的 `S` 第一次等于 prompt 长度，后续等于 1。
- K/V cache 的 `T` 会随着生成逐步增长。
- logits 始终是 `[B, vocab]`，因为每轮只采样下一个 token。

### 练习 C：比较 KV Cache

运行配套示例：

```bash
cargo run -p candle-examples \
  --example rust-candle-tour \
  -- \
  --cpu \
  --compare-cache \
  --prompt 1,5,9,2 \
  -n 4
```

观察 cached 与 recompute-all：

- cached 第一次处理完整 prompt，后续只处理最新 token。
- recompute-all 每一步都从第一个 token 重新算到当前最后一个 token。
- 两者 greedy 结果应一致；差异在计算量，不在模型语义。
- tiny demo 的 token/s 只适合观察方法，不代表真实 LLM 的性能结论。
- `--compare-cache` 目前要求 `--temperature 0.0`，避免随机采样让对比变成 RNG 测试。

---

# 第 7 课：Rust 与 Candle 工程化

本课关注“能运行”之后的事情：如何组织 crate、选择 feature、测试、诊断错误和分析性能。

阅读范围：

- [workspace Cargo.toml](../Cargo.toml)
- [candle-examples features](../candle-examples/Cargo.toml#L66)
- [设备选择](../candle-examples/src/lib.rs#L11)
- [LLaMA tracing spans](../candle-transformers/src/models/llama.rs)

## 7.1 package、crate、module

三个词容易混淆：

| 概念 | 含义 |
| --- | --- |
| package | 一个包含 `Cargo.toml` 的发布/构建单元 |
| crate | 一次编译得到的 Rust library 或 binary |
| module | crate 内通过 `mod` 组织的命名空间与可见性单元 |

一个 package 可以包含：

- 一个 library crate：`src/lib.rs`
- 多个 binary crate：`src/bin/*.rs`
- examples：`examples/*.rs`
- tests 和 benches

Candle 根目录又是 workspace，把多个 package 统一管理。

## 7.2 Cargo feature 是编译期配置

示例 features：

```toml
[features]
cuda = [
    "candle/cuda",
    "candle-nn/cuda",
    "candle-transformers/cuda",
]
metal = [
    "candle/metal",
    "candle-nn/metal",
]
flash-attn = [
    "cuda",
    "candle-transformers/flash-attn",
]
```

feature 可以：

- 启用 optional dependency。
- 把 feature 继续传递给依赖 crate。
- 配合 `#[cfg(feature = "...")]` 编译不同代码。

例如 FlashAttention：

```rust
#[cfg(feature = "flash-attn")]
fn flash_attn(...) -> Result<Tensor> {
    // 真实实现
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(...) -> Result<Tensor> {
    unimplemented!("compile with '--features flash-attn'")
}
```

它不像普通配置文件那样运行时随意切换。改变 feature 通常需要重新编译。

## 7.3 常用构建命令

仅做类型检查：

```bash
cd candle
cargo check -p candle-examples --example llama
```

运行 CPU release：

```bash
cargo run -p candle-examples \
  --example llama \
  --release \
  -- \
  --cpu \
  --sample-len 32
```

Apple Silicon Metal：

```bash
cargo run -p candle-examples \
  --example llama \
  --release \
  --features metal \
  -- \
  --sample-len 32
```

CUDA：

```bash
cargo run -p candle-examples \
  --example llama \
  --release \
  --features cuda \
  -- \
  --sample-len 32
```

注意 `--`：

- 前面参数属于 Cargo。
- 后面参数传给 LLaMA example。

开发时先 `cargo check`；评估推理性能必须使用 `--release`。

## 7.4 `cfg` 与平台代码

```rust
#[cfg(feature = "accelerate")]
extern crate accelerate_src;
```

`cfg` 在编译前决定代码是否存在。常见条件包括：

```rust
#[cfg(feature = "cuda")]
#[cfg(target_os = "macos")]
#[cfg(target_arch = "aarch64")]
#[cfg(test)]
```

与 C++ `#ifdef` 相比，Rust `cfg` 作用于语法项和属性，并与 Cargo feature、target 信息紧密
集成；条件内代码仍然是 Rust AST。

## 7.5 错误上下文

仅有：

```text
cannot find tensor
```

往往不足以定位模型问题。应用层可以增加上下文：

```rust
use anyhow::Context;

let config_bytes = std::fs::read(&config_filename)
    .with_context(|| format!("failed to read {config_filename:?}"))?;
```

建议错误信息包含：

- 正在执行的阶段。
- 文件或 tensor 参数名。
- 期望和实际 shape/dtype/device。
- 底层错误链。

不要把所有错误转换成字符串后丢失类型和 source chain。

## 7.6 测试分层

### 纯函数单元测试

采样参数验证等逻辑不需要模型：

```rust
#[test]
fn rejects_zero_top_k() {
    assert!(validate_sampling(1.0, Some(0), None).is_err());
}
```

### 小 Tensor 测试

Linear、reshape、mask 可以用 CPU 小矩阵验证：

```rust
#[test]
fn scale_module_doubles_values() -> candle::Result<()> {
    let x = Tensor::new(&[1f32, 2., 3.], &Device::Cpu)?;
    let y = Scale { factor: 2.0 }.forward(&x)?;
    assert_eq!(y.to_vec1::<f32>()?, vec![2., 4., 6.]);
    Ok(())
}
```

### 模型集成测试

固定：

- 模型版本/权重 hash。
- tokenizer。
- prompt。
- seed。
- sampling 配置。

可以比较 token id 序列或 logits 容差。比较自然语言字符串更容易受 tokenizer 流式边界影响。

### 后端一致性测试

同一小输入分别运行 CPU/CUDA/Metal，按 dtype 使用合理容差比较。F16/BF16 不能期待逐 bit
一致。

## 7.7 tracing

LLaMA 代码为 Attention、RoPE、MLP、Block 创建了 tracing span。示例的 `--tracing` 会把
事件写成 Chrome trace 文件。

性能分析时至少区分：

```text
model load
prefill
decode
sampling
tokenizer/output
```

只看整体 token/s 会掩盖：

- 首 token 延迟 TTFT。
- steady-state decode 吞吐。
- 权重加载时间。
- host/device 同步。
- 单次异常分配。

建议记录：

| 指标 | 含义 |
| --- | --- |
| load time | 权重映射、传输和模型构建时间 |
| TTFT | prompt 输入到首 token 的延迟 |
| prefill tok/s | prompt 阶段吞吐 |
| decode tok/s | 稳态逐 token 吞吐 |
| peak memory | 权重、临时 Tensor 和 KV Cache 峰值 |

## 7.8 dtype 与性能

示例支持 F16、BF16、F32。选择不是简单的“越小越快”：

- 设备是否原生支持。
- 某些算子是否内部升到 F32。
- kernel 实现是否优化。
- 数值稳定性。
- 模型权重原始 dtype。
- dtype 转换是否引入额外分配。

Attention 中把 Q/K/V 转成 F32 就是数值稳定性与性能之间的明确取舍。

## 7.9 并发服务中的所有权设计

一个典型服务状态：

```text
Arc<Model>              多请求共享，只读权重
Request/Session Cache   每请求独立，持续可变
Tokenizer               视实现决定共享或请求内持有
Sampler RNG             每请求独立，保证 seed 和状态隔离
```

Rust 的 `Send`/`Sync` 会阻止明显不安全的跨线程共享，但不会自动解决：

- GPU stream 调度。
- 显存容量。
- session 生命周期。
- 公平排队。
- continuous batching。
- 锁竞争。

不要因为类型能放进 `Arc<Mutex<_>>` 就把整个模型 forward 锁住；这可能把并发完全串行化。

## 7.10 Clippy 与格式化

常用命令：

```bash
cargo fmt --check
cargo clippy -p candle-examples --example llama -- -D warnings
```

`rustfmt` 统一格式；Clippy 提供惯用法和潜在 bug 检查。Clippy 建议需要结合语义判断，不要机械
修改性能敏感代码。

## 7.11 本课练习

### 练习 A：建立基准记录

选择一个本地可用小模型，记录：

```text
commit:
device:
feature:
dtype:
prompt tokens:
generated tokens:
TTFT:
decode token/s:
```

以后每个性能修改都与同一基线比较。

### 练习 B：添加阶段计时

分别计时：

- tokenizer encode
- prefill
- decode
- sampling

注意 GPU 操作可能异步；需要时调用 device synchronize 才能测得真实执行时间。

### 练习 C：错误上下文

给 config、tokenizer 和权重加载分别增加 `with_context`，故意传入错误文件路径，比较修改前后
错误信息。

---

# 第 8 课：系统边界与 `unsafe`

本课不要求立即编写 CUDA kernel，而是学习如何安全阅读 Candle 最底层代码。

阅读范围：

- [BackendDevice/BackendStorage](../candle-core/src/backend.rs)
- [Storage runtime dispatch](../candle-core/src/storage.rs)
- [mmap VarBuilder](../candle-nn/src/var_builder.rs#L636)
- [CPU backend](../candle-core/src/cpu_backend/mod.rs)
- [CUDA backend](../candle-core/src/cuda_backend/mod.rs)
- [Metal backend](../candle-core/src/metal_backend/mod.rs)

## 8.1 Safe Rust 保证什么

Safe Rust 主要防止：

- use-after-free。
- double free。
- 空悬引用。
- 未同步的数据竞争。
- 越界 slice 访问。
- 违反引用别名规则。

它不保证：

- 算法正确。
- shape 逻辑正确。
- 不死锁。
- 不 OOM。
- GPU kernel 没有 bug。
- FFI 对端遵守约定。
- 性能一定好。

## 8.2 `unsafe` 允许的额外操作

典型能力包括：

- 解引用 raw pointer。
- 调用 unsafe function。
- 访问 `static mut`。
- 实现 unsafe trait。
- 访问 union 字段。

一个 `unsafe { ... }` 块并不会关闭借用检查。它只是允许上述少数操作，并要求程序员维护
相关 safety invariant。

## 8.3 Candle 中的两类 unsafe

### mmap

```rust
pub unsafe fn from_mmaped_safetensors(...)
```

调用者需要保证文件映射有效期间，底层文件不会以破坏映射的方式被修改。

### 未初始化分配

BackendDevice：

```rust
unsafe fn alloc_uninit(
    &self,
    shape: &Shape,
    dtype: DType,
) -> Result<Self::Storage>;
```

分配完成后数据尚未初始化。调用者必须在任何读取之前把所有需要的元素写好。

这种 API 避免“先清零、马上又完整覆盖”的额外成本，但把初始化不变量交给调用者。

## 8.4 阅读 unsafe 的五步法

看到 unsafe 时依次问：

1. 哪个具体操作需要 unsafe？
2. 函数文档声明了哪些 safety requirements？
3. 调用者如何证明这些要求成立？
4. unsafe 块是否小到可以独立审计？
5. safe wrapper 是否阻止外部制造非法状态？

不要因为一段代码“看起来底层”就扩大 unsafe 范围。

## 8.5 trait backend 与 enum dispatch 为什么同时存在

Candle 内部既有：

```rust
trait BackendStorage
```

又有：

```rust
enum Storage {
    Cpu(...),
    Cuda(...),
    Metal(...),
}
```

两者解决不同问题：

- trait 描述每个具体后端必须实现的操作集合和类型关系。
- enum 让普通 `Tensor` 在运行时持有任意受支持的 Storage，并对外暴露统一非泛型 API。

如果 `Tensor` 泛型化为 `Tensor<B>`，静态分发可能更直接，但模型代码的类型会传播 backend
参数，异构设备和公共 API 也更复杂。

当前设计是在易用性、动态设备选择和后端扩展之间的折中。

## 8.6 从 Matmul 到 kernel

完整路径：

```text
Tensor::matmul
    │ shape validation
    ▼
Storage::matmul
    │ runtime enum dispatch
    ▼
BackendStorage::matmul implementation
    │ layout/dtype specific selection
    ├── CPU: Rust loop / gemm / MKL / Accelerate
    ├── CUDA: CUDA kernels / cuBLAS 等
    └── Metal: Metal compute kernels
```

后端实现需要处理：

- dtype。
- contiguous/strided layout。
- batch。
- 转置。
- 对齐。
- kernel 能力。
- stream/command buffer。
- 错误映射。

上层 `Linear::forward` 主动把 contiguous 3D 输入展平，就是为了让这里更容易进入高效 matmul
路径。

## 8.7 FFI 的所有权问题

调用 C/CUDA API 时必须明确：

- 谁创建 handle/buffer？
- 谁负责释放？
- 错误路径是否也释放？
- 指针在异步 kernel 完成前是否仍有效？
- host buffer 是否 pinned？
- stream 同步发生在哪里？
- 外部 API 是否会保存传入指针？

Rust 通常用 RAII wrapper 实现 `Drop`，把原始 handle 封装成安全类型。但如果 `Drop` 时机早于
异步设备使用完成，仍然可能出错，所以必须理解后端同步语义。

## 8.8 自定义算子的边界

Candle 提供 CustomOp/InplaceOp 等扩展接口。一个自定义算子通常需要：

1. 定义输入 shape/dtype 检查。
2. 为 CPU 实现执行。
3. 可选实现 CUDA/Metal。
4. 决定是否支持反向传播。
5. 把底层错误转成 Candle Error。
6. 明确 output layout 和初始化。

推理算子即使不需要 backward，也必须正确处理：

- 空 Tensor。
- 非 contiguous 输入。
- 多 batch。
- dtype/device mismatch。
- 溢出与边界。

## 8.9 什么时候才应该自己写 kernel

优先级通常是：

1. 用现有 Tensor 算子表达。
2. 调整 shape/layout，让已有 fused/optimized path 生效。
3. 使用现有库 kernel。
4. profiling 证明瓶颈后再写 custom kernel。

“少一次 Rust 函数调用”通常不是 LLM 推理的关键优化；减少显存流量、中间 Tensor、同步和
kernel launch 更可能有效。

## 8.10 本课练习

### 练习 A：审计一个 unsafe

选择 `from_mmaped_safetensors` 或 `alloc_uninit`，写下：

```text
unsafe operation:
safety invariant:
who guarantees it:
what happens if violated:
safe wrapper boundary:
```

### 练习 B：追踪 CPU Matmul

从 `Tensor::matmul` 开始，依次找到：

1. shape 检查。
2. Storage enum 分发。
3. CpuStorage 的 `matmul` 实现。
4. 具体 dtype/layout 的选择。

不要一次读完整 CPU backend，只围绕这条路径。

### 练习 C：设计一个安全 wrapper

假设某 C API：

```c
handle_t* create_handle();
void destroy_handle(handle_t*);
int run(handle_t*, const float*, float*, size_t);
```

设计 Rust wrapper 的字段、构造函数、`Drop` 和 `run` 签名。标出真正需要 unsafe 的最小
代码范围。

---

# 综合项目：做一个自己的 Candle LLM Runner

学完各课后，把知识汇总为一个小项目。目标不是复制现有 `llama/main.rs`，而是写出自己能
解释的版本。

## 里程碑 1：最小推理

- 从本地路径加载 config、tokenizer 和 SafeTensors。
- CPU 运行。
- greedy decoding。
- 固定短 prompt。
- 打印生成 token。

对应课程：第 1、2、5、6 课。

## 里程碑 2：可配置生成

- clap 命令行参数。
- temperature/top-k/top-p。
- seed。
- repeat penalty。
- EOS 与最大长度。
- 流式 tokenizer 输出。

对应课程：第 1、2、7 课。

## 里程碑 3：设备与性能

- CPU/Metal/CUDA 选择。
- F16/BF16/F32。
- KV Cache 开关。
- TTFT、prefill、decode 分段计时。
- tracing。

对应课程：第 3、6、7、8 课。

## 里程碑 4：服务化设计

- `Arc<Model>` 共享权重。
- 每请求独立 Cache 和 RNG。
- 请求取消。
- 并发限制。
- 错误上下文。
- 基础 metrics。

这里会进入比 Candle 示例更接近生产推理系统的主题。

## 完成后的能力检查

你应该能解释：

1. Candle 的四层 crate 各自负责什么。
2. 一个 matmul 如何分发到 CPU/CUDA/Metal。
3. Tensor clone、view、contiguous 和物理 copy 的区别。
4. VarBuilder 如何把参数路径映射到 SafeTensors。
5. LLaMA Attention 每一步的 shape。
6. prefill 和 decode 的计算差异。
7. KV Cache 为什么需要每请求独立维护。
8. Rust 的所有权如何帮助设计并发推理服务。
9. mmap 和未初始化分配为什么需要 unsafe。
10. 如何用 tracing 和基准数据判断优化是否有效。

---

# C++ 开发者速查表

这张表只用于快速建立直觉，后续课程会逐项修正不完全等价的地方。

| Rust | C++ 近似概念 | 重要差异 |
| --- | --- | --- |
| `let x = value` | `const auto x = value` | Rust 默认不可变 |
| `let mut x` | `auto x` | 可变性必须明确声明 |
| `T` | value/object | 默认涉及 move 语义 |
| `&T` | `const T&` | 生命周期由编译器检查 |
| `&mut T` | `T&` | 必须是独占可变借用 |
| `Box<T>` | `std::unique_ptr<T>` | 所有权和 trait object 语义更严格 |
| `Rc<T>` | `std::shared_ptr<T>` | 非线程安全引用计数 |
| `Arc<T>` | 原子引用计数 `shared_ptr` | 能否跨线程还取决于 `Send`/`Sync` |
| `Option<T>` | `std::optional<T>` | 模式匹配使用更普遍 |
| `Result<T, E>` | `expected<T, E>` | `?` 提供标准错误传播 |
| `enum` | `enum` + `variant` | variant 可携带不同类型的数据 |
| `match` | `visit` + `switch` | 必须穷尽并可解构 |
| `trait` | interface/concept | 同时支持静态和动态分发 |
| `impl Trait` | 受约束的推导类型 | 具体语义取决于参数或返回位置 |
| `Vec<T>` | `std::vector<T>` | 借用规则阻止迭代器/切片失效 |
| `&[T]` | `std::span<const T>` | 生命周期和别名规则受检查 |
| `String` | `std::string` | 保证 UTF-8 |
| `&str` | `std::string_view` | 生命周期受检查且保证 UTF-8 |
| `Drop` | destructor/RAII | 不能显式直接调用 `drop` 方法 |
| `unsafe` | 手动承担额外不变量 | 不会关闭借用检查，也不是“随便做任何事” |

## 学习日志

后续每完成一课，在这里记录：

```text
日期：
完成课程：
我已经能解释：
仍然困惑：
编译器帮我发现的问题：
```
