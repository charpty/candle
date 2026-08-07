# Candle 源码阅读路线图

这份文档解决一个具体问题：面对 Candle 这样的大仓库，不应该从第一行开始通读。你要按一次
推理请求的生命周期阅读，每次只追一条链路。

如果你在阅读链路时卡在某个 Rust/Candle 写法，例如 `Result`、`Option`、`contiguous()`、
`VarBuilder::pp` 或 `Arc<RwLock<_>>`，回到
[Rust + Candle 典型用法精讲](./typical-patterns.md) 查对应条目。

读源码时先定目标：

```text
我要解释一次生成请求如何从 CLI 参数走到 Tensor 算子和硬件后端。
```

不要定这种目标：

```text
我要把 candle-core 全部看完。
```

第二个目标太大，也没有反馈闭环。下面按调用链拆开。

---

## 0. 总路线

一条最小 LLaMA 推理路径：

```text
candle-examples/examples/llama/main.rs
    Args / Device / DType / tokenizer / safetensors
    |
    v
candle-transformers/src/models/llama.rs
    Llama::load / Llama::forward / Block / Attention / Cache
    |
    v
candle-nn
    Linear / Embedding / VarBuilder / Module
    |
    v
candle-core
    Tensor / Shape / Layout / Storage / Device
    |
    v
CPU / CUDA / Metal backend
```

配套 tiny tour 是这条链路的缩小版：

```text
candle-examples/examples/rust-candle-tour/main.rs
candle-examples/examples/rust-candle-tour/model.rs
candle-examples/examples/rust-candle-tour/sampling.rs
candle-examples/examples/rust-candle-tour/backend_walkthrough.rs
candle-examples/examples/rust-candle-tour/layout_walkthrough.rs
```

先把 tiny tour 讲清楚，再读真实 LLaMA。

---

## 1. 入口：CLI 参数和运行时选择

读：

- `candle-examples/examples/llama/main.rs`
- `candle-examples/examples/rust-candle-tour/main.rs`

关注：

```rust
#[derive(Parser, Debug)]
struct Args { ... }

let device = candle_examples::device(args.cpu)?;
let dtype = match args.dtype.as_deref() { ... };
```

你要能解释：

1. 哪些参数属于 Cargo，哪些参数属于 example。
2. `--features cuda/metal` 是编译期能力，不是运行时开关。
3. `--cpu` 是运行时选择。
4. `DType` 决定权重和中间 Tensor 的数值类型。

不要在这里追：

- 每个模型仓库如何下载。
- 每个后端如何初始化。
- 所有 clap 属性。

停下来的标准：

你能说清楚“这个二进制能不能用 CUDA”和“这次运行实际用不用 CUDA”是两个问题。

---

## 2. 文件定位：config、tokenizer、SafeTensors

读：

- `candle-examples/examples/llama/main.rs`
- `candle-examples/src/lib.rs`
- `candle-nn/src/var_builder.rs`

关注：

```rust
let config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;
let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
```

你要能解释：

1. config 是结构化模型元数据。
2. tokenizer 把字符串和 token id 互相转换。
3. SafeTensors 保存命名权重。
4. VarBuilder 用参数路径取权重，不是模型层本身。
5. mmap 为什么需要 unsafe。

不要在这里追：

- tokenizer 的完整算法。
- safetensors 文件格式细节。
- hub 客户端所有分支。

停下来的标准：

你能画出：

```text
config.json -> LlamaConfig
tokenizer.json -> Tokenizer
*.safetensors -> VarBuilder -> model layers
```

---

## 3. 模型构造：`Llama::load`

读：

- `candle-transformers/src/models/llama.rs`
- `candle-examples/examples/rust-candle-tour/model.rs`
- `candle-nn/src/linear.rs`
- `candle-nn/src/embedding.rs`

关注：

```rust
let embed_tokens = embedding(vocab_size, hidden_size, vb.pp("model.embed_tokens"))?;
let q_proj = linear_no_bias(hidden, hidden, vb.pp("q_proj"))?;
let blocks = (0..num_layers)
    .map(|i| Block::load(vb.pp(format!("model.layers.{i}")), cfg))
    .collect::<Result<Vec<_>>>()?;
```

你要能解释：

1. `pp` 如何累积参数路径。
2. `linear_no_bias` 最终取什么权重名。
3. 多层 Block 为什么适合 iterator + `collect::<Result<Vec<_>>>()?`。
4. 模型结构用 struct 组合，而不是继承树。

不要在这里追：

- 每个模型变体所有字段。
- 每个权重初始化方式。

停下来的标准：

你能从 `vb.pp("model.layers.3").pp("self_attn").pp("q_proj")` 推出最终权重路径。

---

## 4. 生成循环：prefill 与 decode

读：

- `candle-examples/examples/llama/main.rs`
- `candle-examples/examples/rust-candle-tour/main.rs`

关注：

```rust
let context_size = if step == 0 { tokens.len() } else { 1 };
let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;
let logits = model.forward(&input, index_pos, &mut cache, trace_shapes)?;
```

你要能解释：

1. 第一次为什么喂完整 prompt。
2. 后续为什么只喂最新 token。
3. `ctxt` 为什么是 slice 借用。
4. `index_pos` 为什么要随已处理 token 推进。
5. cache 为什么必须是 `&mut`。

用 tiny tour 验证：

```bash
cargo run -p candle-examples --example rust-candle-tour -- --cpu --trace-shapes -n 2
cargo run -p candle-examples --example rust-candle-tour -- --cpu --compare-cache -n 4
```

不要在这里追：

- attention 的所有数学细节。
- backend kernel。

停下来的标准：

你能解释 cached 和 recompute-all 在 greedy 下为什么 token 相同，但计算量不同。

---

## 5. Block：残差、Norm、Attention、MLP

读：

- `candle-transformers/src/models/llama.rs`
- `candle-examples/examples/rust-candle-tour/model.rs`

真实 LLaMA block 通常是：

```text
x
  -> RMSNorm
  -> Attention
  -> residual add
  -> RMSNorm
  -> MLP
  -> residual add
```

tiny tour 只保留：

```text
Embedding
  -> Attention
  -> residual add
  -> lm_head
```

你要能解释：

1. 为什么 residual 需要保存旧 hidden。
2. `Module::forward` 适合 Linear、Embedding、Norm。
3. 带 cache、位置、mask 的模型 forward 往往需要自定义签名。
4. 每一步 Tensor shape 是否保持 `[B, S, H]`。

不要在这里追：

- 每种激活函数细节。
- 所有模型变体。

停下来的标准：

你能沿一个 block 说明哪些 Tensor shape 不变，哪些会临时展开成 head 维。

---

## 6. Attention：Q/K/V、mask、KV Cache

读：

- `candle-transformers/src/models/llama.rs`
- `candle-examples/examples/rust-candle-tour/model.rs`

关注 shape：

```text
hidden [B, S, H]
q      [B, N, S, D]
k/v    [B, Nkv, T, D]
score  [B, N, S, T]
prob   [B, N, S, T]
ctx    [B, N, S, D] -> [B, S, H]
```

tiny tour 简化了 GQA，所以 Q/K/V head 数相同。真实 LLaMA 常见 GQA：

```text
Nq > Nkv
```

需要 `repeat_kv` 或等价处理，让 K/V 与 Q head 兼容。

你要能解释：

1. Q 是当前输入位置的 query。
2. K/V 是历史 token 和当前 token 的 key/value。
3. prefill 需要 causal mask。
4. decode 时 `S=1`，通常不需要屏蔽“当前输入内部的未来 token”。
5. cache 保存的是每层 K/V，不是 logits。

不要在这里追：

- RoPE 的所有公式。
- FlashAttention kernel 实现。

停下来的标准：

你能手算 `B=1, S=4, H=16, N=4` 时每一步 shape。

---

## 7. Tensor API：shape 与 layout

读：

- `candle-core/src/tensor.rs`
- `candle-core/src/layout.rs`
- `candle-core/src/shape.rs`
- `candle-examples/examples/rust-candle-tour/layout_walkthrough.rs`

关注：

```rust
x.dims()
x.stride()
x.layout().start_offset()
x.is_contiguous()
x.transpose(1, 2)?
x.contiguous()?
```

你要能解释：

1. Shape 是维度大小。
2. Layout 是 shape + stride + start offset。
3. Storage 是真实数据。
4. transpose/narrow 通常是 view。
5. contiguous 必要时分配并复制。

用 tiny tour 验证：

```bash
cargo run -p candle-examples --example rust-candle-tour -- --cpu --show-layout-path -n 0
```

不要在这里追：

- 每个 Tensor 方法。
- 所有 indexer 组合。

停下来的标准：

你能看到一个 Tensor 操作后判断“shape 变了没有、stride 变了没有、Storage 可能复制没有”。

---

## 8. Matmul：从 Tensor 到 backend

读：

- `candle-core/src/tensor.rs`
- `candle-core/src/storage.rs`
- `candle-core/src/backend.rs`
- `candle-examples/examples/rust-candle-tour/backend_walkthrough.rs`

调用链：

```text
Tensor::matmul
  -> shape/rank/batch 检查
  -> Storage::matmul
  -> match Cpu/Cuda/Metal
  -> backend-specific matmul
  -> new Tensor with result Storage/Layout
```

你要能解释：

1. Tensor 层负责统一 API 和 shape 逻辑。
2. Storage 层负责按 Device 分发。
3. backend trait 描述每个具体后端必须实现的能力。
4. enum dispatch 和 trait backend 为什么同时存在。

用 tiny tour 验证：

```bash
cargo run -p candle-examples --example rust-candle-tour -- --cpu --show-backend-path -n 0
```

不要在这里追：

- BLAS/GEMM 全部实现细节。
- CUDA/Metal kernel 每一行。

停下来的标准：

你能说出 `a.matmul(&b)?` 从 API 到 CPU/CUDA/Metal 分发经过哪几层。

---

## 9. Sampling：logits 到 token

读：

- `candle-transformers/src/generation/mod.rs`
- `candle-examples/examples/rust-candle-tour/sampling.rs`

流程：

```text
logits [vocab]
  -> optional repeat penalty
  -> temperature
  -> softmax
  -> top-k/top-p
  -> sample or argmax
  -> next token id
```

你要能解释：

1. `temperature == 0` 为什么走 ArgMax。
2. `LogitsProcessor` 为什么需要 `&mut self`。
3. top-k 和 top-p 的组合为什么适合 enum。
4. greedy 对比 cache 时为什么更干净。

不要在这里追：

- 所有概率采样数学证明。
- tokenizer decode 细节。

停下来的标准：

你能把 `(temperature, top_k, top_p)` 映射到一个明确的 `Sampling` variant。

---

## 10. 错误、测试和回归保护

读：

- `candle-examples/examples/rust-candle-tour/*.rs`
- `candle-core/tests`
- `candle-examples/examples/*/main.rs`

tiny tour 的测试覆盖：

```text
prompt parse
sampling parameter validation
matmul demo result
layout stride snapshots
prefill/decode cache length
cached vs recompute-all token equality
```

你要能解释：

1. 为什么测试函数可以返回 `Result<()>`。
2. 哪些测试是纯函数测试。
3. 哪些测试需要 Tensor。
4. 为什么测试输入要小且不依赖网络。

不要在这里追：

- 全仓库所有测试。
- GPU 后端一致性测试的全部容差策略。

停下来的标准：

你能为一个新学习点补一个小测试，而不是只靠手动运行。

---

## 11. 性能观察：不要只看 token/s

读：

- LLaMA example 的 tracing 初始化。
- `candle-transformers/src/models/llama.rs` 中的 tracing span。
- 主讲义第 7 课。

至少拆：

```text
load time
tokenizer time
prefill time
time to first token
decode token/s
sampling time
host/device copy
peak memory
```

你要能解释：

1. 首次运行和稳态运行不同。
2. GPU 异步执行会影响计时。
3. tiny demo 的 token/s 不能代表真实 LLM。
4. tracing span 帮你定位慢在哪里，而不是只给一个总耗时。

不要在这里追：

- 一上来写自定义 kernel。
- 未测量就优化。

停下来的标准：

你能说出一个性能结论需要哪些指标支持。

---

## 12. `unsafe` 和系统边界

读：

- `candle-nn/src/var_builder.rs`
- `candle-core/src/backend.rs`
- CUDA/Metal backend 的 FFI 边界。

常见 unsafe 来源：

```text
mmap
未初始化分配
FFI
GPU resource handle
```

读 unsafe 不要问“这里是不是危险”，先问：

1. 具体哪个操作需要 unsafe？
2. 额外不变量是什么？
3. 谁保证它？
4. safe wrapper 把边界收在哪里？

不要在这里追：

- 把所有 unsafe 都展开到系统调用。
- 因为看到 unsafe 就否定整段设计。

停下来的标准：

你能为一个 unsafe 块写出 safety invariant。

---

## 13. 服务化边界：共享什么，独立什么

读：

- 主讲义第 7 课。
- tiny tour 的 `run_generation`。

合理边界：

```text
共享:
  Arc<Model>
  Device
  read-only weights

每请求独立:
  tokens
  KvCache
  sampler/RNG
  cancellation state
```

你要能解释：

1. 权重共享是为了内存。
2. cache 独立是为了请求语义。
3. RNG 独立是为了采样可复现和请求隔离。
4. continuous batching 需要显式调度，不是简单线程池。

不要在这里追：

- 直接实现 vLLM 级 scheduler。
- 过早设计复杂 cache allocator。

停下来的标准：

你能画出两个请求共享同一个模型，但各自拥有 cache 的状态图。

---

## 14. 每次读源码的记录模板

建议每次读一个小片段时都记录：

```text
文件：
函数：
这段代码属于哪层：example / model / nn / core / backend
输入 Tensor shape：
输出 Tensor shape：
是否借用外部数据：
是否修改状态：
是否可能失败：
是否可能分配：
是否可能 host/device 拷贝：
我还不能解释的点：
下一次只追哪一层：
```

这个模板比“看了很多行”更有价值。你每填完一次，就会更接近能独立改 Candle 代码。

---

## 15. 推荐学习顺序

如果从零开始，按这个顺序推进：

1. 跑通 `rust-candle-tour --cpu -n 2`。
2. 阅读 `main.rs`，解释 tokens、ctxt、input、logits、next token。
3. 阅读 `sampling.rs`，解释 `Option` 和 `Sampling` enum。
4. 运行 `--trace-shapes`，手算每个 Tensor shape，并解释关键 dtype。
5. 阅读 `model.rs`，解释 Embedding、Attention、KV Cache。
6. 运行 `--compare-cache`，解释 cached 和 recompute-all。
7. 运行 `--show-layout-path`，解释 stride 和 contiguous。
8. 运行 `--show-backend-path`，解释 matmul 分发层次。
9. 回到真实 `llama/main.rs`，对照 tiny tour 找相同结构。
10. 只追一个真实函数，例如 `CausalSelfAttention::forward`。
11. 补一个小测试或小打印验证你的理解。
12. 再进入下一个函数。

不要跳过第 11 步。没有验证的阅读很容易变成“感觉看懂了”。
