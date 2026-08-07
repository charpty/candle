# Rust + Candle 动手实验手册：把概念跑到手上

这份文档不是再讲一遍概念，而是逼你把概念落到代码、命令输出和编译器错误里。

配套代码在：

- [`candle-examples/examples/rust-candle-tour/main.rs`](../candle-examples/examples/rust-candle-tour/main.rs)
- [`candle-examples/examples/rust-candle-tour/model.rs`](../candle-examples/examples/rust-candle-tour/model.rs)
- [`candle-examples/examples/rust-candle-tour/sampling.rs`](../candle-examples/examples/rust-candle-tour/sampling.rs)
- [`candle-examples/examples/rust-candle-tour/backend_walkthrough.rs`](../candle-examples/examples/rust-candle-tour/backend_walkthrough.rs)
- [`candle-examples/examples/rust-candle-tour/layout_walkthrough.rs`](../candle-examples/examples/rust-candle-tour/layout_walkthrough.rs)

建议不要只读。每个实验都至少跑一次命令，再做一次“小破坏”，看编译器或测试怎么把你带回来。

## 使用方式

每个实验分成六步：

1. **目标**：本节真正要掌握什么。
2. **运行**：先跑一条不会破坏代码的命令。
3. **观察**：解释输出对应的 Rust/Candle 语义。
4. **动手改**：改一两行代码，制造一个可控变化。
5. **验证**：用 `cargo test` 或运行示例确认理解。
6. **达标**：不用看文档也能回答的问题。

统一从仓库根目录运行：

```bash
cd candle
```

基础验证命令：

```bash
cargo test -p candle-examples --example rust-candle-tour
```

如果本机 Cargo 配置指向失效的 Git 索引，可以临时使用已经配置过的 sparse 索引：

```bash
cargo --config 'source.crates-io.replace-with="rsproxy-sparse"' \
  test -p candle-examples \
  --example rust-candle-tour
```

## 实验 0：先建立基线

### 目标

确认你当前工作区可以编译、运行、测试。后面的每个实验都依赖这个基线。

### 运行

```bash
cargo run -p candle-examples --example rust-candle-tour -- --cpu -n 2
```

你应该看到类似：

```text
device: Cpu
prompt token ids: [1, 5, 9, 2]
generation:
  cached step=00 input=[1, 5, 9, 2] -> token=...
  cached step=01 input=[...] -> token=...
all token ids: [...]
```

具体 token 取决于权重和采样参数，但结构应该稳定。

### 观察

这条命令同时走过：

- `Args::parse()`：命令行参数进入强类型结构。
- `parse_prompt`：字符串转 `Vec<u32>`，失败走 `Result`。
- `Tensor::new(...).unsqueeze(0)`：host token 进入 Candle Tensor。
- `TinyCausalLm::forward`：Embedding、Attention、lm_head。
- `LogitsProcessor::sample`：从 logits 得到下一个 token。
- `tokens.push(next_token)`：生成状态推进。

这个例子小，但它和真实 LLaMA CLI 的外层骨架是一致的。

### 验证

```bash
cargo test -p candle-examples --example rust-candle-tour
```

达标标准不是“测试过了”，而是你能说清楚其中至少三个测试分别保护了什么行为。

### 达标问题

1. `main() -> Result<()>` 为什么比 `main()` 更适合这个示例？
2. `--cpu` 为什么是 bool，而 `--top-k` 是 `Option<usize>`？
3. 生成循环里哪些变量必须是 `mut`？

## 实验 1：Result 不是异常，是函数签名的一部分

### 目标

掌握 `Result<T>`、`?`、`bail!` 在 Candle 示例里的典型用法。

Candle 中几乎所有 Tensor 操作都可能失败：

- shape 不匹配。
- dtype 不支持。
- device 不一致。
- 后端 kernel 不存在。
- 权重名缺失。

所以 `Result` 会从 `Tensor::new`、`reshape`、`matmul` 一路传到 `main`。

### 运行

```bash
cargo run -p candle-examples --example rust-candle-tour -- --cpu --prompt "1,,2"
```

你应该看到 `prompt contains an empty token id` 这一类错误。

再运行：

```bash
cargo run -p candle-examples --example rust-candle-tour -- --cpu --prompt "1,32"
```

你应该看到 token 超出词表范围的错误。

### 观察

`parse_prompt` 的核心写法是：

```rust
let tokens = prompt
    .split(',')
    .map(str::trim)
    .map(|piece| {
        if piece.is_empty() {
            bail!("prompt contains an empty token id")
        }
        piece
            .parse::<u32>()
            .map_err(|error| anyhow::anyhow!("invalid token id {piece:?}: {error}"))
    })
    .collect::<Result<Vec<_>>>()?;
```

关键点：

- `bail!` 直接构造错误并从当前闭包返回。
- `parse::<u32>()` 返回标准库的 `Result<u32, ParseIntError>`。
- `map_err` 把底层错误包成更适合用户看的错误。
- `collect::<Result<Vec<_>>>()` 会在第一处错误停止。
- 最后的 `?` 把错误继续交给调用者。

C++ 里你可能会用异常、返回码、`StatusOr<T>` 或 `expected<T, E>`。Rust 标准库直接把这套模式放进类型系统和 `?` 语法里。

### 动手改

把 `parse_prompt` 里的：

```rust
if prompt.trim().is_empty() {
    bail!("prompt must contain at least one token id")
}
```

临时删掉，然后运行：

```bash
cargo test -p candle-examples --example rust-candle-tour parse_prompt_rejects_blank_prompt
```

这个测试应该失败。

### 修回来

恢复空字符串检查。这个小实验的重点是：参数校验不是“边角逻辑”，它保护的是后续 Tensor shape。

如果空 prompt 没有被拦住，后面很容易出现：

- 空 `Vec`。
- `[1, 0]` 形状 Tensor。
- `seq_len - 1` 下溢或越界。
- 生成循环语义变得不明确。

### 达标问题

1. `?` 会不会吞掉错误？
2. 为什么错误信息最好带上用户输入值？
3. `collect::<Result<Vec<_>>>()` 遇到第三个元素失败时，会不会继续处理第四个元素？

## 实验 2：Option 是状态空间，不是“可空指针语法糖”

### 目标

理解 `Option<T>` 在采样参数和 KV Cache 中的两种典型含义。

本示例有两类 `Option`：

- CLI 参数：`top_k: Option<usize>`、`top_p: Option<f64>`。
- 运行状态：`KvCache { kv: Option<(Tensor, Tensor)> }`。

前者表示“用户有没有提供这个参数”。后者表示“请求是否已经有历史 K/V”。

### 运行

```bash
cargo run -p candle-examples --example rust-candle-tour -- --cpu --temperature 0.7 --top-k 5 -n 2
```

再运行：

```bash
cargo run -p candle-examples --example rust-candle-tour -- --cpu --temperature 0.7 --top-p 0.9 -n 2
```

再运行：

```bash
cargo run -p candle-examples --example rust-candle-tour -- --cpu --temperature 0.7 --top-k 5 --top-p 0.9 -n 2
```

### 观察

`sampling.rs` 里：

```rust
match (top_k, top_p) {
    (None, None) => Sampling::All { temperature },
    (Some(k), None) => Sampling::TopK { k, temperature },
    (None, Some(p)) => Sampling::TopP { p, temperature },
    (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
}
```

这不是为了炫技。它的价值是把四种状态穷尽列出来。

如果后来新增一个采样参数，比如 `min_p`，你不能只“顺手加个 if”。你应该重新审视状态空间：

- 哪些参数可以组合？
- 哪些参数互斥？
- 默认值是什么？
- 错误应该在 CLI 解析层、采样构建层还是模型 forward 层暴露？

### 动手改

临时把：

```rust
if top_k == Some(0) {
    bail!("top-k must be greater than zero")
}
```

删掉，然后运行：

```bash
cargo test -p candle-examples --example rust-candle-tour rejects_invalid_top_k
```

测试应该失败。

### 观察 KV Cache

运行：

```bash
cargo run -p candle-examples --example rust-candle-tour -- --cpu --trace-shapes -n 3
```

你会看到第一步是 prefill，后面是 decode：

```text
cached prefill: index_pos=0 context_size=4 input_shape=[1, 4]
...
cached decode: index_pos=4 context_size=1 input_shape=[1, 1]
...
```

`KvCache` 里的 `Option` 状态变化是：

```text
None
  -> Some(K/V length = prompt_len)
  -> Some(K/V length = prompt_len + 1)
  -> Some(K/V length = prompt_len + 2)
```

### 达标问题

1. `Option<Tensor>` 和空 Tensor 是不是一回事？
2. 为什么 `cache.len()` 用 `as_ref()`？
3. `Sampling::ArgMax` 为什么可以忽略 `top_k/top_p`？

## 实验 3：借用检查器保护生成循环里的 Vec

### 目标

通过一个真实的 E0502 场景理解“共享借用”和“可变借用”为什么不能重叠。

`run_generation` 中有这段：

```rust
let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
...
println!("  ... input={ctxt:?} ...");
cache_lens.push(cache_len);
tokens.push(next_token);
```

这里 `ctxt` 是借用 `tokens` 的 slice。

### 动手改

临时把：

```rust
tokens.push(next_token);
```

移动到 `println!` 前面，并且仍然在 `println!` 里使用 `ctxt`。

然后运行：

```bash
cargo test -p candle-examples --example rust-candle-tour
```

你应该看到类似 E0502 的错误：不能在不可变借用仍被使用时可变借用 `tokens`。

### 为什么这不是编译器小题大做

`Vec::push` 可能重新分配底层内存。`ctxt` 是指向旧底层内存的一段 slice。如果允许 `push` 和旧 slice 同时存在，`ctxt` 可能悬垂。

Rust 不需要知道这次 `push` 是否真的触发重分配。只要类型语义允许重分配，它就必须禁止这类重叠。

C++ 里类似代码可以编译，但你需要自己记住：

- `std::vector::push_back` 可能让引用、指针、迭代器失效。
- 失效后继续读就是未定义行为。

Rust 把这个规则提前成编译期错误。

### 修法

修法不是“绕过借用检查器”，而是收窄借用生命周期。

当前示例的写法就是一种修法：

```rust
println!("  ... input={ctxt:?} ...");
tokens.push(next_token);
```

另一个等价写法是把 `ctxt` 的使用放进内部 block：

```rust
{
    let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
    let input = Tensor::new(ctxt, device)?.unsqueeze(0)?;
    let logits = model.forward(&input, forward_index_pos, active_cache, trace_shapes)?;
    println!("input={ctxt:?}");
}
tokens.push(next_token);
```

不过当前代码还要在 block 外使用 `next_token` 和 cache 信息，所以不能机械照搬。真实工程里要先理清数据流。

### 达标问题

1. `ctxt` 的类型是什么？
2. 为什么 `tokens.push` 需要 `&mut self`？
3. 如果先 `let ctxt = tokens[...] .to_vec()`，错误会消失吗？代价是什么？

## 实验 4：Tensor shape 和 dtype 是推理系统的合同

### 目标

理解每个 Tensor 操作前后 shape 和 dtype 的变化，并把它们当作系统合同看待。

### 运行

```bash
cargo run -p candle-examples --example rust-candle-tour -- --cpu --trace-shapes -n 1
```

你会看到类似：

```text
token ids [B,S]        shape=[1, 4] dtype=U32
embedding [B,S,H]      shape=[1, 4, 16] dtype=F32
attn input [B,S,H]     shape=[1, 4, 16] dtype=F32
q [B,N,S,D]            shape=[1, 4, 4, 4] dtype=F32
k new [B,N,S,D]        shape=[1, 4, 4, 4] dtype=F32
v new [B,N,S,D]        shape=[1, 4, 4, 4] dtype=F32
scores [B,N,S,T]       shape=[1, 4, 4, 4] dtype=F32
probs [B,N,S,T]        shape=[1, 4, 4, 4] dtype=F32
context [B,S,H]        shape=[1, 4, 16] dtype=F32
last hidden [B,H]      shape=[1, 16] dtype=F32
logits [B,V]           shape=[1, 32] dtype=F32
```

### 观察

把字母记牢：

| 符号 | 含义 | 本例值 |
| --- | --- | --- |
| `B` | batch size | 1 |
| `S` | 当前输入序列长度 | prefill 是 4，decode 是 1 |
| `T` | cache 中总 K/V 长度 | 逐步增长 |
| `H` | hidden size | 16 |
| `N` | attention heads | 4 |
| `D` | head dim | 4 |
| `V` | vocabulary size | 32 |

重要不变量：

```text
H = N * D
q: [B, N, S, D]
k/v cache: [B, N, T, D]
scores: [B, N, S, T]
logits: [B, V]
```

dtype 也有合同：

```text
token ids: 整数 dtype，供 Embedding 做索引
hidden/q/k/v/scores/probs/context: 浮点 dtype，参与矩阵乘和 softmax
logits: F32，采样前保持稳定的数值边界
```

### 动手改

临时把 `TinyConfig` 改成：

```rust
let config = TinyConfig {
    vocab_size: 32,
    hidden_size: 18,
    num_heads: 4,
};
```

然后运行：

```bash
cargo run -p candle-examples --example rust-candle-tour -- --cpu
```

你应该看到配置校验错误。

### 为什么要早失败

如果不在 `config.validate()` 里拦住，错误会拖到：

```rust
reshape((batch, seq_len, self.num_heads, self.head_dim))?
```

那里才暴露。那时错误信息会更像“元素数量对不上”，离真正原因更远。

工程原则：

- 配置不变量在加载模型前验证。
- 请求不变量在构造 Tensor 前验证。
- shape 不变量尽量在模块边界处明确。
- 后端不变量通过 `Result` 传出来，不要 `unwrap`。

### 达标问题

1. 为什么 `head_dim = hidden_size / num_heads` 需要整除？
2. `logits` 为什么只保留最后一个位置？
3. prefill 的 `S` 和 decode 的 `S` 有什么不同？
4. token ids 和 logits 的 dtype 为什么不一样？

## 实验 5：Layout、stride、contiguous 不是性能细节

### 目标

理解 Tensor view 和实际内存布局之间的区别。

### 运行

```bash
cargo run -p candle-examples --example rust-candle-tour -- --cpu --show-layout-path -n 0
```

你会看到：

```text
layout tour:
  base                 dims=[2, 3, 4] stride=[12, 4, 1] offset=0 contiguous=true
  transpose(1,2)       dims=[2, 4, 3] stride=[12, 1, 4] offset=0 contiguous=false
  narrow(dim=1)        dims=[2, 2, 4] stride=[12, 4, 1] offset=4 contiguous=false
  contiguous()         dims=[2, 4, 3] stride=[12, 3, 1] offset=0 contiguous=true
```

### 观察

`transpose(1, 2)` 没有马上复制数据。它改变的是：

- dims：逻辑维度顺序。
- stride：沿每个维度走一步时底层 Storage 偏移多少。

`narrow(dim=1)` 改变的是：

- dims：截出来的范围更短。
- start offset：view 从 Storage 中间开始。

`contiguous()` 做的是：

- 如果当前 Tensor 已经连续，通常可以复用。
- 如果当前 Tensor 是非连续 view，就复制成新的 row-major 布局。

在 Attention 里，`transpose(1, 2)?.contiguous()?` 很常见，因为后续 matmul/kernel 往往更喜欢连续布局。

### 动手改

在 `model.rs` 里临时删掉 q/k/v 上的 `.contiguous()?`：

```rust
.transpose(1, 2)?;
```

然后运行：

```bash
cargo test -p candle-examples --example rust-candle-tour
```

如果当前 CPU 后端仍然能处理非连续布局，测试可能仍然通过。这并不说明 `contiguous()` 没意义。

你要观察的是：

- 某些后端或 kernel 对 layout 更挑剔。
- 非连续输入可能触发内部复制。
- 显式 `contiguous()` 能把复制点固定在你看得见的位置。

### 修回来

恢复 `.contiguous()?`。

真正工程里，不要到处无脑加 `contiguous()`。判断规则：

- 如果马上进入 layout 敏感的 kernel，可以加。
- 如果只是轻量 view 变换后还会继续组合，先不要加。
- 如果性能异常，打印 layout 或用 profiler 验证复制点。

### 达标问题

1. shape 相同的两个 Tensor，layout 一定相同吗？
2. `transpose` 为什么通常可以 O(1)？
3. `contiguous()` 的成本什么时候接近 O(n)？

## 实验 6：VarBuilder 权重路径是模型 ABI

### 目标

理解 `VarBuilder` 的路径拼接规则，以及为什么权重名错误会导致加载失败。

### 运行

```bash
cargo run -p candle-examples --example rust-candle-tour -- --cpu --show-weight-paths -n 0
```

你会看到：

```text
weight path tour:
  model.embed_tokens.weight                    [32, 16]
  model.layers.0.self_attn.q_proj.weight       [16, 16]
  model.layers.0.self_attn.k_proj.weight       [16, 16]
  model.layers.0.self_attn.v_proj.weight       [16, 16]
  model.layers.0.self_attn.o_proj.weight       [16, 16]
  lm_head.weight                               [32, 16]
  path rule: vb.pp("model.layers.0.self_attn").pp("q_proj") -> model.layers.0.self_attn.q_proj.weight
```

### 观察

`model.rs` 里：

```rust
TinyAttention::load(vb.pp("model.layers.0.self_attn"), config)?
```

进入 attention 后：

```rust
linear_no_bias(hidden, hidden, vb.pp("q_proj"))?
```

`linear_no_bias` 会找：

```text
model.layers.0.self_attn.q_proj.weight
```

这就是参数路径 ABI。

ABI 这个说法是刻意的：代码和权重文件之间没有编译器共同检查。路径错了，只有运行加载时才知道。

### 动手改

临时把 `from_demo_weights` 中权重名：

```rust
"model.layers.0.self_attn.q_proj.weight"
```

改成：

```rust
"model.layers.0.self_attn.query_proj.weight"
```

然后运行：

```bash
cargo run -p candle-examples --example rust-candle-tour -- --cpu -n 0
```

你应该看到缺少 q_proj 权重的错误。

### 修回来

恢复权重名。

真实模型加载时，常见问题包括：

- HuggingFace checkpoint 的命名和模型代码不一致。
- 量化模型的权重名后缀不同。
- 模型结构配置和权重 shape 不一致。
- rope scaling、GQA、num heads 配置和 checkpoint 不匹配。

排查顺序：

1. 打印模型代码期望的路径。
2. 列出权重文件实际包含的路径。
3. 对比 shape。
4. 再看 dtype/device。

### 达标问题

1. `vb.pp("a").pp("b")` 会修改原来的 `vb` 吗？
2. 为什么 `from_tensors` 要取得 `HashMap` 的所有权？
3. 权重路径错误属于编译期错误还是运行期错误？

## 实验 7：KV Cache 的正确性不是“结果一样”这么简单

### 目标

理解 cached decode 和 full recompute 的关系。

### 运行

```bash
cargo run -p candle-examples --example rust-candle-tour -- --cpu --compare-cache -n 4
```

你会看到 cached 和 recompute-all 两条路径：

```text
cache comparison:
  cached step=00 input=[1, 5, 9, 2] -> token=...
  cached step=01 input=[...] -> token=...
  ...
  recompute-all step=00 input=[1, 5, 9, 2] -> token=...
  recompute-all step=01 input=[1, 5, 9, 2, ...] -> token=...
  ...
cache comparison result: token sequences match
```

注意：这个 tiny 模型太小，计时容易被调度、编译产物热身、stdout、CPU cache 等噪声淹没。这里的
`elapsed` 和 `token/s` 只用于观察两条路径被执行了，不用于证明 cached 一定更快。真实长上下文
模型中，KV cache 的收益来自避免重复计算历史 K/V。

### 观察

两条路径的目标不同：

- cached：第一步处理完整 prompt，后续只处理新 token，复用历史 K/V。
- recompute-all：每一步都重新处理完整上下文。

在 greedy sampling 下，如果 causal mask、index_pos、cache 拼接都正确，两者应生成同样 token。

为什么限制 greedy？

因为随机采样会消耗 RNG 状态。即使 logits 一样，只要采样路径细节不同，也可能让随机数推进方式不同。这个示例为了把对比聚焦在 cache 正确性上，所以 `--compare-cache` 要求 `Sampling::ArgMax`。

### 动手改

临时把 cached 模式里：

```rust
index_pos += ctxt.len();
```

改成：

```rust
index_pos += 1;
```

再运行：

```bash
cargo test -p candle-examples --example rust-candle-tour cached_generation_matches_full_recompute_for_greedy
```

根据 prompt 长度和模型细节，测试可能出现 token 不一致，或者 cache_len 仍一致但 mask/position 语义已经错了。

更可靠的理解是：`index_pos` 是当前输入在完整序列里的起始位置。

prefill：

```text
tokens=[1,5,9,2]
ctxt=[1,5,9,2]
index_pos=0
处理后 index_pos=4
```

decode：

```text
tokens=[1,5,9,2,next]
ctxt=[next]
index_pos=4
处理后 index_pos=5
```

### 修回来

恢复：

```rust
index_pos += ctxt.len();
```

### 达标问题

1. cache 的 K/V 是沿哪个维度拼接？
2. cached decode 每步输入 shape 是什么？
3. full recompute 为什么每步 `index_pos` 都是 0？

## 实验 8：causal mask 保护的是“不能看未来”

### 目标

理解 `build_causal_mask(seq_len, index_pos, device)` 的输入语义。

在 `TinyAttention::forward` 里：

```rust
let mask =
    candle_transformers::utils::build_causal_mask(seq_len, index_pos, x.device())?
        .broadcast_as(attention_scores.shape())?;
let minus_infinity =
    Tensor::new(f32::NEG_INFINITY, x.device())?.broadcast_as(mask.shape())?;
mask.where_cond(&minus_infinity, &attention_scores)?
```

### 观察

attention_scores 的 shape 是：

```text
[B, N, S, T]
```

mask 原始 shape 是：

```text
[S, T]
```

广播后变成：

```text
[B, N, S, T]
```

mask 的责任不是“让维度能对上”。它表达语义：

```text
第 i 个 query 只能看 <= 它绝对位置的 key。
```

prefill 时，`S > 1`，同一个 prompt 内部存在未来 token，所以必须 mask。

decode 时，`S == 1`，当前 query 位于序列末尾，历史 K/V 都是过去，通常不需要额外 mask。

### 动手改

临时把：

```rust
let attention_scores = if seq_len == 1 {
    attention_scores
} else {
    ...
};
```

改成无条件走 mask 分支。然后运行：

```bash
cargo test -p candle-examples --example rust-candle-tour
```

如果测试通过，说明这个小模型在 shape 上兼容无条件 mask；但你应该继续问：

- 额外构造 mask 有没有性能成本？
- `index_pos` 传错时，decode 会不会屏蔽掉不该屏蔽的位置？
- 真正模型里 RoPE 和 mask 是否使用同一套绝对位置语义？

### 修回来

恢复 decode 快路径。

### 达标问题

1. `S` 和 `T` 为什么不是同一个概念？
2. mask 为什么要 broadcast 到 scores 的 shape？
3. `f32::NEG_INFINITY` 在 softmax 前起什么作用？

## 实验 9：Module trait 统一了“层可以 forward”

### 目标

理解 `candle_nn::Module` 为什么能让 Embedding、Linear 用同样风格调用。

`model.rs` 中：

```rust
let hidden = self.token_embedding.forward(token_ids)?;
let attended = self.attention.forward(&hidden, index_pos, cache, trace_shapes)?;
let logits = self.lm_head.forward(&last_hidden)?.to_dtype(DType::F32)?;
```

其中 `Embedding` 和 `Linear` 来自 `candle-nn`，实现了 `Module`：

```rust
pub trait Module {
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}
```

`TinyAttention` 没实现这个 trait，因为它需要额外参数：

- `index_pos`
- `cache: &mut KvCache`
- `trace_shapes`

### 观察

这就是 trait 设计的边界：只有签名稳定、语义统一的东西才适合放进 trait。

如果强行让所有模块都实现同一个 `Module`：

```rust
fn forward(&self, xs: &Tensor) -> Result<Tensor>
```

那么 Attention 的 cache 和 position 就无处安放。你可能会被迫：

- 把 cache 放进 `self`，导致模型对象变成请求态。
- 用全局变量保存 position，导致并发请求互相污染。
- 做一个过大的上下文对象，所有层都依赖它。

当前示例的选择是：

- Linear/Embedding 用 `Module`。
- Attention 保持自己的显式方法。
- 请求状态通过 `&mut KvCache` 从外面传入。

### 动手改

不要真的提交这个改动，只做思考：

如果把 `KvCache` 放进 `TinyAttention` 里，`TinyCausalLm::forward` 就可以不传 `&mut cache`。看起来调用更短，但代价是：

- 同一个模型实例不能安全服务多个请求。
- 每次新请求前必须清空模型内部状态。
- 测试更难，因为 forward 结果依赖对象历史。
- 并发时要加锁或复制模型。

### 达标问题

1. 为什么 `Module::forward` 用 `&self` 而不是 `&mut self`？
2. 哪些层适合实现 `Module`？
3. 为什么请求级状态不应该藏在模型权重对象里？

## 实验 10：clone 在 Tensor 上通常是句柄复制

### 目标

区分 Rust 的 `Clone` 语义和底层数据复制成本。

`model.rs` 中：

```rust
cache.kv = Some((k.clone(), v.clone()));
```

这里的 `clone()` 不是把整块 K/V 数据复制一遍。Candle 的 `Tensor` 是持有共享内部对象的句柄，clone 主要复制句柄和引用计数。

真正复制通常发生在：

- `contiguous()` 需要把非连续 view 打包。
- device-to-host 的 `to_vec*`。
- dtype/device 转换。
- 某些后端为了满足 kernel 输入要求做内部转换。

### 观察

为什么这里必须 clone？

局部变量 `k`、`v` 还要继续参与 attention：

```rust
let context = probabilities.matmul(&v)?;
```

同时 cache 也要保存它们给下一步 decode 用。

如果不 clone，就会面临所有权问题：

- 把 `k` move 进 cache 后，局部就不能再用。
- 用引用存在 cache 里会引入生命周期问题，因为局部 `k` 很快离开作用域。

共享句柄正好表达了真实语义：cache 和当前 forward 都引用同一份 Tensor 数据。

### 动手改

临时改成：

```rust
cache.kv = Some((k, v));
```

然后保留后面对 `v` 的使用，运行：

```bash
cargo test -p candle-examples --example rust-candle-tour
```

你应该看到 move 之后继续使用的编译错误。

### 修回来

恢复：

```rust
cache.kv = Some((k.clone(), v.clone()));
```

### 达标问题

1. `Tensor::clone()` 和 `Tensor::contiguous()` 哪个更可能复制底层数据？
2. 为什么 cache 不能保存 `&Tensor`？
3. Rust 的 clone 是否总是便宜？

## 实验 11：backend 分发不是魔法

### 目标

看懂一次 `matmul` 从 Tensor API 到后端实现的大致路径。

### 运行

```bash
cargo run -p candle-examples --example rust-candle-tour -- --cpu --show-backend-path -n 0
```

输出：

```text
backend tour: [2, 2] x [2, 2] = [[19.0, 22.0], [43.0, 50.0]]
```

### 观察

`backend_walkthrough.rs` 已把源码路径写在文件头：

```text
candle-core/src/tensor.rs
  Tensor::matmul
candle-core/src/storage.rs
  Storage::matmul
具体 backend
  cpu_backend / cuda_backend / metal_backend
```

阅读源码时不要一头扎进 kernel。先确认外层合同：

- 输入 rank 是否允许？
- batch 维如何广播？
- 左右矩阵的 k 是否相同？
- dtype 是否支持？
- 两个 Tensor 是否在同一个 device？
- 输出 layout 是什么？
- autograd 记录了什么？

然后再看后端。

### 动手改

临时把 `backend_walkthrough.rs` 里的 `b` 改成 2x3：

```rust
let b = Tensor::new(&[[5f32, 6.0, 7.0], [8.0, 9.0, 10.0]], device)?;
```

然后测试：

```bash
cargo test -p candle-examples --example rust-candle-tour demo_matmul_matches_manual_product
```

你会看到测试失败，因为期望值仍是 2x2 乘法结果。

然后自己算出新的 2x3 结果，并更新测试。这个练习训练的是：

- shape 从 `[2,2] @ [2,3]` 到 `[2,3]`。
- 测试应该验证 shape 和数值。
- 示例代码的教学输出要跟测试保持一致。

最后恢复原状或把测试一起正确更新。

### 达标问题

1. `a.matmul(&b)` 为什么不消费 `b`？
2. GPU 上 `to_vec2` 隐含什么成本？
3. eager execution 和静态图执行的调试方式有什么不同？

## 实验 12：测试应该守住语义，不只守住行覆盖率

### 目标

理解当前测试为什么按行为组织，而不是按函数机械覆盖。

当前测试大致分为五类：

- CLI 输入解析：空 prompt、缺 token、越界 token。
- Sampling 参数：temperature、top-k、top-p 组合。
- 配置不变量：hidden size 和 head 数。
- Tensor/layout 示例：stride、offset、contiguous。
- 推理语义：prefill/decode cache 长度，以及 cached/recompute token 一致性。

### 运行

```bash
cargo test -p candle-examples --example rust-candle-tour -- --nocapture
```

### 观察

`-- --nocapture` 会显示测试里的 stdout。当前测试不太依赖打印，所以输出不会很多。

更重要的是测试命名：

```rust
fn cached_generation_matches_full_recompute_for_greedy() -> Result<()>
```

这个名字说明了合同：

```text
在 greedy sampling 下，KV cache 路径和全量重算路径应该生成同样 token。
```

如果未来改 attention 或 mask，这个测试能提醒你：性能优化不能改变语义。

### 动手改

新增一个测试前，先写一句话：

```text
我要保护的行为是：____
```

如果这句话写不出来，就不要急着写测试。否则很容易写出只验证实现细节的测试。

可选练习：

在 `main.rs` 测试模块里增加一个测试，验证 temperature 为 0 时，即使传了 `top_k/top_p`，也会走 greedy。

提示：

```rust
assert_eq!(
    build_sampling(0.0, Some(5), Some(0.9))?,
    Sampling::ArgMax
);
```

这个测试应该放在 `sampling.rs` 更合适，因为行为属于采样参数构造，不属于生成循环。

### 达标问题

1. 为什么 cached/recompute 对比只能在 greedy 下稳定？
2. 哪些错误应该用单元测试，哪些更适合集成测试？
3. 测试是否应该断言完整 stdout？

## 实验 13：读真实 LLaMA 源码时只追一条线

### 目标

把 tiny tour 里的概念迁移到真实 Candle LLaMA。

推荐只追这一条线：

```text
candle-examples/examples/llama/main.rs
  -> model.forward(...)
  -> candle-transformers/src/models/llama.rs
  -> Attention::forward
  -> cache update / RoPE / repeat_kv / matmul / softmax
```

不要同时追 tokenizer、权重下载、量化、CUDA kernel、chat template。一次只追一条系统不变量。

### 阅读任务 A：找模型态和请求态

在真实 LLaMA 代码里标出：

- 哪些字段是模型权重。
- 哪些字段是配置。
- 哪些字段是 cache。
- 哪些参数是本次 forward 的输入。

然后回答：

```text
如果我要并发服务两个请求，哪些对象可以共享，哪些必须按请求拆开？
```

### 阅读任务 B：追 shape

从 token ids 开始，写出每一步 shape：

```text
[B,S]
[B,S,H]
[B,S,num_heads,head_dim]
[B,num_heads,S,head_dim]
...
```

遇到 GQA 时额外标出：

```text
num_attention_heads != num_key_value_heads
```

然后解释 `repeat_kv` 为什么存在。

### 阅读任务 C：追错误边界

每看到一个 `?`，问：

- 这里可能失败的原因是什么？
- 错误应该让当前请求失败，还是应该让整个服务退出？
- 如果放在 HTTP server 中，这个错误应映射成 400、500，还是启动失败？

### 达标问题

1. tiny tour 里没有 RoPE，真实 LLaMA 里 RoPE 依赖什么位置语义？
2. GQA 如何改变 K/V cache 的 head 维？
3. 为什么模型加载错误和单次请求 shape 错误应分开处理？

## 实验 14：做一次小型重构，但不改变语义

### 目标

练习 Rust 里“小步安全改代码”的节奏。

本分支已经把 `run_generation` 中计算 `(context_size, forward_index_pos, stage)` 的逻辑抽成
了一个小函数。你的任务是读懂它，并尝试扩展一个边界测试。

目标签名可以是：

```rust
fn generation_step_context(
    mode: GenerationMode,
    step: usize,
    token_count: usize,
    index_pos: usize,
) -> (usize, usize, &'static str)
```

### 要求

先保持这些测试通过：

```bash
cargo test -p candle-examples --example rust-candle-tour
```

已有测试覆盖：

- cached 第一步：`context_size = prompt_len`，`index_pos = 0`，stage 是 prefill。
- cached decode：`context_size = 1`，`index_pos` 不变传出，stage 是 decode。
- recompute-all：`context_size = token_count`，`forward_index_pos = 0`，stage 是 recompute。

### 动手改

这个函数不需要借用 `tokens`。它只接收长度和位置，所以测试更容易写。

这是 Rust 重构里的常见技巧：

- 能传标量就别传整个对象。
- 能返回纯值就别修改外部状态。
- 把借用范围留在真正需要数据的地方。

本分支也已经有一个边界测试：`sample_len = 0` 时，`run_generation` 不进入循环，输出 token
等于 prompt，cache_lens 为空。这个测试保护的是“零长度生成不会偷偷 prefill”。

你可以继续扩展一个新测试：当 prompt 长度从 4 变成 1 时，cached 第一步仍然是 prefill，
context_size 应该等于 1，而不是走 decode。

### 达标问题

1. 新函数是否需要返回 `Result`？
2. 为什么它可以返回 `&'static str`？
3. 抽函数后，`ctxt` 的借用范围是否改变？

## 实验 15：建立自己的源码阅读记录格式

### 目标

把学习过程变成可复用记录，而不是一次性阅读。

建议每读一个函数，用下面格式记录：

```text
函数：
输入：
输出：
持有的状态：
借用的状态：
可能失败的地方：
shape 输入：
shape 输出：
device/dtype 假设：
性能敏感点：
测试或命令：
我还不确定：
```

示例：

```text
函数：TinyAttention::forward
输入：x [B,S,H]，index_pos，&mut KvCache，trace_shapes
输出：Tensor [B,S,H]
持有的状态：q/k/v/o projection 权重
借用的状态：x，cache
可能失败的地方：dims3、linear forward、reshape、transpose、cat、matmul、mask、softmax
shape 输入：[B,S,H]
shape 输出：[B,S,H]
device/dtype 假设：所有 Tensor 在同一 device，权重 dtype 能参与 matmul
性能敏感点：q/k/v contiguous，K/V cat，scores matmul，softmax
测试或命令：--trace-shapes，forward_extends_cache_from_prefill_to_decode
我还不确定：真实 LLaMA 中 RoPE 插入位置和 repeat_kv 的 shape 细节
```

### 达标问题

1. 你能不能用这个格式记录真实 `Attention::forward`？
2. 你能不能标出哪个状态属于模型、哪个状态属于请求？
3. 你能不能为“不确定”的点设计一个最小验证命令或测试？

## 实验 16：EOS 停止逻辑是生成循环的一部分

### 目标

理解 `--eos-token <u32>` 为什么适合用 `Option<u32>`，以及停止检查应该放在采样后的哪个位置。

### 运行

```bash
cargo run -p candle-examples --example rust-candle-tour -- --cpu --eos-token 22 -n 8
```

默认 greedy 序列中会先生成 23，再生成 22。因为 22 被指定为 EOS，生成循环会把 22 放进输出
序列后停止：

```text
generation:
  cached step=00 input=[1, 5, 9, 2] -> token=23 cache_len=4
  cached step=01 input=[23] -> token=22 cache_len=5
stopped on eos token: 22
all token ids: [1, 5, 9, 2, 23, 22]
```

### 观察

CLI 字段是：

```rust
eos_token: Option<u32>
```

它表达两种状态：

```text
None：没有用户指定的停止 token，只按 sample_len 停止
Some(id)：采样到这个 token 后停止
```

进入生成前会校验：

```rust
validate_optional_token("eos token", args.eos_token, config.vocab_size)?;
```

停止逻辑在采样之后、下一轮之前：

```rust
tokens.push(next_token);
if eos_token == Some(next_token) {
    stopped_on_eos = Some(next_token);
    break;
}
```

这个顺序很重要。EOS 通常是模型实际生成出来的 token，应保留在输出序列中。停止检查如果放在
`tokens.push(next_token)` 之前，就要另外决定是否丢弃 EOS；那会让输出语义变得不清楚。

### 错误路径

运行：

```bash
cargo run -p candle-examples --example rust-candle-tour -- --cpu --eos-token 32 -n 1
```

本例词表范围是 `0..32`，所以 32 越界，应该在构造模型输入前失败。

### 和 cache 对比结合

运行：

```bash
cargo run -p candle-examples --example rust-candle-tour -- --cpu --compare-cache --eos-token 22 -n 8
```

cached 和 recompute-all 都应在生成 22 后停止，并得到同样 token 序列。这个对比同时保护三件事：

- logits 语义一致。
- 停止状态一致。
- 早停后性能统计使用实际生成 token 数，而不是原始 `sample_len`。

### 动手改

临时把停止检查移动到 `tokens.push(next_token)` 前面，然后更新测试预期。你会发现需要明确回答：

```text
EOS token 是否应该出现在 all token ids 里？
```

真实 LLM runner 里也必须做这个产品语义决策，不能让实现顺序偶然决定输出格式。

### 达标问题

1. 为什么 `eos_token` 用 `Option<u32>`，而不是默认设置成 0？
2. EOS token 越界为什么应该在生成前报错？
3. 命中 EOS 后，cache_lens 的长度表示什么？
4. 真实 tokenizer 里 EOS 为什么可能需要 enum，而不是单个 `u32`？

## 最后一轮自测

不看文档，回答下面问题：

1. `Tensor` 的 shape、stride、start offset 分别解决什么问题？
2. 为什么 `KV Cache` 不能放在全局静态变量里？
3. `&Tensor`、`Tensor`、`Tensor::clone()`、`Tensor::contiguous()` 在所有权和成本上分别意味着什么？
4. `VarBuilder` 的路径为什么像模型代码和 checkpoint 之间的 ABI？
5. cached decode 和 full recompute 的正确性关系是什么？
6. 为什么 `Result` 在 Candle 代码里不是噪音，而是系统边界？
7. 如果你要把 tiny tour 改成双层 attention，哪些测试必须新增或修改？
8. 如果未来打开 CUDA feature，哪些地方的语义应保持不变，哪些地方的性能和错误类型可能变化？
9. EOS 检查放在采样前、采样后 push 前、采样后 push 后，分别会改变什么语义？

## 推荐迭代顺序

第一轮只跑命令：

```bash
cargo run -p candle-examples --example rust-candle-tour -- --cpu --trace-shapes -n 2
cargo run -p candle-examples --example rust-candle-tour -- --cpu --show-layout-path -n 0
cargo run -p candle-examples --example rust-candle-tour -- --cpu --show-weight-paths -n 0
cargo run -p candle-examples --example rust-candle-tour -- --cpu --compare-cache -n 4
cargo run -p candle-examples --example rust-candle-tour -- --cpu --eos-token 22 -n 8
```

第二轮做可恢复的小破坏：

```text
删 prompt 校验
删 top-k 校验
提前 tokens.push
改 hidden_size
改权重路径
改 index_pos 推进
把 EOS 检查移动到 tokens.push 前
```

第三轮做真正重构：

```text
抽 generation_step_context
给它补测试
保持所有测试通过
用 --trace-shapes 确认输出语义没变
```

第四轮迁移到真实源码：

```text
只追真实 LLaMA 的 Attention::forward
只记录 shape、cache、position 三条不变量
不要同时研究 tokenizer、下载、量化和 kernel
```

学到这里，Rust 不再只是语法；它开始变成推理系统的约束语言。
