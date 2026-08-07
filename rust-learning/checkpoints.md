# Rust + Candle 学习自检表

这份文档用来判断“我是不是真的懂了”。每一课都按同一结构：

- 你要能解释什么。
- 你要能跑什么命令。
- 你要能改什么小代码。
- 常见误解是什么。
- 达标标准是什么。

不要只读讲义。每一课至少做一次小验证。

---

## 第 0 课：Candle 工程地图

你要能解释：

1. `candle-core`、`candle-nn`、`candle-transformers`、`candle-examples` 各自负责什么。
2. 一次 LLaMA 推理为什么会跨越 example、model、nn、core、backend。
3. Cargo workspace 和 CMake 顶层工程的相似与差异。
4. feature 是编译期能力，Device 是运行时选择。

要跑：

```bash
cargo run -p candle-examples --example rust-candle-tour -- --cpu --show-backend-path -n 0
```

小改动：

在 `backend_walkthrough.rs` 里换一组 2x2 矩阵，并同步测试期望结果。

常见误解：

- 以为 Candle 只是模型示例集合。
- 以为 `--features cuda` 和 `--cpu` 是同一类参数。
- 试图从 `candle-core/src/lib.rs` 第一行开始通读。

达标标准：

你能画出：

```text
CLI -> model.forward -> nn layers -> Tensor -> Storage -> backend
```

---

## 第 1 课：入口中的 Rust 基础

你要能解释：

1. 为什么 `let` 默认不可变。
2. `Option<T>` 为什么比空指针更明确。
3. `Result<T>` 和 `?` 如何传播错误。
4. `&[u32]` 为什么只是借用 `Vec<u32>` 的一段。
5. 为什么要先最后一次使用 `ctxt`，再 `tokens.push(next_token)`。

要跑：

```bash
cargo test -p candle-examples --example rust-candle-tour parse_prompt
```

小改动：

把 `tokens.push(next_token)` 移到打印 `ctxt` 之前，观察编译器如何阻止潜在悬空 slice。

常见误解：

- 把 `&[T]` 当成拥有数据的容器。
- 以为 `?` 是异常。
- 以为 `Option<T>` 只是 nullable 的语法糖。

达标标准：

你能看到一个函数签名，就说出它会不会消费值、会不会修改状态、会不会返回错误。

---

## 第 2 课：采样与 enum

你要能解释：

1. `Sampling` enum 每个 variant 对应什么策略。
2. `match (top_k, top_p)` 为什么能穷尽四种组合。
3. `temperature == 0` 为什么走 greedy/argmax。
4. `LogitsProcessor::sample` 为什么需要 `&mut self`。
5. 随机采样和 greedy 对测试稳定性的影响。

要跑：

```bash
cargo test -p candle-examples --example rust-candle-tour sampling
cargo run -p candle-examples --example rust-candle-tour -- --cpu --temperature 0.7 --top-k 5 --top-p 0.9 -n 4
```

小改动：

给 `build_sampling` 加一个非法 temperature 的测试，检查错误消息包含 `non-negative`。

常见误解：

- 以为 enum 只能像 C++ enum 那样存整数。
- 忘记 top-k/top-p 都是可选参数。
- 用随机采样测试 cache 语义，导致结果受 RNG 干扰。

达标标准：

你能从命令行参数推出准确的 `Sampling` variant。

---

## 第 3 课：Tensor、Storage、Layout

你要能解释：

1. Tensor 不是一段裸内存，而是 Storage + Shape + Layout + DType + Device。
2. `clone()`、`transpose()`、`narrow()`、`contiguous()` 的复制语义差异。
3. stride 和 start offset 如何描述 view。
4. `Arc<RwLock<Storage>>` 为什么是内部可变性，不是绕过安全。

要跑：

```bash
cargo run -p candle-examples --example rust-candle-tour -- --cpu --show-layout-path -n 0
cargo test -p candle-examples --example rust-candle-tour layout
```

小改动：

在 `layout_walkthrough.rs` 里增加一个 `base.narrow(0, 1, 1)` 快照，并先手算 stride 和 offset。

常见误解：

- shape 一样就认为内存布局一样。
- 看到 `.clone()` 就认为复制了整块 Tensor 数据。
- 以为 `contiguous()` 永远复制。

达标标准：

你能看到一个 Tensor 操作后判断它通常是 view、共享句柄还是物理 copy。

---

## 第 4 课：神经网络抽象

你要能解释：

1. `struct` 保存层状态，`impl` 定义行为。
2. `Module::forward(&self, &Tensor)` 适合什么层。
3. 为什么完整 LLaMA forward 不一定实现简单 `Module` 签名。
4. 泛型 shape API 为什么能接收 tuple、array、Vec。
5. `Box<dyn SimpleBackend>` 为什么适合 VarBuilder。

要跑：

```bash
cargo test -p candle-examples --example rust-candle-tour model
```

小改动：

写一个 `Scale { factor: f64 }`，实现 `Module`，让它对 Tensor 做乘法，并加一个测试。

常见误解：

- 把 trait 当成必须继承的数据基类。
- 以为所有 forward 都必须是 `Tensor -> Tensor`。
- 看到泛型 shape API 就试图展开所有 trait 实现。

达标标准：

你能解释 `Embedding`、`Linear`、`TinyAttention`、`TinyCausalLm` 为什么是不同层级的抽象。

---

## 第 5 课：模型加载

你要能解释：

1. `config.json`、`tokenizer.json`、`*.safetensors` 各自负责什么。
2. Serde 如何把 JSON 转成强类型 config。
3. VarBuilder 如何从路径取权重。
4. `vb.pp(...)` 如何累积路径前缀。
5. mmap 为什么需要 unsafe。

要跑：

```bash
cargo run -p candle-examples --example rust-candle-tour -- --cpu -n 1
```

小改动：

把 `model.layers.0.self_attn.q_proj.weight` 的 demo 权重名故意改错，观察 VarBuilder 加载错误。

常见误解：

- 把 VarBuilder 当成模型层。
- 以为反序列化成功就代表模型维度一定合法。
- 忽略权重名称和 shape 校验。

达标标准：

你能从 `linear_no_bias(..., vb.pp("q_proj"))` 推导最终查找哪个 `.weight`。

---

## 第 6 课：LLaMA 推理全流程

你要能解释：

1. prefill 和 decode 的输入长度差异。
2. Q/K/V 的 shape 如何从 `[B, S, H]` 变成 head 维。
3. causal mask 为什么主要用于 prefill。
4. KV Cache 保存的是 K/V，不是 logits。
5. 为什么只取最后一个位置进 lm_head。

要跑：

```bash
cargo run -p candle-examples --example rust-candle-tour -- --cpu --trace-shapes --prompt 1,5,9,2 -n 2
cargo run -p candle-examples --example rust-candle-tour -- --cpu --compare-cache --prompt 1,5,9,2 -n 4
```

小改动：

在 trace 输出里加一行打印 `cache_len` 和 attention score 的最后一维 `T`，确认两者一致。

常见误解：

- 以为 decode 不看历史 token。
- 以为 cache 可以跨请求共享。
- 以为 tiny demo 的 token/s 能代表真实 LLM 性能。

达标标准：

你能手算一次 prefill 和一次 decode 中所有关键 Tensor 的 shape，并解释 token ids、hidden、logits
的 dtype 为什么不同。

---

## 第 7 课：工程化

你要能解释：

1. Cargo feature、`cfg`、runtime args 的边界。
2. 测试函数为什么可以返回 `Result<()>`。
3. tracing 能回答什么性能问题。
4. TTFT、prefill、decode token/s 的区别。
5. 服务里哪些状态共享，哪些必须请求独立。

要跑：

```bash
cargo test -p candle-examples --example rust-candle-tour
cargo run -p candle-examples --example rust-candle-tour -- --cpu --compare-cache -n 8
```

小改动：

给 `run_generation` 增加 prefill/decode 分段计时，注意不要把打印时间混入模型 forward 时间。

常见误解：

- 把所有状态放进一个 `Arc<Mutex<_>>`。
- 只看总 token/s。
- 改了 feature 却忘记它需要重新编译。

达标标准：

你能设计一个最小服务状态结构：共享模型权重，每请求独立 tokens/cache/sampler。

---

## 第 8 课：系统边界与 unsafe

你要能解释：

1. unsafe 允许哪些额外操作。
2. unsafe 不会关闭借用检查。
3. mmap 和未初始化分配分别有什么不变量。
4. backend trait 和 runtime enum dispatch 为什么同时存在。
5. FFI wrapper 如何用 Drop 管资源。

要跑：

```bash
cargo run -p candle-examples --example rust-candle-tour -- --cpu --show-backend-path -n 0
```

小改动：

不用真的写 unsafe。选一个 unsafe 调用，写出：

```text
unsafe operation:
safety invariant:
who guarantees it:
safe wrapper boundary:
```

常见误解：

- 以为 unsafe 就是不安全代码的任意通行证。
- 把 unsafe 范围写得过大。
- 不写 safety invariant。

达标标准：

你能审计一个 unsafe 边界，而不是只说“这里底层所以需要 unsafe”。

---

## 综合达标标准

学完后，你应该能独立完成：

1. 新增一个命令行参数，并说明它属于 Cargo 参数还是 example 参数。
2. 新增一个采样策略校验，并写测试。
3. 打印一个 Tensor 的 dims/stride/dtype/device。
4. 判断一个 Tensor 操作是否可能复制 Storage。
5. 从参数路径推导 SafeTensors 权重名。
6. 手算一次 attention 的 Q/K/V/cache/logits shape。
7. 比较 cached decode 和 recompute-all decode。
8. 为一个小功能补单元测试。
9. 解释一处 unsafe 的不变量。
10. 设计一个请求独立 cache 的推理服务状态。

如果这 10 项都能做到，说明你已经不是“看过 Rust/Candle 文章”，而是能开始改这个项目里的
真实代码。
