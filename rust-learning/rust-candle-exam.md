# Rust + Candle 水平测评题

这份题用来测试你现在的 Rust 和 Candle 源码阅读能力。题目正文不附标准答案；做完后再看
[`评分参考与答案要点`](./rust-candle-exam-grading-guide.md)，或者把答案写成一份 Markdown
交回来逐项批改。

建议闭卷作答。允许查看本仓库源码，但不要看学习讲义。总分 100 分，建议 120 分钟完成。

作答格式：

```text
姓名/代号：
用时：
是否运行过实操命令：

A1:
A2:
...

实操题改了哪些文件：
遇到的编译器错误：
你如何修复：
```

评分等级：

| 分数 | 水平判断 |
| --- | --- |
| 90-100 | 能独立阅读并小范围修改 Candle 推理代码 |
| 75-89 | Rust 基础和 Tensor 推理主线扎实，复杂工程边界还需练 |
| 60-74 | 能读懂主流程，但所有权、layout、trait 或 cache 仍有盲点 |
| 40-59 | 能运行示例，但还不能稳定解释源码行为 |
| 0-39 | 需要回到第 0-3 课重新打基础 |

---

## A. Rust 基础与所有权（20 分）

### A1. `let`、`mut`、move、borrow（4 分）

阅读代码：

```rust
let mut tokens = vec![1u32, 5, 9, 2];
let ctxt = &tokens[1..];
tokens.push(23);
println!("{ctxt:?}");
```

问题：

1. 这段代码能否编译？
2. 如果不能，错误的本质是什么？
3. 给出一种不复制整段 `tokens` 的修改方式。
4. 用 C++ 的 `std::vector` 和 `std::span` 类比说明这个 bug 在 C++ 中为什么危险。

评分点：

- 能指出 `push` 可能重新分配。
- 能说明 `ctxt` 借用了 `tokens`。
- 能给出让借用在 `push` 前结束的修改。

### A2. `Option<T>` 的移动语义（4 分）

阅读代码：

```rust
let name: Option<String> = Some("llama".to_string());
match name {
    Some(value) => println!("{value}"),
    None => println!("none"),
}
println!("{name:?}");
```

问题：

1. 这段代码能否编译？
2. `Option<usize>` 换成同样写法为什么行为可能不同？
3. 如何只借用 `String` 而不移动？
4. `as_ref()` 和 `match &name` 各自表达什么？

评分点：

- 区分 `Copy` 和非 `Copy`。
- 能解释 `Some(value)` 会移动内部值。
- 能写出借用匹配方式。

### A3. `Result`、`?`、`bail!`（4 分）

写一个函数签名和核心代码，完成：

```text
输入 &str，例如 "1,5,9"
输出 Vec<u32>
空字符串报错
任一 token 解析失败报错
```

要求：

1. 函数返回 `anyhow::Result<Vec<u32>>`。
2. 至少用一次 `?`。
3. 至少用一次 `bail!`。
4. 解释 `?` 展开的控制流含义。

评分点：

- 签名正确。
- 错误路径清晰。
- 不用 `unwrap()`。

### A4. `match` 是表达式（4 分）

阅读代码：

```rust
let value = match mode {
    Mode::Fast => 1,
    Mode::Slow => {
        println!("slow");
        2
    }
};
```

问题：

1. 为什么 `Mode::Slow` 分支最后的 `2` 后面不能加分号？
2. 如果新增 `Mode::Auto` 但不改这个 `match`，编译器会怎样？
3. `match` 和 C++ `switch` 的关键差异是什么？
4. 举一个 Candle tour 中 `match` 同时完成“分支选择 + 解构”的例子。

评分点：

- 能解释表达式返回值。
- 能解释穷尽检查。
- 能联系实际代码。

### A5. 闭包和 `collect::<Result<Vec<_>>>()?`（4 分）

解释下面这行的类型流：

```rust
let blocks = (0..n)
    .map(|i| Block::load(vb.pp(format!("model.layers.{i}")), cfg))
    .collect::<Result<Vec<_>>>()?;
```

问题：

1. `map` 后 iterator 的 item 类型是什么？
2. `collect::<Result<Vec<_>>>()` 做了什么？
3. 遇到第一个错误时会发生什么？
4. 为什么这比 `.unwrap()` 更适合模型加载？

评分点：

- 能说出 item 是 `Result<Block>`。
- 能解释短路错误传播。
- 能说明模型加载不能静默失败。

---

## B. Candle Tensor 与后端（20 分）

### B1. Tensor 五要素（4 分）

对下面 Tensor：

```rust
let x = Tensor::arange(0f32, 24f32, &Device::Cpu)?.reshape((2, 3, 4))?;
```

回答：

1. shape 是什么？
2. contiguous stride 是什么？
3. dtype 是什么？
4. device 是什么？
5. Storage 和 Layout 分别负责什么？

评分点：

- stride 手算正确。
- 能区分 Storage 和 Layout。

### B2. view 与 copy（4 分）

判断下列操作通常是共享句柄、view、可能 copy 还是必然新 Storage：

```rust
let b = x.clone();
let b = x.transpose(1, 2)?;
let b = x.narrow(1, 1, 2)?;
let b = x.transpose(1, 2)?.contiguous()?;
let b = x.to_dtype(DType::F16)?;
```

要求：

1. 逐行判断。
2. 说明 `Tensor::clone()` 和 `Vec::clone()` 的成本差异。
3. 说明为什么 attention 里常见 `.contiguous()?`。

评分点：

- 复制语义判断正确。
- 不把所有 clone 都当成深拷贝。

### B3. `IndexOp` 和 shape 变化（4 分）

设：

```rust
let x = Tensor::zeros((2, 3, 4), DType::F32, &Device::Cpu)?;
```

回答以下表达式的输出 shape：

```rust
x.i((.., 2, ..))?;
x.i((0, .., ..))?;
x.i((.., 1..3, ..))?;
x.i((.., .., 0))?;
```

再解释：

```rust
hidden.i((.., seq_len - 1, ..))?.contiguous()?
```

为什么用于取最后位置 hidden。

评分点：

- rank 变化判断正确。
- 能解释具体 index 会去掉维度。

### B4. matmul 后端路径（4 分）

以：

```rust
let c = a.matmul(&b)?;
```

回答：

1. `Tensor::matmul` 负责哪些检查？
2. 为什么会进入 `Storage::matmul`？
3. `Storage::matmul` 如何区分 CPU/CUDA/Metal？
4. 返回的新 Tensor 至少包含哪些信息？
5. eager execution 在这里是什么意思？

评分点：

- 能说出 Tensor -> Storage -> backend 层次。
- 能区分运行时 enum dispatch 和 trait backend。

### B5. Device/DType 边界（4 分）

回答：

1. Cargo feature `cuda` 和运行时 `Device::new_cuda(0)?` 有什么区别？
2. `Tensor::new(&[1u32, 2], &device)?` 的 dtype 从哪里来？
3. GPU Tensor 调 `to_vec1::<f32>()?` 可能发生什么？
4. 为什么 logits 采样前通常转成 F32？

评分点：

- 编译期和运行时边界清楚。
- host/device 拷贝意识清楚。

---

## C. 模型加载与神经网络抽象（15 分）

### C1. `Module` trait（3 分）

解释：

```rust
pub trait Module {
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}
```

问题：

1. 为什么 `Linear`、`Embedding` 适合实现它？
2. 为什么完整 LLaMA forward 往往还要自定义签名？
3. `fn run<M: Module>` 和 `fn run(m: &dyn Module)` 有什么分发差异？

### C2. VarBuilder 参数路径（4 分）

给定：

```rust
let vb = vb.pp("model.layers.3").pp("self_attn");
let q = linear_no_bias(hidden, hidden, vb.pp("q_proj"))?;
```

回答：

1. 最终会查找哪个权重名？
2. `pp` 会不会修改原 builder？
3. 权重名存在但 shape 错误时应该静默继续还是报错？
4. 为什么 VarBuilder 适合隐藏 mmap、buffer、HashMap、VarMap 等不同权重来源？

### C3. Serde config（3 分）

设计一个 `TinyConfig` 的 JSON 反序列化结构：

```text
vocab_size: 必填
hidden_size: 必填
num_heads: 必填
num_key_value_heads: 可选
```

要求：

1. 写出 struct 字段。
2. 标出哪些字段用 `Option<T>`。
3. 写出至少两个 `validate()` 应检查的条件。

### C4. SafeTensors 与 mmap（3 分）

回答：

1. `config.json`、`tokenizer.json`、`*.safetensors` 分别保存什么？
2. mmap SafeTensors 的目的是什么？
3. 为什么 mmap API 可能是 unsafe？

### C5. trait object 与生命周期（2 分）

解释：

```rust
pub type VarBuilder<'a> = VarBuilderArgs<'a, Box<dyn SimpleBackend + 'a>>;
```

问题：

1. `Box<dyn SimpleBackend>` 表示什么？
2. `'a` 在这里大致约束什么？

---

## D. LLaMA 推理、Attention、KV Cache（20 分）

### D1. prefill/decode 主循环（4 分）

给定 prompt tokens 长度为 4，生成 3 个 token。

回答 cached decode 模式每一步：

| step | 输入 token 数 | `index_pos` | cache 长度 |
| --- | --- | --- | --- |
| 0 | ? | ? | ? |
| 1 | ? | ? | ? |
| 2 | ? | ? | ? |

再回答 recompute-all 模式每一步输入 token 数。

评分点：

- prefill 和 decode 区分正确。
- cached 和 recompute-all 区分正确。

### D2. Attention shape 手算（5 分）

设：

```text
B=1
S=4
H=16
N=4
D=H/N
```

回答：

1. Embedding 输出 shape。
2. Q reshape + transpose 后 shape。
3. K/V 新增 shape。
4. prefill attention score shape。
5. 生成一个 token 后，decode attention score shape。

### D3. causal mask（3 分）

回答：

1. prefill 为什么需要 causal mask？
2. decode 时 `S=1`，为什么通常可以省略“当前输入内部未来 token”的 mask？
3. 如果一次 decode 多个新 token，mask 问题会不会变化？

### D4. KV Cache 语义（4 分）

回答：

1. KV Cache 保存什么？
2. 为什么每个请求必须独立 cache？
3. 为什么教学示例用 `Tensor::cat`，生产系统通常不会这么做？
4. GQA 为什么能降低 KV Cache 内存和带宽？

### D5. sampling 与 EOS（4 分）

回答：

1. logits 到 next token 的基本流程。
2. temperature/top-k/top-p 分别影响什么。
3. repeat penalty 读取哪一段历史 token？
4. EOS 可能是单 token 或多 token 时，为什么 enum 比裸 `Option<u32>` 更合适？

---

## E. 工程化、并发、unsafe（15 分）

### E1. Cargo feature 与 `cfg`（3 分）

解释下面两条命令的区别：

```bash
cargo run -p candle-examples --example llama --features metal -- --sample-len 8
cargo run -p candle-examples --example llama -- --features metal --sample-len 8
```

再说明 `#[cfg(feature = "metal")]` 什么时候生效。

### E2. tracing 与性能指标（3 分）

回答：

1. 为什么只看总 token/s 不够？
2. TTFT、prefill、decode token/s 分别衡量什么？
3. GPU 异步执行会怎样影响计时？

### E3. 服务化所有权设计（4 分）

设计一个最小推理服务状态，要求支持两个并发请求。

必须说明哪些共享，哪些每请求独立：

```text
Model
Device
Tokenizer
tokens
KvCache
sampler/RNG
cancel flag
```

评分点：

- 权重共享。
- cache/RNG/tokens 独立。
- 不把整个 forward 锁成串行。

### E4. `Arc<RwLock<T>>`（2 分）

回答：

1. `Arc<T>` 解决什么？
2. `RwLock<T>` 解决什么？
3. 为什么 `RwLock` 不是“绕过借用检查”？

### E5. unsafe 审计（3 分）

任选一个：

- mmap SafeTensors。
- 未初始化 Tensor Storage 分配。
- FFI handle wrapper。

填写：

```text
unsafe operation:
safety invariant:
who guarantees it:
what happens if violated:
safe wrapper boundary:
```

---

## F. 实操题（10 分）

从下面三题任选两题，每题 5 分。要求能编译、能测试或能用命令验证。

### F1. 为 tiny tour 扩展 layout trace

目标：

```text
--trace-shapes 时在已有 shape/dtype 基础上，同时打印 contiguous 状态。
```

要求：

1. 修改 trace helper。
2. 输出至少包含 token ids、Q、last hidden、logits 的 contiguous 状态。
3. 跑通：

```bash
cargo run -p candle-examples --example rust-candle-tour -- --cpu --trace-shapes -n 1
```

评分点：

- 不破坏现有 shape/dtype 输出。
- 能解释哪些 Tensor 因为 `contiguous()` 明确变成连续布局。

### F2. 把 tiny tour 的单 EOS 扩展为多 EOS

目标：

```text
在现有 --eos-token <u32> 基础上，新增 --eos-tokens <csv>，任一 token 命中后停止。
```

要求：

1. 参数类型能表达“未设置、单 token、多 token”。
2. 复用或抽象已有 token 解析和 vocab 校验逻辑。
3. 停止逻辑仍发生在采样后、下一轮前，并把命中的 EOS 保留在输出里。
4. 增加至少两个测试：越界 EOS 失败；多 EOS 中任一命中会停止。

评分点：

- 状态表达清晰，不用裸默认值假装 None。
- 不把多 EOS 解析散落在生成循环里。

### F3. 给 layout walkthrough 增加一个 view 案例

目标：

新增一个快照：

```rust
base.narrow(0, 1, 1)
```

要求：

1. 先写出你预测的 dims、stride、offset、contiguous。
2. 修改 `layout_walkthrough.rs`。
3. 更新测试。
4. 跑通：

```bash
cargo test -p candle-examples --example rust-candle-tour layout
```

评分点：

- stride/offset 手算正确。
- 测试覆盖新增案例。

---

## G. 加分题（最多 10 分）

### G1. 解释一个真实 LLaMA 代码片段（5 分）

从 `candle-transformers/src/models/llama.rs` 任选一个函数，例如：

```text
CausalSelfAttention::forward
Block::forward
Llama::forward
```

写一段解释：

1. 输入 Tensor shape。
2. 输出 Tensor shape。
3. 读了哪些权重。
4. 修改了哪些状态。
5. 可能在哪些地方分配或复制。

### G2. 设计一个最小 LLM runner（5 分）

只写设计，不要求实现。

要求说明：

1. CLI 参数。
2. 模型加载。
3. tokenizer。
4. generation loop。
5. KV Cache 生命周期。
6. 错误处理。
7. 测试策略。

---

## 交卷自查清单

交卷前确认：

- 没有用 `unwrap()` 回避题目要求的错误处理。
- 每个 `Option` 都解释了 `None` 的语义。
- 每个 `&mut` 都解释了要修改什么状态。
- 每个 Tensor 问题都同时考虑 shape 和 layout。
- 每个性能结论都有指标支撑。
- 每个 unsafe 问题都写了不变量。
- 实操题跑过命令，并记录输出摘要。

把答案贴回来时，我会按上面的分值逐项批改，并指出你下一步最该补的 Rust/Candle 知识点。
