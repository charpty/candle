# Rust + Candle 水平测评题：评分参考与答案要点

这份文档配合 [`rust-candle-exam.md`](./rust-candle-exam.md) 使用。

建议顺序：

1. 先闭卷或半开卷完成考题。
2. 再用本文逐题对照。
3. 把没答扎实的题回到 [`rust-candle-labs.md`](./rust-candle-labs.md) 做实验。

评分时不要只看结论。一个答案能拿高分，必须同时说明：

- Rust 类型或所有权语义。
- Candle 推理系统里的实际后果。
- 如果改错了，编译器、测试或运行时错误会在哪里暴露。

## A. Rust 基础与所有权

### A1. `let`、`mut`、move、borrow

参考答案：

```rust
let mut tokens = vec![1u32, 5, 9, 2];
let ctxt = &tokens[1..];
tokens.push(23);
println!("{ctxt:?}");
```

这段不能编译。`ctxt` 是对 `tokens` 内部连续区间的共享借用，`tokens.push(23)` 需要对
`tokens` 做可变借用。更关键的是，`push` 可能触发 `Vec` 重新分配，使原来指向内部 buffer 的
slice 失效。Rust 禁止在共享借用仍会被使用时进行可变借用。

不复制整段 `tokens` 的一种修法：

```rust
let mut tokens = vec![1u32, 5, 9, 2];
{
    let ctxt = &tokens[1..];
    println!("{ctxt:?}");
}
tokens.push(23);
```

或者先把 `println!` 放在 `push` 前，让 `ctxt` 的最后一次使用早于 `push`。

C++ 类比：`std::span` 或引用指向 `std::vector` 内部元素时，`push_back` 可能导致 vector
reallocate，旧 span/引用/迭代器变成悬垂引用。C++ 允许你写出来，后果可能是未定义行为；Rust
在编译期拒绝。

满分要点：

- 指出不能编译。
- 指出 `ctxt` 借用 `tokens`。
- 指出 `push` 可能重新分配。
- 给出缩短借用生命周期的修法，而不是只说“clone 一份”。

常见扣分：

- 只说“Rust 不允许同时读写”，但不解释 `Vec` reallocation。
- 用 `tokens.clone()` 回避问题，却不说明复制代价。
- 把 `mut tokens` 理解成允许任意可变操作。

### A2. `Option<T>` 的移动语义

参考答案：

```rust
let name: Option<String> = Some("llama".to_string());
match name {
    Some(value) => println!("{value}"),
    None => println!("none"),
}
println!("{name:?}");
```

这段不能编译。`String` 不是 `Copy`，`Some(value)` 会把 `String` 从 `name` 里 move 出来。
`match` 结束后，`name` 已经被部分或整体消费，不能再打印。

`Option<usize>` 可能不同，因为 `usize: Copy`，匹配时复制内部值，不会让原 `Option<usize>`
失效。

只借用的写法：

```rust
match &name {
    Some(value) => println!("{value}"),
    None => println!("none"),
}
println!("{name:?}");
```

或者：

```rust
match name.as_ref() {
    Some(value) => println!("{value}"),
    None => println!("none"),
}
```

`as_ref()` 把 `Option<String>` 转成 `Option<&String>`，表达“借用里面的值”。`match &name`
则是直接匹配 `&Option<String>`。两者效果接近，但 `as_ref()` 在链式调用里更常见。

满分要点：

- 区分 `Copy` 和非 `Copy`。
- 说明 `Some(value)` 的 move。
- 写出 `match &name` 或 `name.as_ref()`。
- 能迁移到 `KvCache::len()` 为什么用 `as_ref()`。

### A3. `Result`、`?`、`bail!`

参考实现：

```rust
use anyhow::{bail, Result};

fn parse_ids(input: &str) -> Result<Vec<u32>> {
    if input.trim().is_empty() {
        bail!("input must contain at least one token id");
    }

    input
        .split(',')
        .map(str::trim)
        .map(|piece| {
            if piece.is_empty() {
                bail!("empty token id");
            }
            let token = piece.parse::<u32>()?;
            Ok(token)
        })
        .collect::<Result<Vec<_>>>()
}
```

也可以在 `parse` 后用 `map_err` 包更清楚的错误信息。

`?` 的控制流含义：

```text
Ok(value) => 取出 value 继续执行
Err(error) => 从当前函数提前返回 Err(error.into())
```

`bail!` 等价于构造一个错误并立刻 `return Err(...)`。

满分要点：

- 函数返回 `Result<Vec<u32>>`。
- 空输入和解析失败都走错误路径。
- 没有 `unwrap()`。
- 能解释 `collect::<Result<Vec<_>>>()` 的短路。

### A4. `match` 是表达式

参考答案：

`Mode::Slow` 分支最后的 `2` 是这个 block 的返回值。加分号后它变成语句，block 返回 `()`，
会和 `Mode::Fast => 1` 的类型不一致。

新增 `Mode::Auto` 后，如果 `match` 没覆盖它，编译器会报 non-exhaustive patterns。Rust 的
`match` 默认要求穷尽，这和 C++ `switch` 常见的 fallthrough/default 风格不同。

C++ `switch` 主要是语句；Rust `match` 是表达式，可以产生值，并且可以解构 enum、tuple、
struct。tour 里的例子：

```rust
match (top_k, top_p) {
    (None, None) => Sampling::All { temperature },
    (Some(k), None) => Sampling::TopK { k, temperature },
    (None, Some(p)) => Sampling::TopP { p, temperature },
    (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
}
```

它同时完成分支选择和 `Some(k)`、`Some(p)` 解构。

### A5. 闭包和 `collect::<Result<Vec<_>>>()?`

参考答案：

```rust
let blocks = (0..n)
    .map(|i| Block::load(vb.pp(format!("model.layers.{i}")), cfg))
    .collect::<Result<Vec<_>>>()?;
```

`map` 后 iterator 的 item 类型是 `Result<Block>`。`collect::<Result<Vec<_>>>()` 会把
`Iterator<Item = Result<Block>>` 汇总成 `Result<Vec<Block>>`：

- 如果所有元素都是 `Ok(block)`，返回 `Ok(Vec<Block>)`。
- 如果遇到第一个 `Err(e)`，立即短路返回 `Err(e)`。

最后的 `?` 在失败时把加载错误继续返回给调用者。

这比 `.unwrap()` 适合模型加载，因为模型加载失败通常要把缺失权重名、shape 错误、dtype
错误向上传递；`unwrap()` 只会 panic，错误边界粗糙，也不适合服务化。

## B. Candle Tensor 与后端

### B1. Tensor 五要素

```rust
let x = Tensor::arange(0f32, 24f32, &Device::Cpu)?.reshape((2, 3, 4))?;
```

答案：

- shape 是 `[2, 3, 4]`。
- contiguous stride 是 `[12, 4, 1]`。
- dtype 是 `F32`，由 `0f32` 和 `24f32` 决定。
- device 是 `Cpu`。
- Storage 保存底层数据和后端位置；Layout 保存 shape、stride、start offset 等逻辑视图信息。

手算 stride：

```text
最后一维 stride = 1
第二维 stride = 4
第一维 stride = 3 * 4 = 12
```

高分答案要强调：shape 只描述逻辑维度，不能推出所有布局信息。

### B2. view 与 copy

逐行判断：

```rust
let b = x.clone();
```

通常是 Tensor 句柄复制，共享底层存储，不是深拷贝整块数据。

```rust
let b = x.transpose(1, 2)?;
```

通常是 view，改变 dims/stride，不复制 Storage。

```rust
let b = x.narrow(1, 1, 2)?;
```

通常是 view，改变 dims 和 start offset。

```rust
let b = x.transpose(1, 2)?.contiguous()?;
```

`transpose` 先产生非连续 view，`contiguous()` 可能复制出新的连续 Storage。

```rust
let b = x.to_dtype(DType::F16)?;
```

通常会产生新 dtype 的 Tensor，可能分配新 Storage 或执行转换。

Attention 里常见 `.contiguous()?`，是为了让后续 matmul/kernel 面对更规整的布局，避免在更深
的后端里发生隐式复制或不支持非连续 layout。

### B3. `IndexOp` 和 shape 变化

设：

```rust
let x = Tensor::zeros((2, 3, 4), DType::F32, &Device::Cpu)?;
```

输出 shape：

```rust
x.i((.., 2, ..))?;     // [2, 4]
x.i((0, .., ..))?;     // [3, 4]
x.i((.., 1..3, ..))?;  // [2, 2, 4]
x.i((.., .., 0))?;     // [2, 3]
```

整数 index 会去掉对应维度；range index 会保留维度但改变长度。

```rust
hidden.i((.., seq_len - 1, ..))?.contiguous()?
```

用于从 `[B, S, H]` 中取最后一个 token 位置，得到 `[B, H]`。生成下一个 token 只需要最后
位置的 hidden state，再通过 lm_head 得到 `[B, V]` logits。

### B4. matmul 后端路径

`Tensor::matmul` 外层负责：

- 检查 rank 和矩阵乘维度。
- 处理 batch 维。
- 取得输入 Tensor 的 Storage/Layout 信息。
- 调用 Storage 层完成实际分发。
- 构造输出 Tensor 的 Storage/Layout。
- 在需要 autograd 时记录反向操作。

进入 `Storage::matmul` 是因为 Tensor 是高层句柄；真正数据在 Storage 里。Storage 是 CPU、
CUDA、Metal 等后端存储的 enum，运行时通过 `match` 分发到具体后端实现。

返回的新 Tensor 至少包含：

- 输出 shape/layout。
- 输出 dtype。
- 输出 device/storage。
- 可能的 autograd operation。

eager execution 指 `matmul` 调用时计算已经提交或执行，不是只登记一张未来才运行的静态计算图。
GPU 后端可能异步，所以精确计时时要考虑同步。

### B5. Device/DType 边界

Cargo feature 和运行时 device 是两层：

```text
--features cuda
```

决定编译时是否把 CUDA 支持编进二进制。

```rust
Device::new_cuda(0)?
```

决定运行时是否创建第 0 张 CUDA 设备。编译了 feature 不代表机器有可用 GPU；机器有 GPU 也不代表
当前二进制编了 CUDA。

`Tensor::new(&[1u32, 2], &device)?` 的 dtype 来自输入 Rust slice 的元素类型 `u32`。

GPU Tensor 调 `to_vec1::<f32>()?` 可能触发 device-to-host 拷贝和同步。

logits 采样前常转成 F32，是为了让 softmax、概率裁剪、随机采样等数值逻辑更稳定，也避免低精度
logits 在采样端产生额外误差。

## C. 模型加载与神经网络抽象

### C1. `Module` trait

`Linear`、`Embedding` 适合实现：

```rust
fn forward(&self, xs: &Tensor) -> Result<Tensor>
```

因为它们的输入输出签名稳定，只读权重，不需要请求级可变状态。

完整 LLaMA forward 往往还需要：

- token position。
- KV Cache。
- attention mask。
- 是否启用 tracing。
- metadata 或 cache policy。

所以模型或 attention 模块常有自定义 forward 签名。

`fn run<M: Module>(m: &M)` 是泛型静态分发，编译器可为具体类型单态化；`fn run(m: &dyn Module)`
是 trait object 动态分发，通过 vtable 调用，适合异构容器或运行时选择。

### C2. VarBuilder 参数路径

给定：

```rust
let vb = vb.pp("model.layers.3").pp("self_attn");
let q = linear_no_bias(hidden, hidden, vb.pp("q_proj"))?;
```

最终查找：

```text
model.layers.3.self_attn.q_proj.weight
```

`pp` 返回带前缀的新 builder，不应该修改原 builder。这样同一个上层 builder 可以派生多个子路径。

权重名存在但 shape 错误必须报错，不能静默继续。静默继续会让模型结构和 checkpoint 不一致，后续
matmul 或推理结果都不可信。

VarBuilder 适合隐藏不同权重来源，因为上层 layer 只关心“按路径取 Tensor”，不关心底层来自：

- SafeTensors mmap。
- `HashMap<String, Tensor>`。
- 可训练 VarMap。
- 已量化 buffer。

### C3. Serde config

参考结构：

```rust
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct TinyConfig {
    vocab_size: usize,
    hidden_size: usize,
    num_heads: usize,
    num_key_value_heads: Option<usize>,
}
```

`num_key_value_heads` 可选，因为非 GQA 模型可以默认等于 `num_heads`。

`validate()` 至少检查：

- `vocab_size > 0`。
- `num_heads > 0`。
- `hidden_size % num_heads == 0`。
- 如果有 `num_key_value_heads`，它应大于 0，且通常要求 `num_heads % num_key_value_heads == 0`。

### C4. SafeTensors 与 mmap

典型文件：

- `config.json`：模型结构超参数。
- `tokenizer.json`：文本和 token id 的转换规则。
- `*.safetensors`：权重 Tensor 的二进制数据和 metadata。

mmap 的目的：

- 避免一次性把整个权重文件复制进用户态 buffer。
- 让 OS 按页加载需要的数据。
- 多进程或多加载路径下可能复用 page cache。

mmap API 可能 unsafe，因为需要保证：

- 映射期间文件内容和长度有效。
- 指针不会越界。
- 字节解释成 dtype/shape 时满足对齐和长度要求。
- 底层映射生命周期长于所有 view。

### C5. trait object 与生命周期

```rust
pub type VarBuilder<'a> = VarBuilderArgs<'a, Box<dyn SimpleBackend + 'a>>;
```

`Box<dyn SimpleBackend>` 表示堆上保存一个实现了 `SimpleBackend` 的具体对象，但调用方只通过 trait
接口使用它。这是动态分发。

`'a` 大致约束 builder 和 backend 中借用数据的生命周期，避免 VarBuilder 比它引用的 mmap、
buffer 或其它后端数据活得更久。

## D. LLaMA 推理、Attention、KV Cache

### D1. prefill/decode 主循环

prompt 长度为 4，生成 3 个 token，cached decode：

| step | 输入 token 数 | `index_pos` | cache 长度 |
| --- | --- | --- | --- |
| 0 | 4 | 0 | 4 |
| 1 | 1 | 4 | 5 |
| 2 | 1 | 5 | 6 |

recompute-all：

| step | 输入 token 数 | `index_pos` | cache 长度 |
| --- | --- | --- | --- |
| 0 | 4 | 0 | 4 |
| 1 | 5 | 0 | 5 |
| 2 | 6 | 0 | 6 |

recompute-all 每步新建临时 cache，所以它不是复用历史 K/V，而是用完整 token 序列重算。

### D2. Attention shape 手算

给定：

```text
B=1
S=4
H=16
N=4
D=H/N=4
```

答案：

- Embedding 输出 `[1, 4, 16]`。
- Q projection 后仍是 `[1, 4, 16]`。
- Q reshape 为 `[1, 4, 4, 4]`，含义 `[B, S, N, D]`。
- Q transpose 后为 `[1, 4, 4, 4]`，含义 `[B, N, S, D]`。本例数值碰巧一样，但维度语义不同。
- K/V 新增也是 `[1, 4, 4, 4]`。
- prefill score 是 `[B, N, S, T] = [1, 4, 4, 4]`。
- 生成一个 token 后 decode，输入 `S=1`，cache 总长 `T=5`，score 是 `[1, 4, 1, 5]`。

高分答案会指出：shape 数字相同不代表维度语义相同。

### D3. causal mask

prefill 需要 causal mask，因为同一个 prompt 中第 i 个位置不能看第 i+1、i+2 等未来 token。

decode 时 `S=1`，当前输入只有最新 token，它的 K/V cache 代表历史加当前，通常没有“当前输入内部的
未来 token”，所以可以省略这层 mask。

如果一次 decode 多个新 token，`S > 1`，新 token 之间又出现未来关系，mask 问题会回来。

### D4. KV Cache 语义

KV Cache 保存历史 token 经过 k/v projection 和 RoPE 后的 key/value。它避免每生成一个新 token 都
对完整上下文重新计算 K/V。

每个请求必须独立 cache，因为不同用户、不同 prompt、不同生成历史对应不同 K/V。共享 cache 会造成
请求串扰。

教学示例用 `Tensor::cat` 是为了直观展示 cache 增长；生产系统通常会预分配 KV buffer 或使用分页
KV cache，因为每步 cat 会反复分配和复制，长上下文下成本很高。

GQA 中 `num_key_value_heads < num_attention_heads`，K/V head 更少，再通过 repeat/broadcast 服务多
个 query heads，从而降低 KV cache 内存和带宽。

### D5. sampling 与 EOS

logits 到 next token：

```text
logits [V]
  -> 可选 repeat penalty
  -> temperature 缩放
  -> top-k/top-p 裁剪
  -> softmax/probability
  -> RNG 或 argmax 选择 token
```

temperature 控制分布尖锐程度；top-k 只保留概率最高的 k 个候选；top-p 保留累计概率达到 p 的候选
集合。

repeat penalty 需要读取一段历史 token，常见是最近窗口，而不是只看当前 token。

EOS 可能是单 token、多个 token 序列、或模型配置中的多个终止 id。enum 比裸 `Option<u32>` 更能
表达状态：

```rust
enum Eos {
    None,
    Single(u32),
    AnyOf(Vec<u32>),
    Sequence(Vec<u32>),
}
```

## E. 工程化、并发、unsafe

### E1. Cargo feature 与 `cfg`

```bash
cargo run -p candle-examples --example llama --features metal -- --sample-len 8
```

这里 `--features metal` 是 Cargo 参数，表示编译时启用 metal feature；`--` 后面才是 example CLI
参数。

```bash
cargo run -p candle-examples --example llama -- --features metal --sample-len 8
```

这里 `--features metal` 被传给运行中的程序。如果程序没有定义这个 CLI 参数，就会报未知参数。

`#[cfg(feature = "metal")]` 在编译期生效。没开 feature 的代码分支不会被编译进当前目标。

### E2. tracing 与性能指标

只看总 token/s 不够，因为 LLM 推理分 prefill 和 decode：

- TTFT：time to first token，用户感知首 token 延迟。
- prefill：处理 prompt 的吞吐，通常由 prompt 长度和大矩阵计算主导。
- decode token/s：逐 token 生成吞吐，常受 KV cache 读写、batching、memory bandwidth 影响。

GPU 异步会让 CPU 侧计时偏短。精确计时需要同步，或者使用后端 profiler/event。

### E3. 服务化所有权设计

最小设计：

```text
共享：
Model weights
Device handle
Tokenizer
只读 config

每请求独立：
tokens
KvCache
sampler/RNG
stopping state
cancel flag
stream output buffer
```

权重可以用 `Arc<Model>` 共享。KV Cache、tokens、RNG 必须按请求独立。否则一个请求生成的历史会
污染另一个请求。

不要用一个大 `Mutex<ModelWithCache>` 把整个 forward 串行化。更好的边界是共享只读模型权重，把
可变请求状态放在 request/session 对象里。真正 GPU batching 需要调度器，但语义上仍要隔离请求状态。

### E4. `Arc<RwLock<T>>`

`Arc<T>` 让多个所有者共享同一个堆对象，靠原子引用计数管理生命周期。

`RwLock<T>` 让多读单写在运行时受锁保护，适合读多写少的共享状态。

`RwLock` 不是绕过借用检查。它把“同一时间多读或单写”的检查从编译期转到运行时；拿不到锁会阻塞或
返回错误，锁保护范围仍然要设计清楚。

### E5. unsafe 审计

mmap SafeTensors 示例：

```text
unsafe operation:
  把文件映射为内存，并把其中字节解释成 Tensor 数据。

safety invariant:
  文件在映射生命周期内有效；offset/length 不越界；dtype 和 shape 对应的字节数正确；
  数据对齐满足读取要求；所有 Tensor view 不比映射活得更久。

who guarantees it:
  SafeTensors parser 验证 metadata，mmap wrapper 管理映射生命周期，VarBuilder/Storage
  通过生命周期参数约束借用关系。

what happens if violated:
  可能读越界、读到被修改或释放的文件内容、产生未定义行为或错误推理结果。

safe wrapper boundary:
  对外只暴露安全的 Tensor/VarBuilder API；unsafe 被封装在 mmap 创建和字节解释的最小范围。
```

未初始化 Storage 示例：

```text
unsafe operation:
  分配未初始化 buffer，然后由 kernel 写入。

safety invariant:
  所有元素在被读取前都已写入；kernel 不越界；dtype layout 和 buffer 大小一致。

who guarantees it:
  shape/dtype 检查、后端 kernel contract、safe Tensor 构造函数。

what happens if violated:
  读取未初始化内存、越界写、数据竞争或错误结果。

safe wrapper boundary:
  public API 返回 Tensor 前必须保证 Storage 已完整初始化。
```

## F. 实操题

### F1. layout trace

当前 tour 已经在 `--trace-shapes` 下打印 shape/dtype。这个实操题要求继续扩展 trace helper，
增加 contiguous 状态，而不是每个调用点复制打印逻辑。

示例方向：

```rust
fn trace_tensor(enabled: bool, label: &str, tensor: &Tensor) {
    if enabled {
        println!(
            "    {label:<22} shape={:?} dtype={:?} contiguous={}",
            tensor.dims(),
            tensor.dtype(),
            tensor.is_contiguous()
        );
    }
}
```

如果代码已经有 `tensor_trace_line`，更好的改法是只修改这个格式化函数，并补一个测试断言输出
包含 `contiguous=`。

满分要求：

- `--trace-shapes` 原有 shape/dtype 信息仍在。
- contiguous 能显示 token ids、Q、last hidden、logits。
- 不在主流程里到处写重复 `println!`。
- 能解释 `transpose` 后为什么常接 `contiguous()`，以及 `contiguous()` 可能复制。

### F2. 多 EOS

参考设计：

当前 tour 已经有单个 `--eos-token <u32>`。本题要求扩展为多个 EOS。

一种可接受设计：

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
enum EosTokens {
    None,
    One(u32),
    Any(Vec<u32>),
}
```

CLI 可以保留：

```rust
eos_token: Option<u32>
eos_tokens: Option<String>
```

然后在进入生成前归一化成 `EosTokens`：

```rust
let eos = build_eos_tokens(args.eos_token, args.eos_tokens.as_deref(), vocab_size)?;
```

生成循环不要关心 CLI 细节，只处理已经归一化后的停止策略：

```rust
tokens.push(next_token);
if eos.matches(next_token) {
    stopped_on_eos = Some(next_token);
    break;
}
```

注意停止发生在采样后，因为 EOS 本身通常也要进入输出 token 序列。

满分要求：

- 未设置、单 token、多 token 三种状态清楚。
- 解析和 vocab 校验集中在生成前。
- 多 EOS 命中任一 token 都能停止。
- 越界 token 有清楚错误。
- 单 EOS 的旧用法仍可工作，或者明确给出兼容性迁移说明。

### F3. layout walkthrough 增加 view

对：

```rust
base.narrow(0, 1, 1)
```

原 base：

```text
dims=[2,3,4]
stride=[12,4,1]
offset=0
```

沿 dim 0 从 index 1 开始取 length 1：

```text
dims=[1,3,4]
stride=[12,4,1]
offset=12
contiguous=true
```

Candle 的 `Layout::is_contiguous()` 看 shape 和 stride，不要求 `start_offset == 0`。这个 view
虽然从 Storage 的 offset 12 开始，但它覆盖的是一块连续区域，并且 `[1,3,4]` 的标准 contiguous
stride 正是 `[12,4,1]`。

## G. 加分题

### G1. 真实 LLaMA 代码片段

高分解释模板：

```text
函数：Attention::forward
输入：hidden [B,S,H]，position/index_pos，cache
输出：[B,S,H]
读权重：q_proj/k_proj/v_proj/o_proj，可能还有 norm 或 rope 参数
修改状态：当前请求的 KV cache
shape：
  q [B,Nq,S,D]
  k/v [B,Nkv,S,D]
  cache [B,Nkv,T,D]
  repeat_kv 后 [B,Nq,T,D]
  scores [B,Nq,S,T]
可能复制：
  transpose 后 contiguous
  cache cat 或 cache 写入
  repeat_kv
  dtype/device 转换
```

必须把 `num_attention_heads` 和 `num_key_value_heads` 区分开。没有提 GQA 的答案很难拿满分。

### G2. 最小 LLM runner

高分设计应包括：

- CLI：model path、tokenizer path、prompt、sample len、temperature、top-k/top-p、seed、device。
- 加载：config、tokenizer、weights、VarBuilder、model。
- 请求状态：tokens、KvCache、LogitsProcessor/RNG、eos tracker。
- 循环：prefill 第一轮，decode 后续轮，采样，停止条件，流式输出。
- 错误处理：启动期错误和请求期错误分开。
- 测试：参数校验、prompt tokenization、cache/recompute 一致性、EOS、layout/shape 边界。

不要把 cache、RNG、tokens 放进全局模型对象。这个错误会直接暴露并发设计不合格。

## 批改时的优先级

同样是 80 分，差异可能很大。优先看这些能力：

1. 能不能把 Rust 所有权错误和真实内存风险联系起来。
2. 能不能稳定手算 Tensor shape 和 layout。
3. 能不能区分模型态、请求态、临时 Tensor。
4. 能不能把 `Result` 当作系统边界，而不是语法负担。
5. 能不能说明 cache 优化为什么不能改变 logits 语义。
6. 能不能在 unsafe 问题里说清楚不变量和封装边界。

如果某一类题低于一半分，建议回到对应材料：

| 薄弱项 | 回看 |
| --- | --- |
| 所有权、借用、Option、Result | [`typical-patterns.md`](./typical-patterns.md) |
| shape、layout、cache | [`rust-candle-labs.md`](./rust-candle-labs.md) |
| 源码阅读路线 | [`source-reading-playbook.md`](./source-reading-playbook.md) |
| 系统不变量 | [`deep-inference-systems.md`](./deep-inference-systems.md) |
| 阶段性自检 | [`checkpoints.md`](./checkpoints.md) |
