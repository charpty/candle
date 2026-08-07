# Rust + Candle 典型用法精讲

这份文档专门讲主讲义和 `rust-candle-tour` 示例中反复出现的典型用法。读源码时不要把它们
当成零散语法点，它们通常是在表达工程约束：

- 这个值可能不存在。
- 这个操作可能失败。
- 这段数据只是借用，不拥有。
- 这个 Tensor 是 view 还是物理 copy。
- 这个状态会在下一次 token 生成时继续使用。

建议读法：

1. 先运行 `rust-candle-tour`。
2. 遇到看不懂的写法，到本文对应小节查。
3. 回到源码里确认同一种写法出现在哪里。
4. 小改一行代码，让编译器或测试告诉你理解是否正确。

如果你已经知道要追哪条调用链，但不知道该看哪些文件，先看
[Candle 源码阅读路线图](./source-reading-playbook.md)。

配套命令：

```bash
cargo run -p candle-examples --example rust-candle-tour -- --cpu -n 2
cargo run -p candle-examples --example rust-candle-tour -- --cpu --trace-shapes -n 2
cargo run -p candle-examples --example rust-candle-tour -- --cpu --show-layout-path -n 0
cargo run -p candle-examples --example rust-candle-tour -- --cpu --compare-cache -n 4
cargo test -p candle-examples --example rust-candle-tour
```

---

## 1. `Result<T>`、`?` 和 `bail!`

典型位置：

- [main.rs](../candle-examples/examples/rust-candle-tour/main.rs)
- [model.rs](../candle-examples/examples/rust-candle-tour/model.rs)
- [sampling.rs](../candle-examples/examples/rust-candle-tour/sampling.rs)

核心理解：

`Result<T, E>` 表示“这一步可能失败”。`?` 表示“成功就取出里面的值，失败就立刻从当前函数
返回这个错误”。`bail!` 是直接构造错误并返回。

在应用入口里常见：

```rust
fn main() -> Result<()> {
    let device = candle_examples::device(args.cpu)?;
    let model = TinyCausalLm::from_demo_weights(config, &device)?;
    Ok(())
}
```

这不是异常机制。调用点必须从函数签名上承认自己也可能失败：

```rust
fn parse_prompt(prompt: &str, vocab_size: usize) -> Result<Vec<u32>>
```

如果 `parse_prompt` 里面用了 `?`，它自己的返回类型就必须能承接错误。

C++ 类比：

```cpp
expected<Device, Error> device = make_device(cpu);
if (!device) return unexpected(device.error());
```

Rust 的 `?` 把这种样板写法压缩了，但控制流仍然是显式的。

常见坑：

- `?` 只能用在返回 `Result`、`Option` 等兼容类型的函数里。
- 框架内部更适合使用结构化错误；示例和 CLI 使用 `anyhow::Result` 更方便汇总不同库错误。
- `bail!` 后面的代码不会继续执行，它相当于 `return Err(...)`。

练习：

把 `parse_prompt("1,,2", 32)` 跑进测试，确认错误来自空 token，而不是 `parse::<u32>` 的
默认错误。这样你会看到“先做业务校验，再做类型解析”的价值。

---

## 2. `Option<T>` 表达“可能没有”

典型位置：

- `top_k: Option<usize>`
- `top_p: Option<f64>`
- `KvCache { kv: Option<(Tensor, Tensor)> }`

`Option<T>` 不是“可空指针”的小修补，而是把状态放进类型系统。

```rust
enum Option<T> {
    None,
    Some(T),
}
```

在采样参数里：

```rust
match (top_k, top_p) {
    (None, None) => Sampling::All { temperature },
    (Some(k), None) => Sampling::TopK { k, temperature },
    (None, Some(p)) => Sampling::TopP { p, temperature },
    (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
}
```

这里一眼能看出四种配置组合都被处理了。少写一个分支，编译器会报错。

在 KV Cache 里：

```rust
match &cache.kv {
    None => (k_new, v_new),
    Some((k_cached, v_cached)) => (...),
}
```

`None` 代表还没有 prefill；`Some` 代表已有历史 K/V。这个状态不需要额外的布尔变量。

C++ 类比：

```cpp
std::optional<std::pair<Tensor, Tensor>> kv;
```

重要差异是 Rust 代码通常用模式匹配打开它，而不是到处写 `has_value()` 和 `value()`。

常见坑：

- `match option` 会按值消费 `option`。
- 如果里面是 `String`、`Tensor` 这类非 `Copy` 类型，消费后原变量不能再用。
- 只想读取时用 `match &option` 或 `option.as_ref()`。

练习：

把 `KvCache::len` 里的 `as_ref()` 去掉，观察编译器如何提示你不能从共享引用里移动 Tensor。

---

## 3. `match` 不只是 `switch`

典型位置：

- 采样策略选择。
- cache 是否已有历史 K/V。
- 生成模式 `Cached` / `RecomputeAll`。

Rust 的 `match` 同时做三件事：

1. 分支选择。
2. 解构数据。
3. 穷尽检查。

例如：

```rust
let active_cache = match mode {
    GenerationMode::Cached => &mut cache,
    GenerationMode::RecomputeAll => {
        recompute_cache = KvCache::default();
        &mut recompute_cache
    }
};
```

这段代码不是单纯判断枚举值。它还保证了两种模式都返回同一种类型：`&mut KvCache`。

`match` 是表达式，所以可以直接给变量赋值。每个正常分支必须产生兼容类型。

常见坑：

- 分支最后一个表达式不要加分号；加了分号会变成 `()`。
- 所有分支的返回类型必须一致。
- 对非 `Copy` 值使用 `match value` 会移动所有权；只读时优先 `match &value`。

练习：

在 `GenerationMode` 里新增一个枚举值但不修改 `match`，看编译器如何指出漏处理的分支。

---

## 4. `.collect::<Result<Vec<_>>>()?`

典型位置：

- prompt 字符串解析。
- 真实模型里构造多层 block。

示例：

```rust
let tokens = prompt
    .split(',')
    .map(str::trim)
    .map(|piece| piece.parse::<u32>().map_err(...))
    .collect::<Result<Vec<_>>>()?;
```

每个 `piece.parse::<u32>()` 都返回一个 `Result<u32, _>`。最终 iterator 的元素类型是
`Result<u32, _>`，而不是 `u32`。

`collect::<Result<Vec<_>>>()` 的意思是：

- 每一项都是 `Ok(value)`：得到 `Ok(Vec<value>)`。
- 遇到第一个 `Err(error)`：立刻得到 `Err(error)`。

这在 Rust 工程里非常常见。它避免了手写：

```rust
let mut values = Vec::new();
for item in items {
    values.push(item?);
}
```

常见坑：

- turbofish `::<Result<Vec<_>>>()` 是在帮助编译器确定收集目标。
- `?` 作用在整个 collect 结果上，不是作用在单个 iterator item 上。
- 如果你想收集所有错误，而不是遇到第一个错误就返回，就不能用这种写法。

练习：

把 `parse_prompt` 改成显式 `for` 循环，再改回来。两种写法的行为应该一致。

---

## 5. `Vec<T>`、`&[T]` 和借用结束点

典型位置：

```rust
let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
let input = Tensor::new(ctxt, device)?.unsqueeze(0)?;
println!("input={ctxt:?}");
tokens.push(next_token);
```

`tokens` 拥有 token id。`ctxt` 是借用出来的一段 slice，不拥有数据。

C++ 类比：

| Rust | C++ 近似 |
| --- | --- |
| `Vec<u32>` | `std::vector<uint32_t>` |
| `&[u32]` | `std::span<const uint32_t>` |

关键差异：

Rust 编译器知道 `ctxt` 借用了 `tokens`。只要 `ctxt` 后面还要用，编译器就不会允许
`tokens.push(...)`，因为 `push` 可能重新分配底层内存。

这解释了示例里为什么先打印 `ctxt`，再 `tokens.push(next_token)`。

常见坑：

- 不要把 slice 当成“便宜 vector”。它没有所有权。
- `push`、`resize`、`extend` 都可能让旧 slice 失效。
- Rust 的 non-lexical lifetime 会尽量把借用结束点缩短到最后一次使用处，不一定等到作用域结束。

练习：

把 `tokens.push(next_token)` 移到 `println!("input={ctxt:?}")` 前面，观察 E0502 这类错误。

---

## 6. `&self`、`&mut self` 和显式状态

典型位置：

```rust
model.forward(&input, index_pos, &mut cache, trace_shapes)?;
sampler.sample(&logits)?;
```

模型权重不变，所以模型 forward 接收 `&self`。KV Cache 会追加 K/V，所以传 `&mut cache`。
采样器内部有 RNG 状态，所以 `LogitsProcessor::sample` 需要 `&mut self`。

函数签名就是状态说明：

| 签名 | 含义 |
| --- | --- |
| `&self` | 只需要共享读取这个对象 |
| `&mut self` | 会独占修改这个对象 |
| `T` | 消费或接管这个值 |

C++ 中这些语义常藏在 `const`、指针、引用和文档里。Rust 把它们放在类型系统里。

常见坑：

- 同一时间不能既有 `&mut cache`，又有另一个还在使用的 `&cache`。
- `&mut` 不是“可空可别名指针”，它表达独占访问。
- 需要共享模型权重时，通常共享 `Arc<Model>`，但每个请求仍应有独立 cache。

练习：

尝试把两个请求共用同一个 `KvCache`。你会发现即使能编译，语义也是错的：第二个请求会看到第
一个请求的历史 token。

---

## 7. `Tensor::new(..., device)?.unsqueeze(0)?`

典型位置：

```rust
let input = Tensor::new(ctxt, device)?.unsqueeze(0)?;
```

这行完成两件事：

1. 把 host 上的 token slice 变成目标 `Device` 上的 Tensor。
2. 加 batch 维，从 `[S]` 变成 `[1, S]`。

Embedding 通常期待 token id shape 是 `[batch, seq]`。单条 prompt 也要保留 batch 维。

dtype 怎么来：

- `ctxt` 是 `&[u32]`，所以 token Tensor 是整数 dtype。
- Embedding 权重是 F32，所以输出 hidden 是 F32。
- logits 最后显式 `to_dtype(DType::F32)`，方便采样器处理。

常见坑：

- 少了 `unsqueeze(0)`，后面 `dims2()` 或 Embedding shape 约定会不匹配。
- `Tensor::new` 可能涉及 host-to-device 拷贝。
- GPU Tensor 调 `to_vec*` 会把数据读回 CPU。

练习：

运行：

```bash
cargo run -p candle-examples --example rust-candle-tour -- --cpu --trace-shapes -n 1
```

确认 token ids 是 `[1, S]`，Embedding 输出是 `[1, S, H]`。

---

## 8. shape、stride、view 和 `contiguous()`

典型位置：

```rust
let q = self
    .q_proj
    .forward(x)?
    .reshape((batch, seq_len, self.num_heads, self.head_dim))?
    .transpose(1, 2)?
    .contiguous()?;
```

`reshape` 改 shape。`transpose` 改维度顺序。它们通常可以只创建新的 layout view，共享同一块
Storage。

但是很多 backend kernel 更喜欢或要求 row-major contiguous 输入，所以 attention 里紧跟
`.contiguous()?`。

用命令观察：

```bash
cargo run -p candle-examples --example rust-candle-tour -- --cpu --show-layout-path -n 0
```

你会看到类似：

```text
base           dims=[2, 3, 4] stride=[12, 4, 1] contiguous=true
transpose      dims=[2, 4, 3] stride=[12, 1, 4] contiguous=false
contiguous()   dims=[2, 4, 3] stride=[12, 3, 1] contiguous=true
```

判断是否复制：

| 操作 | 通常行为 |
| --- | --- |
| `clone()` | 复制 Tensor 句柄，共享 Storage |
| `transpose()` | 创建 view，通常不复制 |
| `narrow()` | 创建 view，通常不复制 |
| `contiguous()` 且已连续 | 返回共享句柄 |
| `contiguous()` 且不连续 | 分配新 Storage 并复制 |
| `to_dtype()` | 创建新 dtype 的 Storage |

常见坑：

- shape 相同不代表内存布局相同。
- view 便宜，但后续 kernel 可能要先打包成 contiguous。
- `Tensor::clone()` 和 `Vec::clone()` 成本完全不同。

练习：

在 `layout_walkthrough.rs` 里增加一个 `base.narrow(0, 1, 1)`，预测 offset 和 stride，再跑测试。

---

## 9. `Tensor::clone()` 不是复制整块数据

典型位置：

```rust
cache.kv = Some((k.clone(), v.clone()));
```

Candle 的 `Tensor` 是一个共享句柄。`clone()` 通常只是增加引用计数，让两个 Tensor 句柄指向
同一份底层 Storage 和 layout。

这和 `Vec<T>::clone()` 很不一样：

| 类型 | `clone()` 常见成本 |
| --- | --- |
| `Vec<f32>` | 复制整段元素 |
| `Tensor` | 复制句柄，底层 Storage 共享 |

为什么 cache 里要 clone：

- 局部变量 `k`、`v` 后面还要参与 attention 计算。
- cache 也要保存一份句柄供下一轮 decode 使用。
- Tensor 句柄共享底层数据，不需要复制整块 K/V。

常见坑：

- 不要看到 `.clone()` 就自动认为性能很差。
- 也不要认为所有 clone 都便宜；要看类型。
- 如果你真的要复制 Tensor 数据，看 `copy()`、`contiguous()` 或 dtype/device 转换。

练习：

在文章里遇到 `.clone()` 时，先写下被 clone 的类型。只有知道类型，才能判断成本。

---

## 10. `Module` trait 和 `.forward`

典型位置：

```rust
self.token_embedding.forward(token_ids)?;
self.attention.forward(&hidden, index_pos, cache, trace_shapes)?;
self.lm_head.forward(&last_hidden)?;
```

Candle 里 layer 通常实现 `Module`：

```rust
pub trait Module {
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}
```

这让 `Embedding`、`Linear`、闭包和自定义模块都可以用统一风格调用。

和 C++ 的区别：

- 它可以像 interface 一样做动态分发。
- 也可以通过泛型做静态分发。
- 不需要继承模型基类才能组合层。

示例中的 `TinyCausalLm` 没有强行实现 `Module`，因为它的 forward 还需要 `index_pos`、
`cache`、`trace_shapes` 这些额外参数。真实 LLaMA 也会有类似原因：推理 forward 不只是
`Tensor -> Tensor`。

常见坑：

- `Module::forward(&self, &Tensor)` 适合无额外状态的层。
- 需要 cache、mask、位置、训练/推理模式时，模型会定义自己的 forward 签名。
- trait 不是“必须用继承”的信号；Rust 更偏组合。

练习：

写一个 `Scale { factor: f64 }`，实现 `Module`，让它对输入 Tensor 做乘法。主讲义第 4 课已有
这个练习。

---

## 11. `VarBuilder` 和参数路径

典型位置：

```rust
linear_no_bias(hidden, hidden, vb.pp("q_proj"))?
```

`VarBuilder` 可以理解为“带路径前缀、dtype、device 和权重来源的参数查询器”。它不是 layer，
也不是简单文件句柄。

路径如何组成：

```text
vb.pp("model.layers.0.self_attn").pp("q_proj").get("weight")
```

最终就是：

```text
model.layers.0.self_attn.q_proj.weight
```

示例里用 `from_demo_weights` 构造内存 HashMap：

```rust
let vb = VarBuilder::from_tensors(weights, DType::F32, device);
```

真实模型通常用 SafeTensors：

```rust
VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)?
```

常见坑：

- 参数名错了，加载会失败。
- shape 不匹配，加载会失败。
- `pp` 不修改原 builder，而是返回带新前缀的新 builder。
- builder 的生命周期可能受底层权重来源约束，尤其是从内存 slice 读取 SafeTensors 时。

练习：

把示例里某个权重名改错，例如 `q_proj.weight` 改成 `query_proj.weight`，观察加载错误。

---

## 12. KV Cache：算法状态与 Rust 状态一致

典型位置：

```rust
pub struct KvCache {
    kv: Option<(Tensor, Tensor)>,
}
```

第一次 prefill：

```text
input S = prompt_len
cache T = prompt_len
```

后续 decode：

```text
input S = 1
cache T = prompt_len + generated_len
```

这正是 `--trace-shapes` 展示的内容。

`--compare-cache` 展示两个模式：

- `cached`：第一次喂完整 prompt，后续只喂最新 token。
- `recompute-all`：每一步都喂完整历史 tokens。

两者在 greedy 下应该生成同样 token。区别在计算量：

```text
cached:        prompt_len + generated_len * 1 个输入位置
recompute-all: prompt_len + (prompt_len+1) + (prompt_len+2) + ...
```

真实服务里 cache 还有更多系统问题：

- 每个请求独立 cache。
- cache 生命周期跟请求绑定。
- batch serving 要管理多个请求的 cache。
- 高性能实现会预分配和分页，而不是每步 `Tensor::cat`。

常见坑：

- 不要把模型权重和请求 cache 放在同一个共享对象里随便跨请求复用。
- `Tensor::cat` 适合教学，不适合高吞吐服务的最终 cache 结构。
- 随机采样会引入 RNG 状态；比较 cache 语义时先用 greedy。

练习：

运行：

```bash
cargo run -p candle-examples --example rust-candle-tour -- --cpu --compare-cache -n 4
```

观察 cached 的输入长度一直是 1，而 recompute-all 的输入逐步增长。

---

## 13. 测试里的 `Result<()>`

典型位置：

```rust
#[test]
fn forward_extends_cache_from_prefill_to_decode() -> Result<()> {
    let prompt = Tensor::new(&[1u32, 5, 9], &device)?.unsqueeze(0)?;
    Ok(())
}
```

Rust 测试函数可以返回 `Result<()>`。这样测试里也能直接用 `?`。

适合场景：

- 构造 Tensor 可能失败。
- 模型 forward 可能失败。
- 你真正想断言的是结果，而不是每一步都手写 `unwrap()`。

常见坑：

- `assert_eq!` 失败会 panic；`?` 返回的是错误。这两种失败都会让测试失败。
- 不要在测试里滥用 `unwrap()`，除非你确实想让 panic 信息直接暴露。
- 单元测试尽量保持输入很小，避免依赖网络和模型下载。

练习：

给 `build_sampling` 新增一个非法 temperature 测试，先用 `is_err()`，再改成检查错误消息。

---

## 14. 读 Candle 源码时的判断顺序

看到一段 Rust + Candle 代码，先按这个顺序问：

1. 这里的数据所有者是谁？
2. 函数拿的是 `T`、`&T` 还是 `&mut T`？
3. 这一步可能失败吗？失败通过 `Result` 还是 panic？
4. Tensor 的 shape 变了吗？
5. Tensor 的 stride/layout 变了吗？
6. 这一步是否可能分配新 Storage？
7. Device 是否可能发生 host/device 拷贝？
8. 这个状态会不会跨 token、跨层或跨请求保存？

不要一开始就问“这是什么语法”。先问它在表达什么工程事实。Rust 语法只是把这些事实写进了
类型和函数签名。

---

## 15. `#[derive(Parser)]` 和 clap 参数结构

典型位置：

- `candle-examples/examples/llama/main.rs`
- `candle-examples/examples/rust-candle-tour/main.rs`

示例：

```rust
#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    cpu: bool,

    #[arg(long, default_value = "1,5,9,2")]
    prompt: String,

    #[arg(short = 'n', long, default_value_t = 8)]
    sample_len: usize,
}
```

`derive(Parser)` 是过程宏。它在编译期根据结构体字段和 `#[arg(...)]` 属性生成命令行解析代码。
你写的是类型定义，宏补的是 parser、help、默认值和错误消息。

C++ 类比：

```cpp
struct Args {
    bool cpu = false;
    std::string prompt = "1,5,9,2";
    std::size_t sample_len = 8;
};
```

区别是 Rust 这里的结构体还能直接驱动 CLI parser。字段类型就是解析目标类型：

| 字段类型 | CLI 语义 |
| --- | --- |
| `bool` | `--flag` 出现就是 true |
| `String` | 接收字符串参数 |
| `usize`、`f64` | 解析数字，失败自动报错 |
| `Option<T>` | 参数可选，未出现就是 `None` |

常见坑：

- `default_value = "..."` 是字符串形式，适合 string-like 默认值。
- `default_value_t = 8` 是按 Rust 表达式给 typed 默认值。
- Cargo 参数和 example 参数中间要有 `--`：前面给 Cargo，后面给你的程序。

练习：

给 tour 示例加一个 `--vocab-size` 参数。你需要想清楚它应该影响 `TinyConfig`，也要继续校验
prompt token 是否越界。

---

## 16. Cargo workspace、package、crate 和 feature

典型位置：

- 根目录 `Cargo.toml`
- `candle-examples/Cargo.toml`
- `#[cfg(feature = "cuda")]`

先分清三个词：

| 概念 | 含义 | C++ 近似 |
| --- | --- | --- |
| workspace | 多个 package 一起开发 | 顶层 CMake project |
| package | 一个 `Cargo.toml` 描述的包 | 一个库或可执行 target 集合 |
| crate | Rust 编译单元 | library/executable target |

feature 是编译期能力开关。例如：

```bash
cargo run -p candle-examples --example llama --release --features metal
```

它不是运行时配置。启用 `metal` feature 后，Metal 相关代码和依赖会被编译进二进制；没启用时，
相关代码可能根本不存在。

运行时 Device 是另一回事：

```rust
Device::Cpu
Device::new_cuda(0)?
Device::new_metal(0)?
```

所以你要同时问两个问题：

1. 这个二进制有没有编进 CUDA/Metal 能力？
2. 这次运行实际选择哪个 Device？

常见坑：

- `--features cuda` 是 Cargo 参数，必须放在 `--` 前面。
- `--cpu` 是 example 参数，必须放在 `--` 后面。
- 改 feature 通常会触发重新编译。

练习：

比较下面两条命令的参数归属：

```bash
cargo run -p candle-examples --example llama --features metal -- --sample-len 8
cargo run -p candle-examples --example llama -- --features metal --sample-len 8
```

第二条里的 `--features` 已经不是 Cargo feature 了，而是传给 example 的普通参数。

---

## 17. `#[cfg(...)]`：代码是否存在

典型位置：

```rust
#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(test)]
mod tests {
    ...
}
```

`cfg` 是条件编译。条件不满足时，被标注的语法项不会进入编译器后续阶段。

常见条件：

```rust
#[cfg(feature = "cuda")]
#[cfg(target_os = "macos")]
#[cfg(target_arch = "aarch64")]
#[cfg(test)]
```

C++ 类比是 `#ifdef`，但 Rust 的 `cfg` 作用于完整语法项，和 Cargo feature、target triple、
测试构建模式集成得更紧。

常见坑：

- `#[cfg(test)]` 里的代码只在测试构建中存在。
- feature-gated API 如果没启用 feature，普通代码不能引用它。
- 条件编译不是运行时 if；不能用变量控制。

练习：

在 tour 里找 `#[cfg(test)]`。解释为什么这些测试模块不会进入普通 `cargo run` 的二进制。

---

## 18. Serde config：外部 JSON 到强类型 Rust

典型位置：

- `candle-transformers/src/models/llama.rs`
- `candle-examples/examples/llama/main.rs`

真实 LLaMA 示例会读取 `config.json`：

```rust
let config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;
```

模型配置结构体通常派生：

```rust
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
}
```

关键点：

- JSON 字段必须能映射到 Rust 字段。
- 可选字段用 `Option<T>`。
- 默认值、重命名、untagged enum 都可以通过 `#[serde(...)]` 表达。

C++ 常见写法是手写 JSON parser 或宏反射。Rust/Serde 的优势是：配置一旦进入 Rust 结构体，
后续代码不再到处处理 loosely typed JSON。

常见坑：

- 外部配置结构体不等于模型内部最终配置；通常还需要一次校验或转换。
- `Option<T>` 不代表“随便空着也没关系”，只代表文件里可能没有。
- 反序列化成功不代表数学维度合法；例如 `hidden_size % num_heads` 仍要显式校验。

练习：

给 `TinyConfig` 想一个 JSON 版本：哪些字段必须有，哪些字段可以给默认值，哪些字段需要
`validate()`？

---

## 19. `Device`、`DType` 和 host/device 边界

典型位置：

```rust
let device = candle_examples::device(args.cpu)?;
let dtype = DType::F16;
let x = Tensor::new(values, &device)?;
let y = x.to_dtype(DType::F32)?;
```

`Device` 决定 Tensor 数据在哪里以及用哪个 backend 执行算子。`DType` 决定元素类型。

常见组合：

| 目标 | 常用 dtype |
| --- | --- |
| CPU 调试 | F32 |
| GPU 推理 | F16/BF16 |
| token ids | U32/U64/I64 等整数 |
| logits sampling | F32 |

`Tensor::new` 从 host 数据创建 Tensor。如果 device 是 GPU，这可能触发 host-to-device 拷贝。
`to_vec1`、`to_vec2`、`to_scalar` 则把 Tensor 读回 host。

常见坑：

- dtype 不匹配可能导致隐式不支持或显式转换成本。
- device 不一致的 Tensor 不能直接做大多数二元算子。
- GPU 计时要注意异步执行，必要时 synchronize。
- F16/BF16 的数值误差不能按逐 bit 比较。

练习：

运行 `--trace-shapes`，观察 token ids、Embedding 输出和 logits 的 dtype。当前 tour 已经把
shape/dtype 放在同一行；下一步可以继续扩展 contiguous 或 device trace。

---

## 20. Tokenizer、EOS 和 repeat penalty

典型位置：

- `candle-examples/examples/llama/main.rs`
- `candle-transformers/src/generation/mod.rs`

真实 LLaMA 示例不是直接输入 token id，而是：

```text
prompt string -> tokenizer.encode -> Vec<u32> token ids
```

生成循环里通常还有：

1. 从 logits 采样 next token。
2. 根据 tokenizer 增量解码成文本。
3. 检查 EOS 是否结束。
4. 对最近 token 应用 repeat penalty。

EOS 可能是单个 token，也可能是多个 token。Rust 里适合用 enum 表达：

```rust
match eos_token_id {
    Some(LlamaEosToks::Single(eos_tok_id)) if next_token == eos_tok_id => ...
    Some(LlamaEosToks::Multiple(ref eos_ids)) if eos_ids.contains(&next_token) => ...
    _ => ...
}
```

这里的 `ref eos_ids` 很关键：它借用 enum variant 里的集合，而不是把集合移动出来。

repeat penalty 的典型模式：

```rust
let start_at = tokens.len().saturating_sub(repeat_last_n);
let context = &tokens[start_at..];
let logits = apply_repeat_penalty(&logits, penalty, context)?;
```

`saturating_sub` 防止 `usize` 下溢。如果 `tokens.len() < repeat_last_n`，结果是 0。

常见坑：

- tokenizer 的 token id 类型要和模型 Embedding 输入约定匹配。
- EOS 检查应该发生在采样后、继续下一轮前。
- repeat penalty 读取历史 token，不应该修改历史 token。
- 字符串流式输出要处理 tokenizer 的 byte-level 或 sentencepiece 细节，不能简单按 token id 打印。

练习：

运行 tiny tour 的 `--eos-token 22`，确认命中 EOS 后停止且 EOS token 保留在输出里。下一步练习
可以把单 EOS 扩展成多个 EOS token，重点是不要把 CLI 解析细节散落到生成循环里。

---

## 21. `IndexOp` 和 `.i((.., seq_len - 1, ..))?`

典型位置：

```rust
use candle::IndexOp;

let last_hidden = hidden.i((.., seq_len - 1, ..))?.contiguous()?;
```

这行从 `[B, S, H]` 中取最后一个序列位置，得到 `[B, H]`。生成下一个 token 只需要最后位置的
hidden state。

`..` 是 Rust range 语法。组合在 tuple 里时，Candle 的 `IndexOp` trait 把它解释成多维索引。

直觉：

```text
(.., seq_len - 1, ..)
  │       │       │
  │       │       └── 保留 hidden 维全部
  │       └────────── 取 seq 维最后一个位置
  └────────────────── 保留 batch 维全部
```

为什么后面 `.contiguous()`：

索引得到的 Tensor 可能是 view。后续 `lm_head` 是 Linear/matmul 路径，连续布局更可控。

常见坑：

- `seq_len - 1` 要保证 `seq_len > 0`。
- `.i(...)` 可能改变 rank，例如取一个具体 index 会去掉对应维度。
- 索引可能返回非 contiguous view。

练习：

预测下面索引的输出 shape：

```rust
let x = Tensor::zeros((2, 3, 4), DType::F32, &Device::Cpu)?;
x.i((.., 2, ..))?;
x.i((0, .., ..))?;
x.i((.., 1..3, ..))?;
```

---

## 22. 泛型 shape API：`Into<Shape>`、tuple 和 `D::Minus1`

典型位置：

```rust
Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
x.reshape((batch, seq_len, hidden))?;
logits.argmax(candle::D::Minus1)?;
```

Candle 让很多 shape 参数接收 tuple、数组、Vec：

```rust
pub fn zeros<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self>
```

`S: Into<Shape>` 读作：“任何能转换成 Shape 的类型 S”。所以 `(2, 3)`、`[2, 3]`、`Vec<usize>`
都可以通过对应实现变成 `Shape`。

`D::Minus1` 表示最后一维。它比手写 `rank - 1` 更直接，也避免在调用点到处取 rank。

C++ 类比：

- `template <ShapeLike S>`
- 重载 tuple/vector/array
- 用 enum 或 sentinel 表示最后一维

常见坑：

- 泛型 shape API 很方便，但错误消息可能比固定类型复杂。
- `reshape((2, (), 4))` 这类 one-hole shape 让 Candle 自动推导一个维度；只能有一个 hole。
- `D::Minus1` 是维度选择，不是数值 `-1`。

练习：

把同一个 `Tensor::zeros` 分别写成 `(2, 3)`、`[2, 3]`、`vec![2, 3]`，确认都能编译。

---

## 23. trait、`dyn Trait`、`Box<dyn Trait>` 和关联类型

典型位置：

- `Module`
- `VarBuilder<'a> = VarBuilderArgs<'a, Box<dyn SimpleBackend + 'a>>`
- `BackendStorage`

trait 定义一组能力：

```rust
pub trait Module {
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}
```

两种分发方式：

```rust
fn run<M: Module>(m: &M, x: &Tensor)       // 静态分发
fn run(m: &dyn Module, x: &Tensor)         // 动态分发
```

`Box<dyn SimpleBackend>` 表示：堆上放一个实现了 `SimpleBackend` 的具体对象，但调用点只通过
trait object 使用它。

关联类型例子：

```rust
pub trait BackendStorage {
    type Device: BackendDevice;
    fn device(&self) -> &Self::Device;
}
```

`type Device` 表示每种 Storage 实现都有确定的 Device 类型：

```text
CpuStorage  -> CpuDevice
CudaStorage -> CudaDevice
```

常见坑：

- `dyn Trait` 有动态分发和对象安全规则，不是所有 trait 都能直接做 trait object。
- `Box<dyn Trait>` 是拥有一个 trait object；`&dyn Trait` 是借用一个 trait object。
- 关联类型用于表达类型关系，比到处写泛型参数更清晰。

练习：

解释为什么 `VarBuilder` 适合 `Box<dyn SimpleBackend>`：因为权重来源可能是 mmap、buffer、
HashMap、VarMap，但加载层的代码不想关心具体来源。

---

## 24. `Arc<RwLock<T>>` 和内部可变性

典型位置：

```rust
pub struct Tensor(Arc<Tensor_>);

storage: Arc<RwLock<Storage>>,
```

`Arc<T>` 是线程安全引用计数共享所有权。`RwLock<T>` 把读写互斥检查放到运行时。

为什么 Tensor 里有这套结构：

- 多个 Tensor view 可以共享同一份 Storage。
- 多个 Tensor 句柄可以共享同一个 Tensor_。
- 后端执行时需要安全地读或写 Storage。

C++ 类比：

| Rust | C++ 近似 |
| --- | --- |
| `Arc<T>` | `std::shared_ptr<T>`，引用计数是原子的 |
| `RwLock<T>` | `std::shared_mutex` + 被保护对象 |

重要差异：

- `Arc` 不等于自动可变共享。
- `RwLock` 不等于绕过安全，它只是把互斥规则延迟到运行时检查。
- 能否跨线程还取决于内部类型是否满足 `Send`/`Sync`。

常见坑：

- 不要为了“能共享”就把所有东西放进 `Arc<Mutex<_>>`。
- 模型权重适合共享；每请求 cache 和 RNG 通常不适合全局共享。
- 锁粒度过大会把并发 forward 串行化。

练习：

画出 `base`、`transposed`、`packed` 三个 Tensor 的关系：哪些共享 Storage，哪个可能有新
Storage？

---

## 25. 闭包 trait：`FnOnce`、`FnMut`、`Fn`

典型位置：

```rust
pub fn sample_f(&mut self, logits: &Tensor, f: impl FnOnce(&mut [f32])) -> Result<u32>
```

Rust 闭包根据如何使用捕获值自动实现不同 trait：

| trait | 含义 |
| --- | --- |
| `FnOnce` | 至少能调用一次，可能消费捕获值 |
| `FnMut` | 可多次调用，但可能修改捕获状态 |
| `Fn` | 可多次调用，只共享读取捕获状态 |

`sample_f` 里 `f` 只会被调用一次，用来在采样前修改概率数组。因此 `FnOnce` 足够宽松。

常见坑：

- `Fn` 最严格，`FnOnce` 最宽松。
- 如果闭包要修改捕获变量，通常只能实现 `FnMut` 或 `FnOnce`。
- 如果闭包 move 走捕获值，通常只能实现 `FnOnce`。

练习：

写一个闭包对概率数组做截断：

```rust
sampler.sample_f(&logits, |prs| {
    prs[0] = 0.0;
})?;
```

思考为什么它不需要捕获任何外部状态。

---

## 26. tracing 和性能观察

典型位置：

- LLaMA example 的 `--tracing`
- `tracing_chrome`
- 模型内部 span

tracing 用来回答：

- 时间花在 tokenizer、prefill、decode 还是采样？
- Attention、MLP、RoPE 哪个阶段慢？
- GPU kernel 是否被过多小操作打碎？

典型流程：

```text
enable tracing -> run inference -> 生成 trace-*.json -> Chrome tracing 查看时间线
```

只看整体 token/s 不够。至少要拆：

| 指标 | 说明 |
| --- | --- |
| TTFT | time to first token，主要受 prefill 影响 |
| decode token/s | 逐 token decode 吞吐 |
| peak memory | 权重、临时 Tensor、KV Cache 峰值 |
| backend time | CPU/GPU kernel 时间 |

常见坑：

- 首次运行可能包含编译、下载、cache warmup，不适合当稳态性能。
- GPU 异步执行会影响计时边界。
- tiny demo 的 token/s 不代表真实 LLM 性能，只适合观察计算模式。

练习：

给 tour 的 `run_generation` 增加 prefill/decode 分段计时，而不是只统计总 elapsed。

---

## 27. `unsafe`：边界小，不变量清楚

典型位置：

- `VarBuilder::from_mmaped_safetensors`
- backend 里的未初始化分配
- CUDA/Metal FFI

`unsafe` 不是“关闭 Rust 安全”。它只是允许少数编译器无法静态验证的操作，例如：

- 调用 unsafe function。
- 解引用裸指针。
- FFI。
- 读写 mutable static。
- 实现 unsafe trait。

读 unsafe 五步：

1. 哪个具体操作需要 unsafe？
2. 需要维护哪些额外不变量？
3. 谁保证这些不变量？
4. unsafe 块是否足够小？
5. 外面是否提供了 safe wrapper？

mmap 的典型不变量：

- 映射存活期间文件不能被不安全地截断或修改。
- 解析出的 Tensor view 不能比映射活得更久。
- dtype/shape/offset 必须符合文件元数据。

未初始化分配的典型不变量：

- 分配后必须在读取前完整写入。
- 错误路径不能读取未初始化内容。
- dtype 和 shape 计算出的字节大小必须正确。

常见坑：

- 不要把大段逻辑包进一个巨大 unsafe 块。
- unsafe 函数的调用者必须阅读 safety contract。
- safe wrapper 的意义是把不变量收束到小范围。

练习：

选择一个 unsafe 调用，写下：

```text
unsafe operation:
safety invariant:
who guarantees it:
safe wrapper boundary:
```

---

## 28. `Drop` 和资源生命周期

典型位置：

- mmap 文件映射。
- GPU buffer、stream、event。
- tracing guard。
- FFI handle wrapper。

Rust 的 `Drop` 类似 C++ RAII destructor：值离开作用域时释放资源。

示例里 tracing 常见模式：

```rust
let _guard = if args.tracing {
    let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
    tracing_subscriber::registry().with(chrome_layer).init();
    Some(guard)
} else {
    None
};
```

`_guard` 必须活到 main 结束，否则 trace writer 可能提前关闭。变量名前面的 `_` 表示“我知道它
没有被直接读取，但我要保留它的生命周期”。

常见坑：

- 不要手动调用 `drop` 方法；如果要提前释放，调用标准库 `drop(value)`。
- guard 类型常常靠 Drop 完成 flush、unlock、close。
- 资源 wrapper 里实现 Drop 时要确保释放函数不会 panic。

练习：

为一个 C handle 设计 Rust wrapper：

```rust
struct Handle {
    raw: NonNull<handle_t>,
}
```

写出 `new`、`Drop` 和一个 safe `run(&mut self, input: &[f32], output: &mut [f32])` 签名。

---

## 29. 生产推理的所有权边界

典型位置：

主讲义第 7 课的服务化设计。

一个合理的服务状态通常是：

```text
Arc<Model>       多请求共享，只读权重
Device           通常跟模型绑定
Tokenizer        视实现决定共享或每请求持有
RequestState     每请求独立
  - KvCache
  - tokens
  - sampler/RNG
  - cancellation
```

为什么这样分：

- 权重巨大，只读，适合共享。
- KV Cache 是请求历史，不能跨请求共享。
- RNG 是采样状态，共享会让请求互相影响。
- tokens 是请求输入输出历史，也必须独立。

常见坑：

- `Arc<Model>` 可以共享权重，但不代表 forward 里所有东西都应该全局加锁。
- 把 `KvCache` 放进模型对象里，会让模型从“可共享权重”变成“带请求状态”。
- continuous batching 需要显式调度和 cache allocator，不是简单 `Vec<Request>` 就完成。

练习：

画出两个并发请求 A/B 的状态图。标出哪些对象共享，哪些对象必须独立。

---

## 30. 最后总表：看到写法时先翻译成工程意图

| 写法 | 工程意图 |
| --- | --- |
| `Result<T>` | 这一步可能失败 |
| `?` | 失败向上传播 |
| `Option<T>` | 这个值可能不存在 |
| `match &x` | 只读解构，不移动 |
| `&[T]` | 借用一段连续元素 |
| `&mut T` | 独占修改状态 |
| `Tensor::new(..., device)` | host 数据进入 Tensor/Device 世界 |
| `unsqueeze(0)` | 加 batch 维 |
| `reshape` | 改 shape，通常不复制 |
| `transpose` | 改 stride/view，通常不复制 |
| `contiguous` | 必要时打包成 row-major |
| `Tensor::clone` | 复制句柄，不等于复制数据 |
| `Module::forward` | layer 的统一调用接口 |
| `VarBuilder::pp` | 累积参数路径前缀 |
| `Box<dyn Trait>` | 动态权重来源或动态能力对象 |
| `Arc<T>` | 共享所有权 |
| `RwLock<T>` | 运行时读写互斥 |
| `#[cfg]` | 条件编译，代码是否存在 |
| `unsafe` | 调用者承担额外不变量 |
| `Drop` guard | 作用域结束时释放或 flush 资源 |
