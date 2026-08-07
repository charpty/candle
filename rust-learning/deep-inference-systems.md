# Rust + Candle 深度专题：从类型系统到推理系统不变量

这篇文章比前面的讲义更深一层。前面的材料主要回答：

```text
这段 Rust/Candle 代码怎么写？
```

这篇回答：

```text
为什么这段代码必须这样组织？
它背后保护了哪些推理系统不变量？
如果改错，会破坏什么？
```

读 Candle 时，真正难的不是某个 API 名字，而是同时跟住六类事实：

1. 谁拥有数据，谁只是借用。
2. Tensor 的 shape 是否符合模型数学。
3. Tensor 的 layout 是否符合后端 kernel。
4. 数据在哪个 Device，是什么 DType。
5. 哪些状态跨 token、跨 layer、跨 request 保存。
6. 哪些边界 Rust 无法证明，需要人写出 safety invariant。

下面按这些不变量展开。

---

## 0. 先建立一张总图

一次 LLM 推理不是“一段 forward 代码”，而是一条状态机：

```text
外部输入
  prompt string
    |
    v
Tokenizer
  Vec<u32> tokens
    |
    v
Tensor::new + unsqueeze
  token Tensor [B, S] on Device
    |
    v
Embedding
  hidden [B, S, H]
    |
    v
N 个 Transformer Block
  每层读权重
  每层读写 KV Cache
    |
    v
last hidden [B, H]
    |
    v
lm_head
  logits [B, V]
    |
    v
LogitsProcessor
  repeat penalty / temperature / top-k / top-p / argmax
    |
    v
next token
    |
    v
更新 tokens / cache / rng / output stream
```

这条链路里有三类东西：

| 类别 | 例子 | 生命周期 |
| --- | --- | --- |
| 模型常量 | 权重、config、Device | 模型加载后长期存在 |
| 请求状态 | tokens、KV Cache、sampler/RNG | 每个请求独立 |
| 临时 Tensor | hidden、q/k/v、scores、logits | 一次 forward 内部 |

很多 Rust 签名就是在强制你不要混淆这三类东西。

---

## 1. 不变量一：模型权重共享，请求状态独立

推理服务中最重要的所有权边界是：

```text
Arc<Model>    可以共享
KvCache       必须请求独立
Sampler/RNG   必须请求独立
tokens        必须请求独立
```

为什么？

模型权重是只读大对象。多个请求共享它能节省内存，也符合语义：

```rust
model.forward(...)
```

通常接收 `&self`。`&self` 表示这次调用只需要共享读取模型本体。

KV Cache 是历史 token 的派生状态。请求 A 的第 20 个 token 不应该进入请求 B 的 attention。
因此 cache 必须作为请求状态传入：

```rust
model.forward(&input, index_pos, &mut cache, trace_shapes)?;
```

这里 `&mut cache` 有两层含义：

1. 这次 forward 会修改 cache。
2. 调用期间必须独占访问 cache。

如果模型把 cache 放进自己内部：

```rust
struct BadModel {
    weights: Weights,
    cache: KvCache,
}
```

这个类型会把“可共享的只读权重”和“每请求变化的历史状态”绑死在一起。后果是：

- 多请求共享模型时 cache 语义错误。
- 给整个 model 加锁会把并发推理串行化。
- 想清理一个请求的 cache 会影响模型对象本身。

更合理的边界：

```rust
struct Model {
    weights: Weights,
    config: Config,
}

struct RequestState {
    tokens: Vec<u32>,
    cache: KvCache,
    sampler: LogitsProcessor,
}
```

服务层再组合：

```text
Server
  model: Arc<Model>
  device: Device

Request A
  state: RequestState

Request B
  state: RequestState
```

这就是 Rust 所有权设计和推理系统设计对齐的地方：类型结构表达了系统边界。

### 读源码时怎么判断

看到一个字段或变量，问：

```text
它属于模型常量、请求状态，还是 forward 临时值？
```

如果属于模型常量，通常可以共享。

如果属于请求状态，必须明确生命周期。

如果属于临时 Tensor，尽量让它只活在最小作用域内。

---

## 2. 不变量二：Tensor 是句柄，不是裸 buffer

Candle 的 Tensor 可以近似理解为：

```text
Tensor
  Arc<Tensor_>
    storage: Arc<RwLock<Storage>>
    layout: Layout
    dtype: DType
    device: Device
```

这不是一段简单的 `float*`。

它至少回答五个问题：

| 问题 | 字段 |
| --- | --- |
| 数据在哪里 | `Storage` / `Device` |
| 元素怎么解释 | `DType` |
| 逻辑维度是什么 | `Shape` |
| 如何从逻辑坐标映射到底层存储 | `Layout` |
| 谁共享这份描述和存储 | `Arc` |

因此，下面几件事必须分清：

```rust
let y = x.clone();
```

通常只是克隆 Tensor 句柄，共享底层 Storage。

```rust
let y = x.transpose(1, 2)?;
```

通常创建一个新 Layout view，还是共享底层 Storage。

```rust
let y = x.contiguous()?;
```

如果 `x` 已连续，可能只是共享句柄；如果不连续，会分配新 Storage 并按 stride 打包。

```rust
let y = x.to_dtype(DType::F16)?;
```

改变 dtype 通常需要新 Storage。

### 为什么 attention 里频繁 `.contiguous()?`

Attention 中典型代码：

```rust
let q = self
    .q_proj
    .forward(x)?
    .reshape((batch, seq_len, num_heads, head_dim))?
    .transpose(1, 2)?
    .contiguous()?;
```

数学上，`transpose(1, 2)` 后 shape 已经对了：

```text
[B, S, N, D] -> [B, N, S, D]
```

但 layout 可能不是 row-major contiguous。后端 matmul/kernel 往往希望输入布局简单。于是
`.contiguous()?` 把“数学正确的 view”变成“后端友好的物理布局”。

你可以把它理解成两个阶段：

```text
reshape/transpose: 调整逻辑视图
contiguous: 必要时调整物理布局
```

### 深层判断：copy 成本在哪里出现

很多性能问题不是出在 matmul 本身，而是出在 matmul 前后为了满足 layout 要求做了额外 copy。

读到 `.contiguous()?` 时不要机械批评它。问：

1. 输入是否已经 contiguous？
2. 后续 kernel 是否要求 contiguous？
3. 这次 copy 的 Tensor 多大？
4. 它发生在 prefill 还是每个 decode token？
5. 能否通过前面的 layout 设计避免？

同一个 copy，如果只发生在初始化或 prefill，影响可能可接受；如果发生在每个 decode token 的每层，
就可能成为吞吐瓶颈。

---

## 3. 不变量三：shape 是模型数学的类型注释

Rust 类型系统不会把 `[B, S, H]` 写进 Tensor 类型里。Candle 的 shape 是运行时检查。

这意味着你读模型代码时，必须自己维护 shape 注释。

Tiny tour 的 attention：

```text
x       [B, S, H]
q_proj  [B, S, H]
reshape [B, S, N, D]
transpose
q       [B, N, S, D]
k       [B, N, T, D]
scores  [B, N, S, T]
probs   [B, N, S, T]
context [B, N, S, D]
output  [B, S, H]
logits  [B, V]
```

真实 LLaMA 还会引入：

```text
Nq   query heads
Nkv  key/value heads
GQA  Nq / Nkv groups
```

如果 `Nq != Nkv`，K/V 需要 repeat 或广播到与 Q head 兼容。

### shape 错误常见在哪

1. 忘记 batch 维。

```rust
Tensor::new(ctxt, device)?        // [S]
Tensor::new(ctxt, device)?.unsqueeze(0)?  // [1, S]
```

2. head_dim 不能整除。

```rust
hidden_size % num_heads == 0
```

3. 取最后 token 时 rank 变化没想清楚。

```rust
hidden.i((.., seq_len - 1, ..))?  // [B, H]
```

4. logits 给采样器前 batch 维没有 squeeze。

```rust
let logits = logits.squeeze(0)?;  // [V]
```

5. cache 的时间维 `T` 和 attention score 最后一维不一致。

```text
scores [B, N, S, T]
cache  [B, N, T, D]
```

### 如何训练 shape 直觉

运行：

```bash
cargo run -p candle-examples --example rust-candle-tour -- --cpu --trace-shapes -n 2
```

不要只看输出。拿纸手算一遍，再对照输出。

当你能不看代码写出 prefill 和 decode 的 shape 表，才算真正理解了 forward。

---

## 4. 不变量四：编译期 feature 与运行时 Device 分离

Candle 同时有两层“后端选择”：

```text
Cargo feature: 这个二进制编进了哪些能力？
Device:        这次 Tensor 实际放在哪里？
```

例如：

```bash
cargo run -p candle-examples --example llama --features cuda -- --sample-len 8
```

`--features cuda` 是 Cargo 参数。它决定 CUDA 代码和依赖是否编译。

```bash
cargo run -p candle-examples --example llama -- --cpu
```

`--cpu` 是 example 参数。它决定运行时选择 CPU。

两者不是同一层。

### 运行时分发

Tensor API 是统一的：

```rust
let c = a.matmul(&b)?;
```

但 Storage 内部会按实际后端分发：

```text
Storage::Cpu  -> CpuStorage::matmul
Storage::Cuda -> CudaStorage::matmul
Storage::Metal -> MetalStorage::matmul
```

这解释了 Candle 里为什么同时有：

1. trait：描述每个后端必须实现什么能力。
2. enum：让普通 Tensor 在运行时持有任意后端 Storage。

如果把 Tensor 写成强泛型：

```rust
Tensor<B>
```

静态分发可能更直接，但模型代码会到处携带 backend 类型，example 和模型库会复杂很多。

现在的设计更偏：

```text
模型代码非泛型、易用
后端实现内部承担分发和优化
```

这是一种框架工程取舍。

---

## 5. 不变量五：prefill 和 decode 是两种计算形态

LLM 推理不是简单重复同一个 forward。

第一次：

```text
prefill
input: prompt 全部 token
S = prompt_len
cache 从空变成 prompt_len
```

之后：

```text
decode
input: 最新 token
S = 1
cache 每轮增长 1
```

cached 模式：

```text
step 0: [1, 5, 9, 2]
step 1: [23]
step 2: [22]
step 3: [14]
```

recompute-all 模式：

```text
step 0: [1, 5, 9, 2]
step 1: [1, 5, 9, 2, 23]
step 2: [1, 5, 9, 2, 23, 22]
step 3: [1, 5, 9, 2, 23, 22, 14]
```

两者在 greedy 下应该生成相同 token，因为模型看到的历史上下文等价。

但计算量不同：

```text
cached:        prompt_len + generated_len * 1
recompute-all: prompt_len + (prompt_len + 1) + (prompt_len + 2) + ...
```

### 为什么 tiny demo 里 token/s 可能反直觉

Tiny tour 的模型太小，矩阵太小，打印、调度、内存分配、计时边界都可能支配耗时。因此它只适合
观察计算模式，不适合推导真实性能。

真实 LLM 中，KV Cache 的意义会非常明显：

- 避免每轮重算全部历史 token。
- 把 decode 变成 `S=1` 的增量计算。
- 代价是保存每层 K/V，占用显存，并带来 cache layout 管理问题。

---

## 6. 不变量六：随机状态也是状态

采样器：

```rust
let mut sampler = LogitsProcessor::from_sampling(seed, sampling);
let next_token = sampler.sample(&logits)?;
```

为什么 `sample` 需要 `&mut self`？

因为随机数生成器内部状态会推进。即使输入 logits 相同，采样器调用次数不同也可能得到不同结果。

这对测试和服务都有影响：

1. 比较 cached 和 recompute-all 时，最好用 greedy。
2. 每个请求要有独立 sampler/RNG。
3. 如果共享 RNG，请求之间会互相影响可复现性。
4. seed 只能保证同一调用序列可复现，不能保证不同控制流可复现。

### repeat penalty 也是状态读取

repeat penalty 不修改历史 tokens，但会读取最近一段历史：

```rust
let start_at = tokens.len().saturating_sub(repeat_last_n);
let context = &tokens[start_at..];
```

这里再次出现借用：`context` 是 tokens 的 slice。它应该在修改 tokens 前完成使用。

---

## 7. 错误处理不是附属品，而是模型边界

Candle 代码里大量 `Result` 不是噪音。它们表达边界：

| 边界 | 可能错误 |
| --- | --- |
| CLI parse | 参数类型错误 |
| config parse | JSON 字段缺失或类型错误 |
| weight load | 文件不存在、权重名不存在、shape 不符 |
| Tensor op | shape 不匹配、dtype 不支持、device mismatch |
| backend | kernel 不支持、显存不足 |
| tokenizer | 文件损坏、decode 错误 |

用 `unwrap()` 会把这些边界抹掉。教学代码和测试中也应尽量用 `Result<()>`：

```rust
#[test]
fn forward_extends_cache_from_prefill_to_decode() -> Result<()> {
    let prompt = Tensor::new(&[1u32, 5, 9], &device)?.unsqueeze(0)?;
    ...
    Ok(())
}
```

这让测试失败保留错误上下文，而不是在不相关的地方 panic。

### 什么时候可以 panic

不是永远不能 panic。适合 panic 的情况：

- 测试里断言不变量。
- 内部 bug，不是用户输入错误。
- 编译期或初始化时已经保证的 impossible state。

不适合 panic 的情况：

- 用户传错 prompt。
- 权重文件缺失。
- config 和模型不匹配。
- backend 不支持某 dtype。

---

## 8. VarBuilder 是权重命名系统，不只是加载器

真实模型权重是按名字存的：

```text
model.embed_tokens.weight
model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.k_proj.weight
...
lm_head.weight
```

模型加载代码不应该到处手写完整字符串，而是层层组合：

```rust
let layer_vb = vb.pp(format!("model.layers.{i}"));
let attn_vb = layer_vb.pp("self_attn");
let q_proj = linear_no_bias(hidden, hidden, attn_vb.pp("q_proj"))?;
```

这有几个好处：

1. 层级结构和模型结构一致。
2. 子模块只关心自己的局部名字。
3. 父模块决定前缀。
4. 权重来源可以替换：mmap、buffer、HashMap、VarMap。

`VarBuilder` 的动态后端：

```rust
Box<dyn SimpleBackend + 'a>
```

表示加载层不关心具体权重来源，只关心“按名字和 shape 给我 Tensor”这个能力。

这就是 trait object 的合适使用场景：变化点在权重来源，不在模型结构。

---

## 9. 生命周期：不要先背符号，先问是否保存引用

看到：

```rust
pub type VarBuilder<'a> = VarBuilderArgs<'a, Box<dyn SimpleBackend + 'a>>;
```

初学者容易陷入手算 `'a`。更好的问题是：

```text
返回对象内部是否可能保存了某个外部引用？
```

如果权重来源是 `&'a [u8]`，VarBuilder 就不能活得比这段 bytes 更久。

如果权重来源拥有 mmap 对象，可能可以返回 `'static` 风格的 builder，因为映射资源由对象自己持有。

生命周期的工程含义：

```text
引用不能比被引用资源活得更久。
```

在推理系统里，这直接对应资源生命周期：

- config bytes。
- safetensors bytes。
- mmap handle。
- tokenizer 文件内容。
- GPU resource。

不要把 lifetime 当成纯语法谜题。它在回答“这个对象里有没有借来的东西”。

---

## 10. unsafe 是系统边界，不是性能魔法

Candle 里常见 unsafe 来源：

1. mmap。
2. 未初始化分配。
3. FFI。
4. GPU resource handle。

unsafe 的正确读法：

```text
这里有一个 Rust 编译器无法证明的事实。
代码作者必须写出并维护这个事实。
外层 safe API 应该让普通调用者不用重新证明它。
```

mmap 的不变量：

```text
映射存活期间，文件不能被不安全地截断或修改。
Tensor view 不能比映射活得更久。
读取 offset/shape/dtype 必须符合文件元数据。
```

未初始化分配的不变量：

```text
读取前必须完整写入。
错误路径不能暴露未初始化内存。
shape * dtype size 的字节数必须正确。
```

FFI 的不变量：

```text
handle 必须有效。
buffer 指针必须非空且长度正确。
调用线程和 device context 必须符合外部库要求。
释放函数必须调用一次且只调用一次。
```

### 你应该怎么审 unsafe

不要只写：

```text
这里 unsafe 因为底层。
```

要写：

```text
unsafe operation:
safety invariant:
who guarantees it:
what happens if violated:
safe wrapper boundary:
```

如果写不出 invariant，就还没理解这段 unsafe。

---

## 11. Drop guard：生命周期有时靠“保留变量”表达

真实 example 里常见：

```rust
let _guard = if args.tracing {
    let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
    tracing_subscriber::registry().with(chrome_layer).init();
    Some(guard)
} else {
    None
};
```

这个 `_guard` 可能没有被直接读取，但必须活到作用域结束。它的 Drop 会 flush 或关闭 tracing
资源。

这和 C++ RAII 类似：

```cpp
TraceGuard guard = start_trace();
// guard destructor flushes trace
```

Rust 里变量名前的 `_` 不是“没用”。在这种场景下，它是在表达：

```text
我故意保留这个值，让它的 Drop 在作用域结束时运行。
```

常见同类对象：

- lock guard。
- tracing guard。
- mmap owner。
- GPU event/stream owner。
- temporary file guard。

---

## 12. 从 borrow checker 看系统设计

很多人把 borrow checker 当成语法障碍。读 Candle 时应反过来：

```text
borrow checker 在提醒你系统状态边界是否清楚。
```

例子一：tokens slice 与 push。

```rust
let ctxt = &tokens[start..];
tokens.push(next_token);
println!("{ctxt:?}");
```

编译器拒绝，因为 `ctxt` 可能悬空。系统含义：你不能一边持有历史视图，一边修改可能重分配的
历史容器。

例子二：cache 独占修改。

```rust
model.forward(&input, index_pos, &mut cache)?;
```

系统含义：forward 期间 cache 是这个请求的独占可变状态，不能被另一个任务同时写。

例子三：model 可共享。

```rust
&self
```

系统含义：模型权重不随请求变化，因此可以并发共享读取。

当你开始这样读 Rust，很多“为什么要这么写”的问题会变成系统设计问题，而不是语法问题。

---

## 13. 深入案例：tiny tour 的一轮 prefill

假设 prompt：

```text
[1, 5, 9, 2]
```

配置：

```text
B = 1
S = 4
V = 32
H = 16
N = 4
D = 4
```

主循环：

```rust
let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
let input = Tensor::new(ctxt, device)?.unsqueeze(0)?;
let logits = model.forward(&input, index_pos, &mut cache, trace_shapes)?;
```

解释：

```text
ctxt:
  类型 &[u32]
  借用 tokens
  长度 4

input:
  shape [1, 4]
  dtype integer
  device Cpu

model.forward:
  只读模型权重
  独占修改 cache
  可能失败
```

模型内部：

```text
token ids      [1, 4]
embedding      [1, 4, 16]
q/k/v new      [1, 4, 4, 4]
k/v cache      [1, 4, 4, 4]
scores         [1, 4, 4, 4]
context        [1, 4, 16]
last hidden    [1, 16]
logits         [1, 32]
```

状态变化：

```text
tokens 未修改，直到采样后 push
cache 从 None 变为 Some(K,V)
sampler 采样时可能推进 RNG
index_pos 从 0 增加到 4
```

这里每一步都可以映射到 Rust 签名：

| 事实 | Rust 表达 |
| --- | --- |
| ctxt 不拥有 tokens | `&[u32]` |
| model 不变 | `&self` |
| cache 变化 | `&mut KvCache` |
| Tensor op 可能失败 | `Result<Tensor>` |
| 采样器状态变化 | `&mut self` |

---

## 14. 深入案例：tiny tour 的一轮 decode

第二轮：

```text
tokens = [1, 5, 9, 2, 23]
context_size = 1
ctxt = [23]
index_pos = 4
```

input：

```text
[1, 1]
```

attention：

```text
q new          [1, 4, 1, 4]
k/v new        [1, 4, 1, 4]
k/v cached old [1, 4, 4, 4]
k/v cache new  [1, 4, 5, 4]
scores         [1, 4, 1, 5]
logits         [1, 32]
```

数学含义：

```text
当前 token 的 query
对全部 5 个历史/当前 key 做 attention
生成下一个 token 的 logits
```

系统含义：

```text
输入很短，但 cache 逐渐变长。
计算从重复处理历史 token，转为读取历史 K/V。
瓶颈从大 prefill matmul，逐渐转向小 batch decode、显存带宽、cache layout、kernel launch。
```

这就是为什么生产推理系统会关心：

- Paged Attention。
- continuous batching。
- cache allocator。
- prefix sharing。
- speculative decoding。

Candle 模型代码展示算法结构，但不是完整 serving scheduler。

---

## 15. 深入案例：真实 LLaMA 与 tiny tour 的差距

tiny tour 故意省略了很多真实组件：

| 组件 | tiny tour | 真实 LLaMA |
| --- | --- | --- |
| 层数 | 1 个 attention | 多个 Block |
| Norm | 省略 | RMSNorm |
| MLP | 省略 | gated MLP |
| RoPE | 省略 | 位置编码 |
| GQA | Q/K/V head 数相同 | Q heads 可能多于 K/V heads |
| Cache | 单层 Option | 每层 cache |
| Tokenizer | 直接 token id | tokenizer.json |
| 权重 | demo HashMap | SafeTensors |

但核心不变量一致：

```text
tokens -> Tensor [B,S]
hidden [B,S,H]
attention uses Q/K/V
cache stores K/V
last hidden -> logits
sampling -> next token
```

因此学习路径应该是：

1. 用 tiny tour 建立不变量。
2. 在真实 LLaMA 里找同构结构。
3. 只研究多出来的复杂度。

不要反过来：一开始就在真实 LLaMA 里同时理解所有组件。

---

## 16. 怎么从“能读”到“能改”

能读懂不等于能改。能改需要每次修改都保护不变量。

本分支已经实现了一个简化的 `--eos-token`。读这段改动时要看：

需要考虑：

1. CLI 参数类型：`Option<u32>`。
2. vocab 校验：EOS token 不能越界。
3. 停止位置：采样后、push 前还是 push 后？
4. 输出语义：是否打印 EOS？
5. 测试：命中 EOS 时 sample_len 没用完。
6. 与 tokenizer EOS 的关系：tiny tour 是简化版，不代表真实 tokenizer。

这不是一个“加 if”的小事。它跨越：

```text
CLI -> validation -> generation loop -> tokens state -> tests
```

本分支也已经把 `--trace-shapes` 扩展为 shape/dtype trace。读这段改动时要看：

需要考虑：

1. Tensor 是否暴露 dtype。
2. trace helper 是否仍只在 `--trace-shapes` 打印。
3. 输出是否过于嘈杂。
4. CPU/GPU 都能不能打印。
5. 测试是否需要固定输出。

每个改动都要问：

```text
我改动的是模型常量、请求状态，还是临时观测？
我是否改变了 shape/layout/device/dtype/state/error 边界？
```

---

## 17. 读源码的高阶习惯

### 17.1 先找不变量，再看实现细节

不要先问：

```text
这个宏怎么展开？
```

先问：

```text
这个宏在保护什么重复模式？
```

不要先问：

```text
这个 trait object 怎么调用？
```

先问：

```text
这里为什么需要隐藏具体类型？
```

### 17.2 每次只追一层

读 `Tensor::matmul` 时，先停在 `Storage::matmul`。

读 `Storage::matmul` 时，再看 CPU/CUDA/Metal 分支。

读 CPU matmul 时，再看 dtype/layout 分支。

不要一次把所有层展开。否则你会失去主线。

### 17.3 写 shape log 比脑补可靠

`--trace-shapes` 这类工具不是调试玩具，而是学习工具。

只要你开始怀疑 shape，就打印：

```rust
println!("{:?}", tensor.dims());
```

如果你开始怀疑 layout，就打印：

```rust
println!(
    "dims={:?} stride={:?} offset={} contiguous={}",
    tensor.dims(),
    tensor.stride(),
    tensor.layout().start_offset(),
    tensor.is_contiguous()
);
```

如果你开始怀疑性能，就不要只打印总时间。至少拆 prefill/decode。

### 17.4 每个理解都要能变成测试

理解 `parse_prompt`，就写非法输入测试。

理解 layout，就写 stride snapshot 测试。

理解 cache，就写 cached vs recompute-all 测试。

理解 sampling，就写参数组合测试。

学习源码时最稳的闭环是：

```text
阅读 -> 预测 -> 写小测试 -> 运行 -> 修正理解
```

---

## 18. 一张最终心智模型

把 Candle 推理系统压缩成一句话：

```text
Rust 类型系统约束数据和状态的所有权；
Candle Tensor 描述 shape/layout/device/dtype；
模型 forward 维护 Transformer 数学不变量；
KV Cache 维护跨 token 状态；
Storage/backend 把统一 Tensor API 分发到硬件；
Result/unsafe/Drop 把系统边界显式化。
```

再压缩成更短：

```text
所有权管状态，shape 管数学，layout 管内存，Device 管执行，Result/unsafe 管边界。
```

读任何 Candle 代码，都可以套这五个问题：

1. 状态是谁的？
2. 数学 shape 对不对？
3. 内存 layout 对不对？
4. 执行后端对不对？
5. 边界错误和 safety contract 对不对？

这五个问题答清楚，源码就不再是一堆 Rust 语法，而是一套可维护的推理系统。

---

## 19. 深度练习

### 练习 1：写一份 shape ledger

选真实 `CausalSelfAttention::forward`，写表：

```text
变量名：
源码位置：
shape：
stride 是否重要：
是否可能 copy：
是否读写 cache：
```

目标不是完全证明每个数字，而是训练“每个 Tensor 都有账”。

### 练习 2：审一次 `.contiguous()?`

任选一个 `.contiguous()?`，回答：

```text
它前面的 Tensor 为什么可能不 contiguous？
它后面的操作为什么希望 contiguous？
这个 Tensor 大小是多少？
它发生在 prefill 还是 decode？
有没有可能通过改变上游 layout 避免？
```

### 练习 3：重构一个请求状态

把 tiny tour 里的生成状态想象成：

```rust
struct RequestState {
    tokens: Vec<u32>,
    cache: KvCache,
    sampler: LogitsProcessor,
    index_pos: usize,
}
```

写出你会如何拆 `run_generation`。不用实现也可以，但必须说明每个字段为什么属于请求状态。

### 练习 4：画一张服务并发图

两个请求 A/B 同时生成，各生成 3 个 token。

画出：

```text
共享对象
请求 A 独立对象
请求 B 独立对象
每轮变化的字段
```

如果你把 cache 画到共享对象里，说明还没过关。

### 练习 5：unsafe contract

选择 mmap 或未初始化分配，写：

```text
safe wrapper:
unsafe operation:
required invariant:
who proves it:
failure mode:
test or review method:
```

如果你能写清楚这五行，就已经在用 Rust 工程师的方式读底层代码。
