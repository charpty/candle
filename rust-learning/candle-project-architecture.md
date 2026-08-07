# Candle 项目级架构讲义：从 workspace 到一次 token 生成

这篇讲义回答一个问题：当你在 `candle-examples` 里运行一次 LLaMA
推理时，代码到底穿过了哪些层，每一层的职责边界是什么，出错时该往哪里查。

如果只背 API，很容易在真实项目里卡住。更可靠的读法是把 Candle 当成一个分层运行时：

```text
应用入口
  candle-examples: CLI / tokenizer / 下载权重 / 选择 device / 打印输出
      |
      v
模型实现
  candle-transformers: LLaMA、Qwen、BERT、Whisper、generation、pipeline
      |
      v
神经网络积木
  candle-nn: Linear / Embedding / Norm / Conv / KV Cache / VarBuilder / Optim
      |
      v
Tensor 运行时
  candle-core: Tensor / Shape / Layout / DType / Device / Storage / Autograd / Error
      |
      v
执行后端
  CPU / CUDA / Metal / dummy backend / 自定义 kernel
```

这不是“目录树介绍”。每一层都承担一部分稳定合同，上层代码只应该依赖这些合同，而不是越层猜底层实现。

## 1. Workspace 是 Candle 的工程边界

根目录 [Cargo.toml](../Cargo.toml) 定义了一个 workspace。核心成员包括：

| crate | 主要职责 | 读源码时先看哪里 |
| --- | --- | --- |
| `candle-core` | Tensor、Shape、Layout、DType、Device、Storage、算子分发、自动微分、文件格式基础能力 | [candle-core/src/lib.rs](../candle-core/src/lib.rs)、[tensor.rs](../candle-core/src/tensor.rs)、[storage.rs](../candle-core/src/storage.rs) |
| `candle-nn` | 神经网络层、初始化、优化器、KV cache、VarBuilder、常用 ops | [candle-nn/src/lib.rs](../candle-nn/src/lib.rs)、[linear.rs](../candle-nn/src/linear.rs)、[var_builder.rs](../candle-nn/src/var_builder.rs) |
| `candle-transformers` | 具体 transformer 模型、generation、quantized 层、pipeline | [candle-transformers/src/lib.rs](../candle-transformers/src/lib.rs)、[models/llama.rs](../candle-transformers/src/models/llama.rs)、[generation/mod.rs](../candle-transformers/src/generation/mod.rs) |
| `candle-examples` | 可运行 CLI 示例、模型下载、tokenizer、输出格式、feature 演示 | [candle-examples/Cargo.toml](../candle-examples/Cargo.toml)、[examples/llama/main.rs](../candle-examples/examples/llama/main.rs) |
| `candle-pyo3` | Python 绑定 | [candle-pyo3](../candle-pyo3) |
| `candle-datasets` | 数据集读取工具 | [candle-datasets](../candle-datasets) |
| `candle-wasm-*` | WASM 示例和测试 | [candle-wasm-examples](../candle-wasm-examples)、[candle-wasm-tests](../candle-wasm-tests) |
| `candle-ug` / `tensor-tools` | 图/工具相关扩展 | [candle-ug](../candle-ug)、[tensor-tools](../tensor-tools) |

根 `Cargo.toml` 还把 `candle-kernels`、`candle-metal-kernels`、`candle-onnx` 等目录放在
`exclude` 里。它们仍可能作为路径依赖被引用，但不是默认 workspace 成员。读构建错误时要先确认
crate 是否在当前 workspace 成员里，而不是只看目录是否存在。

## 2. 依赖方向：上层组合，下层不反向依赖上层

可以用下面的规则判断代码应该放在哪：

```text
candle-examples
  可以依赖 core / nn / transformers / hf-hub / tokenizers / clap

candle-transformers
  可以依赖 core / nn，并实现具体模型结构

candle-nn
  可以依赖 core，并提供通用 layer 与训练组件

candle-core
  不依赖具体模型，不知道 LLaMA、Qwen，也不负责 tokenizer
```

这个方向非常重要。比如：

- 一个通用 `matmul` dtype 检查应该在 `candle-core`。
- 一个 `Linear` 的输入 reshape 优化应该在 `candle-nn`。
- 一个 LLaMA 的 RoPE/cache 语义应该在 `candle-transformers`。
- 一个命令行参数 `--temperature` 应该在 `candle-examples` 或应用层。

如果你想改代码，先问一句：这个改动是不是会让下层知道上层业务？如果会，位置大概率错了。

## 3. 一次推理请求的真实链路

以 LLaMA CLI 或本仓库的 `rust-candle-tour` 为例，一次请求大致这样走：

```text
Args / prompt / seed / dtype / device
    |
    | parse tokenizer input
    v
Vec<u32> token ids
    |
    | Tensor::new(..., &device)?.unsqueeze(0)?
    v
Tensor [B,S] U32
    |
    | Embedding::forward
    v
hidden [B,S,H] floating dtype
    |
    | model blocks: norm -> attention -> mlp
    v
last hidden [B,H]
    |
    | lm_head Linear
    v
logits [B,V]
    |
    | LogitsProcessor
    v
next token
    |
    | push token, update request state, maybe stop on EOS
    v
next decode step
```

这里有三类状态，必须分开看。

| 状态类别 | 生命周期 | 例子 | 典型 bug |
| --- | --- | --- | --- |
| 模型态 | 进程或模型实例级 | 权重 Tensor、config、layer struct、VarBuilder 路径 | checkpoint 路径不匹配、shape 不匹配、dtype/device 放错 |
| 请求态 | 每个 prompt 独立 | token 列表、KV cache、sampler RNG、EOS、stream 输出 | cache 跨请求串扰、随机性不可复现、EOS 语义不一致 |
| 临时 Tensor 态 | 一次 forward 内部 | hidden、Q/K/V、scores、mask、logits | shape 维度误读、layout 非连续、device mismatch |

Rust 的所有权系统正好可以帮助你表达这三类状态：

- 模型 `forward(&self, ...)` 用共享借用读取权重。
- cache 更新用 `&mut cache` 表示请求态独占推进。
- 中间 Tensor 通过 `Result<Tensor>` 返回，错误不靠全局状态传递。

## 4. `candle-core`：Tensor 是运行时合同，不只是数组

[Tensor](../candle-core/src/tensor.rs) 的内部结构可以概括为：

```text
Tensor
  Arc<Tensor_>
      id: TensorId
      storage: Arc<RwLock<Storage>>
      layout: Layout
      op: BackpropOp
      is_variable: bool
      dtype: DType
      device: Device
```

这解释了几个常见现象：

- `Tensor::clone()` 通常是复制句柄，不是复制整块数据。
- `reshape`、`transpose`、`narrow` 这类操作可能只改 `Layout`。
- `contiguous()` 才可能触发真实拷贝，把 strided view 变成 row-major 存储。
- 同一个 API 可以在 CPU/CUDA/Metal 上运行，因为 Tensor 记录了 `Device`，Storage 做后端分发。

读 Tensor 代码时，先抓住四个字段：

| 字段 | 你要问的问题 |
| --- | --- |
| `shape` | 每个维度的业务含义是什么？例如 `[B,N,S,D]` 里的 `S` 和 `D` 不能混。 |
| `dtype` | 这是索引、浮点激活、量化权重还是 mask？ |
| `device` | 左右 Tensor 是否在同一设备？CPU Tensor 和 CUDA Tensor 不能直接 binary op。 |
| `layout` | 这是真正连续内存，还是一个 view？下一步 matmul 是否会触发拷贝或慢路径？ |

典型执行链是：

```text
Tensor::matmul
    |
    | shape / batching / k 检查
    v
self.storage().matmul(...)
    |
    | Storage enum dispatch
    v
CpuStorage::matmul / CudaStorage::matmul / MetalStorage::matmul
```

所以 debug matmul 时不要从 kernel 开始。先确认 Tensor 层的合同：rank、batch 维、`m/k/n`、dtype、device、layout。

## 5. `candle-nn`：层是通用积木，权重路径是轻量 ABI

`candle-nn` 做两类事情：

1. 提供通用层和 ops，例如 `Linear`、`Embedding`、`LayerNorm`、`Conv2d`、`Dropout`。
2. 提供模型加载和训练相关的工具，例如 `VarBuilder`、`VarMap`、`Init`、`Optimizer`。

一个重要细节：`Module` / `ModuleT` trait 定义在 `candle-core`，`candle-nn` 只是 re-export。
这让基础 Tensor crate 不需要依赖神经网络 crate，但上层仍能统一写：

```rust
use candle_nn::{linear, Module};

let layer = linear(in_dim, out_dim, vb)?;
let ys = layer.forward(&xs)?;
```

`Linear::forward` 里有一个很典型的工程优化：对连续的 3D/4D 输入，先 reshape 成二维大矩阵做
matmul，再 reshape 回来，避免 broadcasted matmul 的慢路径。这说明 layer 不只是薄封装，它会利用
Tensor layout 信息做性能决策。

`VarBuilder` 是读模型时最容易被低估的部分。它把权重来源抽象成 backend：

```text
HashMap<String, Tensor>
SafeTensors / mmaped safetensors
Npz / Pth
VarMap
Zeros
```

上层 layer 只关心：

```rust
vb.pp("model.layers.0.self_attn.q_proj")
  .get((out_dim, in_dim), "weight")?
```

最终路径变成：

```text
model.layers.0.self_attn.q_proj.weight
```

这就是 Candle 模型代码和 checkpoint 之间的 ABI。名字错、shape 错、dtype/device 转换失败，都会在这个边界暴露。

## 6. `candle-transformers`：模型代码把数学结构落到 Tensor 合同上

`candle-transformers` 负责具体模型族。以 [LLaMA](../candle-transformers/src/models/llama.rs) 为例，
你要重点看四件事：

| 结构 | 作用 | 容易误读的地方 |
| --- | --- | --- |
| `Config` / `LlamaConfig` | 模型超参、head 数、hidden size、RoPE、tie embedding | 配置不是注释，shape 全靠它推导 |
| `Cache` | causal mask、cos/sin、KV cache、device | 是请求态或推理态，不是权重 |
| `CausalSelfAttention` | q/k/v/o projection、RoPE、repeat_kv、scores | `S` 当前输入长度，`T` cache 总长度 |
| `Block` / `Mlp` / `Llama` | 层堆叠、norm、residual、lm_head | residual shape 必须保持 `[B,S,H]` |

一次 attention 的形状合同通常是：

```text
hidden:  [B, S, H]
q:       [B, N, S, D]
k/v:     [B, KV, S, D]
k cache: [B, KV, T, D]
repeat:  [B, N, T, D]
scores:  [B, N, S, T]
context: [B, S, H]
logits:  [B, V]
```

prefill 阶段经常 `S == T`，decode 阶段通常 `S == 1` 且 `T` 逐步增长。很多 cache bug 都来自把这两个维度混成“序列长度”。

## 7. `candle-examples`：示例不是核心库，但暴露真实工程问题

`candle-examples` 负责把库能力串成可运行程序：

- 用 `clap` 解析 CLI 参数。
- 用 `hf-hub` 下载模型文件。
- 用 `tokenizers` 做文本和 token id 转换。
- 选择 `Device::Cpu`、`Device::new_cuda(0)` 或 Metal。
- 构造 `VarBuilder`，加载 config 和权重。
- 维护生成循环、stream 输出、采样参数和 EOS 停止条件。

示例代码最适合学习“边界胶水”：

```text
外部世界字符串/文件/网络
    |
    v
强类型 Args / Config / Tensor / Result
    |
    v
模型 forward
    |
    v
token ids / text stream / tracing
```

如果你要把 Candle 嵌进服务，`candle-examples` 比模型文件更值得先读，因为服务里的大多数 bug 不在
GEMM kernel，而在参数校验、状态生命周期、缓存隔离和错误处理。

## 8. 后端和 feature：编译期开关不是运行时参数

Candle 同时有 Cargo feature 和运行时 device 两层开关：

| 层 | 例子 | 影响 |
| --- | --- | --- |
| Cargo feature | `--features cuda`、`--features metal`、`--features accelerate` | 编译哪些代码、链接哪些库 |
| 运行时参数 | `--cpu`、选择 CUDA device id | 本次 Tensor 放在哪个设备执行 |

常见错误是把二者混在一起：

```bash
# feature 属于 cargo，必须在 --example 前后按 Cargo 规则传
cargo run -p candle-examples --features cuda --example llama -- --prompt "..."

# --cpu 属于 example，必须放到 -- 后
cargo run -p candle-examples --example rust-candle-tour -- --cpu
```

源码里也有对应边界：

- `#[cfg(feature = "cuda")]` 决定 CUDA 后端代码是否参与编译。
- `Device` enum 决定运行时 Tensor 在 CPU/CUDA/Metal 哪边。
- `Storage` enum 决定实际 op 分发到哪个 storage 实现。

## 9. 一张定位问题的表

| 症状 | 第一检查点 | 第二检查点 | 常见修复 |
| --- | --- | --- | --- |
| `ShapeMismatchBinaryOp` | Tensor rank 和最后两维 | batch 维是否一致 | 打印 `--trace-shapes`，回到 `[B,S,H]` 合同 |
| `DTypeMismatchBinaryOp` | token ids 是否误入浮点 op | mask/logits dtype 是否被隐式假设 | 显式 `to_dtype`，但先确认语义 |
| `DeviceMismatchBinaryOp` | 权重和输入 device | 新建 Tensor 是否用了 `Device::Cpu` | 所有请求 Tensor 从同一个 `device` 创建 |
| `CannotFindTensor` | `vb.pp` 前缀 | checkpoint key | 打印权重路径，检查模型实现和 checkpoint 转换脚本 |
| 性能突然下降 | `is_contiguous()` | broadcasted matmul 或重复拷贝 | 在正确边界调用 `contiguous()`，避免在循环里反复 copy |
| cached/recompute 结果不同 | `S`、`T`、`index_pos` | mask 和 RoPE offset | 固定 greedy，逐步比较 logits 或 token |
| 空输入崩溃 | public API 是否校验 0 维 | 是否有 `len - 1` / `dim - 1` | 先返回 `Result` 错误，不让库 panic |

## 10. 真实 bug 案例

这次 bug hunt 累计修了 74 个 public API 边界问题，覆盖 Candle 的十八条典型边界：

| bug | 所在层 | 教学价值 |
| --- | --- | --- |
| `replication_pad2d` 空空间维度和 `pad > 1` | `candle-nn` 通用 Tensor op | Public API 要通过 `Result` 暴露普通非法输入；shape rank 正确不代表维度非空。 |
| `Llama::load` block 加载 `unwrap()` | `candle-transformers` 模型加载 | 返回 `Result<Self>` 的加载函数不能在内部 panic；checkpoint 缺权重应该变成可定位错误。 |
| `pixel_shuffle` / `pixel_unshuffle` 0 factor | `candle-nn` 通用 Tensor op | `usize` 除法必须先排除 0；合法输入路径还要用 round-trip 测试保护。 |
| `Cache` / `RotatingCache` 零容量 | `candle-nn` KV cache | `grow_by = 0` 会死循环，rotating index 会取模零；长期保留的 UT 要能避免测试进程卡死。 |
| `KvCache` / `RotatingKvCache` / `ConcatKvCache` 失败半更新 | `candle-nn` KV cache | 有状态组件的 fallible append 必须是失败原子性的，不能 K 成功、V 失败后污染请求态。 |
| `ScatteredCacheBuilder` batch mask 长度 | `candle-nn` KV cache | batch 级输入要显式校验长度，否则短 mask 静默错形状，长 mask 越界 panic。 |
| ViT `PatchEmbeddings` 配置 | `candle-transformers` 模型构造 | 由 config 派生 shape 前要先拒绝 0 和非整除关系，避免除零和后续 token 数不一致。 |
| conv groups / attention heads 配置 | `candle-nn` / `candle-transformers` | `groups`、`num_heads` 参与除法和 shape 派生，必须先校验再计算。 |
| LLaMA2.c / Granite / Voxtral loader | `candle-transformers` 模型加载 | 同一个 `unwrap()` 反模式会在不同模型复制；每个 public loader 都要独立回归测试。 |
| BatchNorm / loss 函数边界 | `candle-nn` 训练组件 | 训练态状态更新前要先校验样本数；loss 的空 batch 和非法超参数要在入口拒绝。 |
| Mimi transformer 配置 | `candle-transformers` 模型构造 | `num_heads`、`kv_repeat`、`d_model` 和 Sin 位置编码 channel 数都属于构造/forward 前置合同。 |
| Gemma4 vision/text 配置 | `candle-transformers` 多模态模型 | 新模型往往有更多分支配置，KV heads、RoPE head dim、patch/pooling 和空输入都要单独打 UT。 |
| Gemma4 audio Conformer 配置 | `candle-transformers` 多模态模型 | 分块 attention、relative position、SSCP conv 和 light conv 都会从 config 派生 shape，数组长度和 0 参数必须前置校验。 |
| Gemma4 multimodal mask 对齐 | `candle-transformers` glue 层 | 特殊 token mask 和 encoder embedding 数量必须严格对齐；单 batch、多 batch和数量不匹配都要测。 |
| Gemma4 audio forward 输入 | `candle-transformers` 多模态模型 | mask 不能靠 Tensor 广播补齐 batch/time；public forward 要先拒绝错形状和空时间维。 |
| Gemma4 vision pooling 边界 | `candle-transformers` 多模态模型 | runtime 图像尺寸也会让合法配置派生出 0 token；pooler 自己要防止零输出长度。 |
| Qwen3-VL vision 配置 | `candle-transformers` 多模态模型 | 新视觉模型的 head、patch、merge、position grid 都会参与除法/reshape/RoPE，需要集中校验。 |
| Qwen3-VL text 配置和空序列 | `candle-transformers` 多模态模型 | GQA 的 heads/KV heads/head_dim 要在构造期校验，`[B,0,H]` 空序列要在 forward 入口拒绝。 |

### 10.1 `replication_pad2d`

这次迭代在 [candle-nn/src/ops.rs](../candle-nn/src/ops.rs) 里发现并修复了一个典型公共 API
边界问题：

- `replication_pad2d(xs, pad)` 原来只支持 `pad = 0` 或 `pad = 1`，和 PyTorch 风格 API 名称不匹配。
- 当输入空间维度为空，例如 shape 为 `[1, 1, 0, 2]`，`pad = 1` 会在 `h - 1` 处触发
  `usize` 下溢，debug 构建下可能 panic。
- 公共 API 返回类型是 `Result<Tensor>`，因此这类非法输入应该变成清晰错误，而不是 panic。

修复后的行为：

- `pad = 0` 仍返回原 Tensor 句柄 clone。
- `pad >= 1` 支持多格 replicate padding。
- `h == 0 || w == 0` 返回 `"empty spatial dimension"` 错误。
- 新增测试覆盖 `pad = 2` 的数值结果和空空间维度拒绝。

完整报告见 [bug-hunt-report.md](./bug-hunt-report.md)。

### 10.2 `Llama::load`

[candle-transformers/src/models/llama.rs](../candle-transformers/src/models/llama.rs) 的
`Llama::load` 返回 `Result<Self>`，但原来在加载 block 时使用了 `unwrap()`：

```text
Block::load(...).unwrap()
```

这会让 checkpoint 缺少某个 block 权重时直接 panic，而不是把 `CannotFindTensor` 或
`UnexpectedShape` 这类错误返回给调用者。对 CLI、服务和批量转换工具来说，panic 会丢失可恢复边界。

修复后的实现用迭代器收集 `Result`：

```text
(0..num_hidden_layers)
  .map(|i| Block::load(...))
  .collect::<Result<Vec<_>>>()?
```

新增测试只提供 embedding 和 final norm 权重，让加载进入第 0 个 block 后缺
`model.layers.0.self_attn.q_proj.weight`。修复前该测试会 panic；修复后返回包含缺失路径的错误。

### 10.3 `pixel_shuffle` / `pixel_unshuffle`

[candle-nn/src/ops.rs](../candle-nn/src/ops.rs) 的 `pixel_shuffle(xs, upscale_factor)` 和
`pixel_unshuffle(xs, downscale_factor)` 都返回 `Result<Tensor>`，但原实现没有先拒绝 0 factor：

```text
c / upscale_factor / upscale_factor
h / downscale_factor
w / downscale_factor
```

当 factor 为 0 时，这些 public API 会直接除零 panic。修复后会先返回清晰错误，并额外检查
channel / spatial dims 的整除关系和输出维度溢出。新增 UT 覆盖了 zero factor、不可整除输入和
合法 shuffle/unshuffle round-trip。

### 10.4 KV cache 与 ViT patch 配置边界

新增的 7 个案例集中在 [candle-nn/src/kv_cache.rs](../candle-nn/src/kv_cache.rs) 和
[candle-transformers/src/models/vit.rs](../candle-transformers/src/models/vit.rs)：

| ID | 问题 | 新增 UT |
| --- | --- | --- |
| BUG-004 | `Cache::append` 在 `max_seq_len = 0` 时扩容步长为 0，可能死循环 | `cache_rejects_zero_max_seq_len_cpu` |
| BUG-005 | `RotatingCache` 零容量时 `positions` 取模零，append/mask 路径无效 | `rotating_cache_zero_max_seq_len_has_fallible_paths_cpu` |
| BUG-006 | `KvCache::append` V 失败后 K 已被写入 | `kv_cache_append_is_atomic_when_value_append_fails_cpu` |
| BUG-007 | `RotatingKvCache::append` V 失败后 K 的 offset/长度已改变 | `rotating_kv_cache_append_is_atomic_when_value_append_fails_cpu` |
| BUG-008 | `ConcatKvCache::append` V cat 失败后 K 已替换成新 Tensor | `concat_kv_cache_append_is_atomic_when_value_cat_fails_cpu` |
| BUG-009 | `ScatteredCacheBuilder::indices_and_mask` 不校验 `batch_mask` 长度 | `scattered_cache_rejects_batch_mask_len_mismatch_cpu` |
| BUG-010 | ViT `patch_size = 0` 除零，非整除 image/patch 会制造 token 数不一致 | `patch_embeddings_rejects_zero_patch_size` |

这组案例比通用 op 更接近推理系统真实风险：KV cache 是请求态，坏状态会污染同一个请求后续所有
decode step。修复时不能只让当前调用返回 `Err`，还要断言失败后对象内部状态没有变化。

### 10.5 conv、attention heads 与 loader 边界

第二轮新增 10 个案例，集中在 public constructor 和 public loader：

| ID | 问题 | 新增 UT |
| --- | --- | --- |
| BUG-011 | `conv1d` / no-bias 在 `groups = 0` 时除零 | `conv1d_rejects_zero_groups_cpu` |
| BUG-012 | `conv2d` / no-bias 在 `groups = 0` 时除零 | `conv2d_rejects_zero_groups_cpu` |
| BUG-013 | `conv_transpose1d` / no-bias 在 `groups = 0` 时除零 | `conv_transpose1d_rejects_zero_groups_cpu` |
| BUG-014 | ViT `SelfAttention::new` 在 `num_attention_heads = 0` 时除零 | `self_attention_rejects_zero_attention_heads` |
| BUG-015 | ViT `hidden_size` 不能被 heads 整除时静默截断 head size | `self_attention_rejects_hidden_size_not_divisible_by_heads` |
| BUG-016 | `llama2_c::Cache::new` 在 `n_heads = 0` 时除零 | `cache_rejects_zero_heads` |
| BUG-017 | `llama2_c::Llama::load` block 加载 `unwrap()` | `models::llama2_c::tests::load_returns_error_when_block_weights_are_missing` |
| BUG-018 | `Granite::Cache::new` 在 `num_attention_heads = 0` 时除零 | `cache_rejects_zero_attention_heads` |
| BUG-019 | `Granite::load` block 加载 `unwrap()` | `models::granite::tests::load_returns_error_when_block_weights_are_missing` |
| BUG-020 | `VoxtralLlama::load` block 加载 `unwrap()` | `models::voxtral::voxtral_llama::tests::load_returns_error_when_block_weights_are_missing` |

这批案例的判断口径更严格：不是“配置不推荐”，而是构造函数内部会直接除零、静默截断必要维度，
或在返回 `Result<Self>` 的加载函数里 panic。详细复现和修复见 [bug-hunt-report.md](./bug-hunt-report.md)。

这 74 个 bug 值得放进学习资料，因为它们展示了 Rust/Candle 源码审计的正确顺序：

1. 先看 public API 签名。返回 `Result` 的函数不应该轻易 panic。
2. 再看 shape 合同。4D Tensor 不等于空间维度一定非空。
3. 找 `usize` 除法/减法、index、narrow、reshape、`unwrap()` 等边界点。
4. 写最小测试复现。
5. 对有状态对象，失败测试要额外断言状态没有半更新。
6. 修复时保持已有语义不变，再扩展合理能力。

## 11. 学习者必须能讲出的完整链路

如果你真的掌握了 Candle 项目结构，应该能不看答案说清下面这段：

```text
CLI 参数决定 device、dtype、prompt 和采样策略。
prompt 被 tokenizer 或教学 parser 变成 Vec<u32>。
Tensor::new 把 token ids 放到同一个 device，并通过 unsqueeze 形成 [B,S]。
Embedding 用 token ids 查表，输出 [B,S,H] 浮点 Tensor。
LLaMA block 通过 RMSNorm、Attention、MLP 和 residual 保持 [B,S,H]。
Attention 里 Q/K/V reshape 到 heads，RoPE 使用 index_pos，KV cache 沿序列维追加。
最后位置 hidden 进入 lm_head，得到 [B,V] logits。
LogitsProcessor 根据 greedy/temperature/top-k/top-p 采样 next token。
生成循环把 token 放回请求态，检查 EOS，然后进入下一步 decode。
每个 Tensor op 在 candle-core 检查 shape/dtype/device/layout，再通过 Storage 分发到后端。
```

注意最后一句：模型数学和后端执行不是两套世界。它们在 Tensor 合同处接上。

## 12. 推荐阅读顺序

1. [rust-candle-tour README](../candle-examples/examples/rust-candle-tour/README.md)
2. [HTML 推理链路讲解站](./html/index.html)
3. [HTML 项目架构全景图](./html/architecture.html)
4. [一次生成全链路拆解](./html/trace-walkthrough.html)
5. [Candle 源码阅读路线图](./source-reading-playbook.md)
6. [Rust + Candle 动手实验手册](./rust-candle-labs.md)
7. [bug hunt 报告](./bug-hunt-report.md)
8. [Rust + Candle 水平测评题](./rust-candle-exam.md)

读完后不要停在“我懂了”。至少跑下面三组验证：

```bash
cargo run -p candle-examples --example rust-candle-tour -- --cpu --trace-shapes -n 2
cargo run -p candle-examples --example rust-candle-tour -- --cpu --compare-cache -n 4
cargo test -p candle-nn --test ops
cargo test -p candle-nn --test ops replication_pad2d_cpu
cargo test -p candle-transformers models::llama::tests::load_returns_error_when_block_weights_are_missing
```
