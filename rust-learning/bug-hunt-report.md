# Candle bug hunt 报告：二十个 public API 边界修复

本报告记录一次真实源码审计：从 public API 合同出发，找到可复现问题，补测试并修复。

## 总结

本轮累计修复二十个问题：

本报告把问题算作 bug 的标准很明确：

- public 或模型构造 API 返回 `Result`，但普通非法输入会 panic、死循环，或绕过错误传播。
- 有状态组件在返回 `Err` 前已经污染内部状态。
- 构造函数接受了必然导致后续 shape/attention 语义错误的配置。
- 每个问题都有 UT；修复前 UT 会 panic、卡死、得到半更新状态，或拿不到期望的 `Err`。

| ID | 位置 | 问题 | 修复 |
| --- | --- | --- | --- |
| BUG-001 | [candle-nn/src/ops.rs](../candle-nn/src/ops.rs) | `replication_pad2d` 空空间维度可能 panic，且不支持 `pad > 1` | 显式拒绝空空间维度，并支持任意 `pad >= 1` |
| BUG-002 | [candle-transformers/src/models/llama.rs](../candle-transformers/src/models/llama.rs) | `Llama::load` block 加载时 `unwrap()`，缺权重会 panic | 用 `collect::<Result<Vec<_>>>()?` 保持错误传播 |
| BUG-003 | [candle-nn/src/ops.rs](../candle-nn/src/ops.rs) | `pixel_shuffle` / `pixel_unshuffle` 接收 0 factor 时会除零 panic | 先校验 factor，再检查整除关系和输出维度溢出 |
| BUG-004 | [candle-nn/src/kv_cache.rs](../candle-nn/src/kv_cache.rs) | `Cache::append` 在 `max_seq_len = 0` 时会因为 `grow_by = 0` 死循环 | append 前拒绝零容量，并用超时 UT 防止回归 |
| BUG-005 | [candle-nn/src/kv_cache.rs](../candle-nn/src/kv_cache.rs) | `RotatingCache` 零容量下 `positions` 会取模零，append/attn mask 语义无效 | `positions` 避免取模零，append/多 token mask 返回错误 |
| BUG-006 | [candle-nn/src/kv_cache.rs](../candle-nn/src/kv_cache.rs) | `KvCache::append` 先写 K 再写 V，V 失败会留下半更新状态 | 在临时副本上完成 K/V 更新，全部成功后提交 |
| BUG-007 | [candle-nn/src/kv_cache.rs](../candle-nn/src/kv_cache.rs) | `RotatingKvCache::append` 同样存在 K 成功、V 失败后的状态污染 | 使用临时 rotating cache 副本保证失败原子性 |
| BUG-008 | [candle-nn/src/kv_cache.rs](../candle-nn/src/kv_cache.rs) | `ConcatKvCache::append` 会先替换 K，再尝试拼接 V | 先计算 `next_k` / `next_v`，成功后一起写回 |
| BUG-009 | [candle-nn/src/kv_cache.rs](../candle-nn/src/kv_cache.rs) | `ScatteredCacheBuilder::indices_and_mask` 不校验 `batch_mask` 长度，可能 panic 或生成错误 batch 形状 | 显式要求 `batch_mask.len() == batch_size` |
| BUG-010 | [candle-transformers/src/models/vit.rs](../candle-transformers/src/models/vit.rs) | ViT `PatchEmbeddings::new` 对 `patch_size = 0` 除零，非整除 image/patch 也会制造后续 token 数不一致 | 构造阶段校验 patch size 和整除关系 |
| BUG-011 | [candle-nn/src/conv.rs](../candle-nn/src/conv.rs) | `conv1d` / `conv1d_no_bias` 对 `groups = 0` 除零 | 构造权重 shape 前校验 groups 和通道整除关系 |
| BUG-012 | [candle-nn/src/conv.rs](../candle-nn/src/conv.rs) | `conv2d` / `conv2d_no_bias` 对 `groups = 0` 除零 | 共用 groups 校验，非法分组返回错误 |
| BUG-013 | [candle-nn/src/conv.rs](../candle-nn/src/conv.rs) | `conv_transpose1d` / no-bias 对 `groups = 0` 除零 | 构造前拒绝零 groups 和不可整除通道 |
| BUG-014 | [candle-transformers/src/models/vit.rs](../candle-transformers/src/models/vit.rs) | ViT `SelfAttention::new` 对 `num_attention_heads = 0` 除零 | 先校验 attention heads 非零 |
| BUG-015 | [candle-transformers/src/models/vit.rs](../candle-transformers/src/models/vit.rs) | ViT `hidden_size % num_attention_heads != 0` 时静默截断 head size | 构造阶段拒绝不可整除配置 |
| BUG-016 | [candle-transformers/src/models/llama2_c.rs](../candle-transformers/src/models/llama2_c.rs) | `llama2_c::Cache::new` 对 `n_heads = 0` 除零 | 构造 RoPE cache 前校验 head 配置 |
| BUG-017 | [candle-transformers/src/models/llama2_c.rs](../candle-transformers/src/models/llama2_c.rs) | `llama2_c::Llama::load` block 加载 `unwrap()` | 用 `collect::<Result<Vec<_>>>()?` 传播缺权重错误 |
| BUG-018 | [candle-transformers/src/models/granite.rs](../candle-transformers/src/models/granite.rs) | `Granite::Cache::new` 对 `num_attention_heads = 0` 除零 | 构造 RoPE cache 前校验 head 配置 |
| BUG-019 | [candle-transformers/src/models/granite.rs](../candle-transformers/src/models/granite.rs) | `Granite::load` block 加载 `unwrap()` | 缺 block 权重返回 `Err` 而不是 panic |
| BUG-020 | [candle-transformers/src/models/voxtral/voxtral_llama.rs](../candle-transformers/src/models/voxtral/voxtral_llama.rs) | `VoxtralLlama::load` block 加载 `unwrap()` | 缺 block 权重返回 `Err` 而不是 panic |

## BUG-001：`replication_pad2d` 的边界行为

受影响函数：

- [candle-nn/src/ops.rs](../candle-nn/src/ops.rs) 的 `replication_pad2d`

问题类型：

- 空空间维度输入可能触发 `usize` 下溢 panic。
- `pad > 1` 被拒绝，和 `ReplicationPad2d` 语义不匹配。

修复状态：

- 已修复。
- 已补 CPU 定向测试。
- 新语义保持 `pad = 0` 和 `pad = 1` 兼容，并扩展支持 `pad >= 1`。

### 为什么这是 bug

函数签名是：

```rust
pub fn replication_pad2d(xs: &Tensor, pad: usize) -> Result<Tensor>
```

这意味着调用方应该通过 `Result` 接收非法输入错误。公共 API 内部不应该因为普通 shape
边界条件直接 panic。

原实现会在 `pad = 1` 分支里取最后一列/最后一行：

```rust
let right = xs.narrow(3, w - 1, 1)?;
let bottom = xs.narrow(2, h - 1, 1)?;
```

如果输入 shape 是 `[1, 1, 0, 2]`，`h - 1` 会对 `usize` 做下溢。debug 构建中这会 panic；
release 构建中也会得到不合理的大索引，再依赖后续 narrow 报错。两种行为都不是理想 API 合同。

另外，函数名和注释指向 PyTorch 风格的 `ReplicationPad2d`，但原实现只接受 `pad = 0` 和
`pad = 1`。对 `pad = 2` 这类正常用法返回 unsupported，不符合调用者预期。

### 最小复现

空空间维度：

```rust
use candle::{DType, Device, Tensor};

let xs = Tensor::zeros((1, 1, 0, 2), DType::F32, &Device::Cpu)?;
let _ = candle_nn::ops::replication_pad2d(&xs, 1)?;
```

多格 padding：

```rust
use candle::{Device, Tensor};

let xs = Tensor::new(&[[[[1f32, 2.], [3., 4.]]]], &Device::Cpu)?;
let padded = candle_nn::ops::replication_pad2d(&xs, 2)?;
assert_eq!(padded.dims(), &[1, 1, 6, 6]);
```

修复前第二段会失败，因为 `pad = 2` 不被支持。

### 修复策略

修复后的逻辑：

1. `pad == 0` 时保持原行为，返回 `xs.clone()`。
2. 对输入调用 `dims4()`，保持只接受 4D Tensor 的合同。
3. 如果 `h == 0 || w == 0`，通过 `candle::bail!` 返回清晰错误。
4. 对 `pad >= 1`，重复拼接左边界列和右边界列，再重复拼接上边界行和下边界行。

核心实现思路：

```text
原始 [B,C,H,W]
  |
  | cat left boundary pad 次 + 原 tensor + right boundary pad 次
  v
[B,C,H,W+2*pad]
  |
  | cat top boundary pad 次 + 中间结果 + bottom boundary pad 次
  v
[B,C,H+2*pad,W+2*pad]
```

### 新增测试

测试文件：

- [candle-nn/tests/ops.rs](../candle-nn/tests/ops.rs)

新增覆盖：

- `replication_pad2d_cpu`
  - 输入 `[1, 1, 2, 2]`。
  - 使用 `pad = 2`。
  - 断言输出 shape 为 `[1, 1, 6, 6]`。
  - 断言边界复制后的完整数值矩阵正确。

- `replication_pad2d_rejects_empty_spatial_dim_cpu`
  - 输入 `[1, 1, 0, 2]`。
  - 使用 `pad = 1`。
  - 断言返回错误，并且错误信息包含 `"empty spatial dimension"`。

已运行：

```bash
cargo test -p candle-nn --test ops replication_pad2d_cpu
cargo test -p candle-nn --test ops replication_pad2d_rejects_empty_spatial_dim_cpu
```

两条 CPU 定向测试均通过。

### 影响面

| 场景 | 修复前 | 修复后 |
| --- | --- | --- |
| `pad = 0` | 返回 clone | 返回 clone |
| `pad = 1` 且 `H/W > 0` | 正常复制一圈 | 正常复制一圈 |
| `pad > 1` 且 `H/W > 0` | unsupported | 支持多圈复制 |
| `H == 0` 或 `W == 0` | 可能 panic 或返回间接错误 | 返回明确错误 |
| 非 4D Tensor | `dims4()` 返回错误 | `dims4()` 返回错误 |

## BUG-002：`Llama::load` 中的 `unwrap()`

受影响函数：

- [candle-transformers/src/models/llama.rs](../candle-transformers/src/models/llama.rs) 的 `Llama::load`

问题类型：

- `Llama::load` 的返回类型是 `Result<Self>`。
- 但 block 加载路径中使用了 `unwrap()`。
- checkpoint 缺 block 权重或 block 权重 shape 错误时，错误会变成 panic，绕过调用方的错误处理。

### 为什么这是 bug

模型加载属于启动期或请求前准备阶段。缺权重、权重名不匹配、shape 不匹配都应该返回结构化错误，
让 CLI 或服务层决定如何展示、重试、换 checkpoint 或退出。

如果在库内部 `unwrap()`：

- CLI 用户看到的是 panic，而不是缺失权重路径。
- 服务进程可能因为一个坏 checkpoint 直接崩溃。
- 测试无法稳定断言错误类型和错误消息。
- 上层 `?` 错误传播链被截断。

### 最小复现

构造一个最小 LLaMA config，只提供顶层 embedding 和 final norm 权重，不提供第 0 个 block 权重：

```rust
let mut tensors = HashMap::new();
tensors.insert(
    "model.embed_tokens.weight".to_string(),
    Tensor::zeros((cfg.vocab_size, cfg.hidden_size), DType::F32, &device)?,
);
tensors.insert(
    "model.norm.weight".to_string(),
    Tensor::ones(cfg.hidden_size, DType::F32, &device)?,
);

let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
let err = Llama::load(vb, &cfg).unwrap_err().to_string();
assert!(err.contains("model.layers.0.self_attn.q_proj.weight"));
```

修复前，代码在加载 block 时 panic。修复后，调用者拿到包含缺失权重路径的错误。

### 修复策略

原逻辑可以概括为：

```text
map Block::load(...).unwrap()
```

修复为：

```text
map Block::load(...)
collect::<Result<Vec<_>>>()?
```

这样第一个 block 加载错误会沿 `?` 返回，并保留 Candle 的错误上下文。

### 新增测试

测试名：

- `models::llama::tests::load_returns_error_when_block_weights_are_missing`

已运行：

```bash
cargo test -p candle-transformers models::llama::tests::load_returns_error_when_block_weights_are_missing
```

测试通过。

### 影响面

| 场景 | 修复前 | 修复后 |
| --- | --- | --- |
| 权重完整 | 正常加载 | 正常加载 |
| 顶层 embedding/norm 缺失 | 返回错误 | 返回错误 |
| block 权重缺失 | panic | 返回包含 block 权重路径的错误 |
| block 权重 shape 错误 | panic | 返回 shape mismatch 错误 |

## BUG-003：`pixel_shuffle` / `pixel_unshuffle` 的 0 factor

受影响函数：

- [candle-nn/src/ops.rs](../candle-nn/src/ops.rs) 的 `pixel_shuffle`
- [candle-nn/src/ops.rs](../candle-nn/src/ops.rs) 的 `pixel_unshuffle`

问题类型：

- `pixel_shuffle(xs, 0)` 会执行 `c / upscale_factor / upscale_factor`，即除零 panic。
- `pixel_unshuffle(xs, 0)` 会执行 `h / downscale_factor` 和 `w / downscale_factor`，同样除零 panic。
- 两个函数都返回 `Result<Tensor>`，普通非法参数应该返回错误，而不是让库 panic。

### 最小复现

```rust
let xs = Tensor::zeros((1, 4, 2, 2), DType::F32, &Device::Cpu)?;
let _ = candle_nn::ops::pixel_shuffle(&xs, 0)?;
```

```rust
let xs = Tensor::zeros((1, 1, 2, 2), DType::F32, &Device::Cpu)?;
let _ = candle_nn::ops::pixel_unshuffle(&xs, 0)?;
```

修复前，这两个调用都会因为除零 panic。修复后，它们返回可断言的 `Result::Err`。

### 修复策略

修复后的 `pixel_shuffle` 会先检查：

- `upscale_factor > 0`
- `upscale_factor * upscale_factor` 不溢出
- channel 维度能被 `upscale_factor^2` 整除
- 输出 height / width 乘法不溢出

修复后的 `pixel_unshuffle` 会先检查：

- `downscale_factor > 0`
- height / width 能被 `downscale_factor` 整除
- 输出 channel 维度乘法不溢出

### 新增测试

测试文件：

- [candle-nn/tests/ops.rs](../candle-nn/tests/ops.rs)

新增覆盖：

- `pixel_shuffle_rejects_zero_upscale_factor_cpu`
- `pixel_unshuffle_rejects_zero_downscale_factor_cpu`
- `pixel_shuffle_rejects_non_divisible_channels_cpu`
- `pixel_unshuffle_rejects_non_divisible_spatial_dims_cpu`
- `pixel_shuffle_unshuffle_round_trip_cpu`

已运行：

```bash
cargo test -p candle-nn --test ops
```

结果：18 个 CPU 用例全部通过。

### 影响面

| 场景 | 修复前 | 修复后 |
| --- | --- | --- |
| shuffle factor 为 0 | panic | 返回明确错误 |
| unshuffle factor 为 0 | panic | 返回明确错误 |
| shuffle channel 不可整除 | reshape 间接错误 | 返回 channel 整除错误 |
| unshuffle H/W 不可整除 | reshape 间接错误 | 返回空间维整除错误 |
| 合法 shuffle/unshuffle | 正常 | 正常，round-trip 测试覆盖 |

## BUG-004：`Cache::append` 的零容量死循环

受影响函数：

- [candle-nn/src/kv_cache.rs](../candle-nn/src/kv_cache.rs) 的 `Cache::append`

问题类型：

- `Cache::new(dim, 0)` 会把 `max_seq_len` 和 `grow_by` 都设为 0。
- `append` 中扩容条件是 `current_seq_len + seq_len > max_seq_len`。
- 扩容循环每次只增加 `grow_by`，当 `grow_by == 0` 时永远无法前进。

### 最小复现

```rust
let mut cache = candle_nn::kv_cache::Cache::new(0, 0);
let t = Tensor::new(&[1f32], &Device::Cpu)?;
cache.append(&t)?;
```

修复前，这段代码不会返回。测试里用 `recv_timeout` 包了一层，防止回归时把测试进程永久卡住。

### 修复策略

`append` 一开始拿到 `seq_len` 后立即校验：

```rust
if self.max_seq_len == 0 {
    candle::bail!("cache max_seq_len must be greater than zero")
}
```

这里没有选择在 `new` 里返回 `Result<Self>`，因为那会改变公开构造函数签名，影响面更大。局部修复让非法容量在第一次真正使用时变成可处理错误。

### 新增测试

测试文件：

- [candle-nn/tests/kv_cache.rs](../candle-nn/tests/kv_cache.rs)

测试名：

- `cache_rejects_zero_max_seq_len_cpu`

断言点：

- `append` 必须在 1 秒内返回。
- 返回错误信息包含 `max_seq_len`。
- 失败后 `current_seq_len == 0`，缓存仍为空。

## BUG-005：`RotatingCache` 的零容量取模和无效写入

受影响函数：

- [candle-nn/src/kv_cache.rs](../candle-nn/src/kv_cache.rs) 的 `RotatingCache::positions`
- [candle-nn/src/kv_cache.rs](../candle-nn/src/kv_cache.rs) 的 `RotatingCache::append`
- [candle-nn/src/kv_cache.rs](../candle-nn/src/kv_cache.rs) 的 `RotatingCache::attn_mask`

问题类型：

- `positions(0)` 在 `max_seq_len = 0` 时会走 `seq_len <= max_seq_len` 分支。
- 该分支计算 `(offset + seq_len) % max_seq_len`，也就是对 0 取模。
- `append` 对零容量没有清晰语义，继续执行会创建零长度 buffer 并进入无效切片路径。
- 多 token attention mask 也依赖 rotating context，零容量下应该拒绝。

### 最小复现

```rust
let cache = candle_nn::kv_cache::RotatingCache::new(0, 0);
let _ = cache.positions(0);
```

修复前，`positions(0)` 会 panic。修复后，零容量下 `positions` 不再取模；真正需要写缓存或构造多 token mask 时返回错误。

### 修复策略

`positions` 是纯查询方法，不能返回 `Result`，所以它在零容量下退化为“即将追加 token 的绝对位置区间”：

```rust
if self.max_seq_len == 0 {
    return (self.current_seq_len..self.current_seq_len + seq_len).collect();
}
```

`append` 和 `attn_mask(seq_len > 1)` 本来就是 fallible API，因此直接 `bail!`。

### 新增测试

测试名：

- `rotating_cache_zero_max_seq_len_has_fallible_paths_cpu`

断言点：

- `positions(0)` 返回空列表，不 panic。
- `positions(2)` 返回 `[0, 1]`，不做取模。
- `attn_mask(1)` 仍返回 `None`，因为单 token decode 不需要 mask。
- `attn_mask(2)` 和 `append` 返回包含 `max_seq_len` 的错误。
- 失败后 `current_seq_len`、`offset` 和缓存内容保持初始状态。

## BUG-006：`KvCache::append` 的半更新状态

受影响函数：

- [candle-nn/src/kv_cache.rs](../candle-nn/src/kv_cache.rs) 的 `KvCache::append`

问题类型：

- 原实现顺序是 `self.k.append(k)?; self.v.append(v)?;`。
- 如果 K 追加成功、V 因维度错误失败，函数返回 `Err`，但 K cache 已经前进。
- 后续调用看到的 `current_seq_len` 来自 K，K/V 长度不一致，注意力缓存被污染。

### 最小复现

```rust
let mut cache = candle_nn::kv_cache::KvCache::new(1, 4);
let k = Tensor::zeros((1, 1), DType::F32, &Device::Cpu)?;
let invalid_v = Tensor::zeros(1, DType::F32, &Device::Cpu)?;
let _ = cache.append(&k, &invalid_v);
```

修复前，`append` 返回错误后 `cache.current_seq_len()` 会变成 1，`cache.k()?` 也已经是 `Some`。

### 修复策略

用 Rust 的值语义表达事务：

```rust
let mut next_k = self.k.clone();
let mut next_v = self.v.clone();
next_k.append(k)?;
next_v.append(v)?;
self.k = next_k;
self.v = next_v;
```

`?` 之前只改临时副本，只有两边都成功后才写回 `self`。

### 新增测试

测试名：

- `kv_cache_append_is_atomic_when_value_append_fails_cpu`

断言点：

- V 维度非法时 `append` 返回错误。
- `current_seq_len` 仍为 0。
- `k()` 和 `v()` 都仍然是 `None`。

## BUG-007：`RotatingKvCache::append` 的半更新状态

受影响函数：

- [candle-nn/src/kv_cache.rs](../candle-nn/src/kv_cache.rs) 的 `RotatingKvCache::append`

问题类型：

- rotating KV cache 的 append 同样先更新 K，再更新 V。
- V 失败后，K 的 `current_seq_len` 和 `offset` 已经改变。
- rotating cache 的错误更隐蔽，因为 buffer 位置也会被推进，下一次写入可能覆盖错误位置。

### 修复策略

和普通 `KvCache` 一样，先 clone 两个内部 rotating cache，在副本上尝试 append：

```rust
let mut next_k = self.k.clone();
let mut next_v = self.v.clone();
let out_k = next_k.append(k)?;
let out_v = next_v.append(v)?;
self.k = next_k;
self.v = next_v;
```

### 新增测试

测试名：

- `rotating_kv_cache_append_is_atomic_when_value_append_fails_cpu`

断言点：

- V append 失败后，`current_seq_len == 0`。
- `offset == 0`。
- K/V 当前数据都还是空。

## BUG-008：`ConcatKvCache::append` 的半更新状态

受影响函数：

- [candle-nn/src/kv_cache.rs](../candle-nn/src/kv_cache.rs) 的 `ConcatKvCache::append`

问题类型：

- 原实现先给 `self.k` 赋值，再给 `self.v` 赋值。
- 第二次 append 时，如果 K 可以 `cat`，但 V 因 batch/head/head_dim 不兼容而 `cat` 失败，K 会被扩长，V 保持旧长度。
- 这类 bug 在推理服务里很危险：一次坏请求可能污染同一个请求后续 decode 状态。

### 最小复现

```rust
let mut cache = candle_nn::kv_cache::ConcatKvCache::new(1);
let k1 = Tensor::zeros((1, 1, 2), DType::F32, &Device::Cpu)?;
let v1 = Tensor::zeros((1, 1, 2), DType::F32, &Device::Cpu)?;
cache.append(&k1, &v1)?;

let k2 = Tensor::zeros((1, 1, 2), DType::F32, &Device::Cpu)?;
let invalid_v2 = Tensor::zeros((2, 1, 2), DType::F32, &Device::Cpu)?;
let _ = cache.append(&k2, &invalid_v2);
```

修复前，第二次失败后 K 的 dim 1 已经从 1 变成 2，V 仍是 1。

### 修复策略

先构造两个局部变量：

```rust
let next_k = match &self.k { ... };
let next_v = match &self.v { ... };
self.k = Some(next_k);
self.v = Some(next_v);
```

这样 V 的 `cat` 失败时，K 的新值还没有写回。

### 新增测试

测试名：

- `concat_kv_cache_append_is_atomic_when_value_cat_fails_cpu`

断言点：

- 第二次 append 返回 shape 错误。
- 失败后 `current_seq_len` 仍是 1。
- K/V 维度都仍是 `[1, 1, 2]`。

## BUG-009：`ScatteredCacheBuilder` 的 batch mask 长度

受影响函数：

- [candle-nn/src/kv_cache.rs](../candle-nn/src/kv_cache.rs) 的 `ScatteredCacheBuilder::indices_and_mask`
- [candle-nn/src/kv_cache.rs](../candle-nn/src/kv_cache.rs) 的 `ScatteredCacheBuilder::indices_and_mask_abs`

问题类型：

- `batch_mask` 是外部传入的 batch 级开关，但原实现没有检查长度。
- 如果 `batch_mask` 太长，循环会访问 `self.indices[batch_i]` 越界并 panic。
- 如果 `batch_mask` 太短，函数不会 panic，但会生成 batch 维度过小的 indices/mask，和 builder 的 batch size 不一致。

### 最小复现

```rust
let mut cache =
    candle_nn::kv_cache::ScatteredCacheBuilder::new(2, 5, DType::F32, &Device::Cpu)?;
let _ = cache.indices_and_mask(1, &[true, false, true])?;
```

修复前，长度为 3 的 mask 会越界 panic。长度为 1 的 mask 会悄悄生成错误形状。

### 修复策略

在相对 mask 和绝对 mask 两条路径入口都加同一个合同校验：

```rust
if batch_mask.len() != self.batch_size() {
    candle::bail!(
        "batch_mask length mismatch, got {}, expected {}",
        batch_mask.len(),
        self.batch_size()
    )
}
```

### 新增测试

测试名：

- `scattered_cache_rejects_batch_mask_len_mismatch_cpu`

断言点：

- `seq_len < context` 的短 mask 返回错误。
- `seq_len < context` 的长 mask 返回错误，不 panic。
- `seq_len >= context` 的绝对 mask 路径也返回同类错误。

## BUG-010：ViT patch embedding 配置校验

受影响函数：

- [candle-transformers/src/models/vit.rs](../candle-transformers/src/models/vit.rs) 的 `PatchEmbeddings::new`

问题类型：

- `patch_size = 0` 时，`image_size / patch_size` 直接除零 panic。
- `image_size` 不能被 `patch_size` 整除时，原实现会向下取整计算 `num_patches`，但卷积输出 token 数按 stride/kernel 实际生成，后续 position embedding 长度可能对不上。

### 最小复现

```rust
let mut cfg = Config::vit_base_patch16_224();
cfg.patch_size = 0;
let _ = candle_transformers::models::vit::Embeddings::new(&cfg, false, vb)?;
```

修复前，构造阶段会 panic。修复后，返回包含 `patch_size` 的配置错误。

### 修复策略

在计算 `num_patches` 前检查：

```rust
if patch_size == 0 {
    candle::bail!("patch_size must be greater than zero")
}
if image_size % patch_size != 0 {
    candle::bail!("image_size {image_size} must be divisible by patch_size {patch_size}")
}
```

### 新增测试

测试文件：

- [candle-transformers/src/models/vit.rs](../candle-transformers/src/models/vit.rs)

测试名：

- `models::vit::tests::patch_embeddings_rejects_zero_patch_size`
- `models::vit::tests::patch_embeddings_rejects_non_divisible_image_size`

已运行：

```bash
cargo test -p candle-transformers models::vit::tests
```

结果：2 个 ViT 配置测试全部通过。

## BUG-011：`conv1d` groups 配置校验

受影响函数：

- [candle-nn/src/conv.rs](../candle-nn/src/conv.rs) 的 `conv1d`
- [candle-nn/src/conv.rs](../candle-nn/src/conv.rs) 的 `conv1d_no_bias`

问题类型：

- `Conv1dConfig::groups` 是公开配置字段。
- 原实现直接用 `in_channels / cfg.groups` 派生权重 shape。
- 当 `groups = 0` 时，构造函数会除零 panic。
- 当通道数不能被 groups 整除时，构造出的权重 shape 与 grouped convolution 合同不一致。

### 最小复现

```rust
let cfg = Conv1dConfig {
    groups: 0,
    ..Default::default()
};
let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
let _ = candle_nn::conv1d(2, 4, 3, cfg, vb)?;
```

修复前，这段代码在权重 shape 计算阶段 panic。修复后返回包含 `groups` 的错误。

### 修复策略

新增共享校验函数：

```rust
fn validate_groups(op: &str, in_channels: usize, out_channels: usize, groups: usize) -> Result<()>
```

它在所有使用 `cfg.groups` 派生 shape 的构造函数前检查：

- `groups > 0`
- `in_channels % groups == 0`
- `out_channels % groups == 0`

### 新增测试

测试文件：

- [candle-nn/tests/conv.rs](../candle-nn/tests/conv.rs)

测试名：

- `conv1d_rejects_zero_groups_cpu`

该测试同时覆盖 bias 和 no-bias helper。

## BUG-012：`conv2d` groups 配置校验

受影响函数：

- [candle-nn/src/conv.rs](../candle-nn/src/conv.rs) 的 `conv2d`
- [candle-nn/src/conv.rs](../candle-nn/src/conv.rs) 的 `conv2d_no_bias`

问题类型：

- `conv2d` 原实现同样在权重 shape 中计算 `in_channels / cfg.groups`。
- `groups = 0` 会除零 panic。
- 不可整除通道配置会让构造阶段接受一个不符合 grouped convolution 语义的权重 shape。

### 新增测试

测试名：

- `conv2d_rejects_zero_groups_cpu`

断言点：

- `conv2d` 和 `conv2d_no_bias` 都返回 `Err`。
- 错误信息包含 `groups`。

## BUG-013：`conv_transpose1d` groups 配置校验

受影响函数：

- [candle-nn/src/conv.rs](../candle-nn/src/conv.rs) 的 `conv_transpose1d`
- [candle-nn/src/conv.rs](../candle-nn/src/conv.rs) 的 `conv_transpose1d_no_bias`

问题类型：

- 转置卷积权重 shape 用到了 `out_channels / cfg.groups`。
- `groups = 0` 会在 public constructor 内部除零 panic。
- 这和普通 conv 一样，应该是 `Result::Err`，不是 panic。

### 新增测试

测试名：

- `conv_transpose1d_rejects_zero_groups_cpu`

已运行：

```bash
cargo test -p candle-nn --test conv
```

结果：3 个 conv groups 测试全部通过。

## BUG-014：ViT self-attention 零 head

受影响函数：

- [candle-transformers/src/models/vit.rs](../candle-transformers/src/models/vit.rs) 的 `SelfAttention::new`

问题类型：

- 原实现直接计算 `cfg.hidden_size / cfg.num_attention_heads`。
- 当 `num_attention_heads = 0` 时，模型构造阶段除零 panic。
- 该函数返回 `Result<Self>`，非法 config 应该返回错误。

### 修复策略

```rust
if cfg.num_attention_heads == 0 {
    candle::bail!("num_attention_heads must be greater than zero")
}
```

### 新增测试

测试名：

- `models::vit::tests::self_attention_rejects_zero_attention_heads`

## BUG-015：ViT self-attention head size 静默截断

受影响函数：

- [candle-transformers/src/models/vit.rs](../candle-transformers/src/models/vit.rs) 的 `SelfAttention::new`

问题类型：

- 原实现使用整数除法计算 `attention_head_size`。
- 当 `hidden_size = 5` 且 `num_attention_heads = 2` 时，head size 被截断为 2。
- Q/K/V 的 `all_head_size` 变成 4，不再等于 hidden size 5。
- 后续 attention 输出和 `SelfOutput` 的线性层合同不一致。

### 修复策略

```rust
if cfg.hidden_size % cfg.num_attention_heads != 0 {
    candle::bail!("hidden_size must be divisible by num_attention_heads")
}
```

### 新增测试

测试名：

- `models::vit::tests::self_attention_rejects_hidden_size_not_divisible_by_heads`

已运行：

```bash
cargo test -p candle-transformers models::vit::tests
```

结果：4 个 ViT 配置测试全部通过。

## BUG-016：`llama2_c::Cache::new` head 配置

受影响函数：

- [candle-transformers/src/models/llama2_c.rs](../candle-transformers/src/models/llama2_c.rs) 的 `Cache::new`

问题类型：

- RoPE cache 构造时先计算 `cfg.dim / cfg.n_heads`。
- `n_heads = 0` 会除零 panic。
- `dim` 不能被 `n_heads` 整除时，head size 也会被静默截断。

### 修复策略

在构造 RoPE 频率前检查：

```rust
if cfg.n_heads == 0 {
    candle::bail!("n_heads must be greater than zero")
}
if cfg.dim % cfg.n_heads != 0 {
    candle::bail!("dim must be divisible by n_heads")
}
```

### 新增测试

测试名：

- `models::llama2_c::tests::cache_rejects_zero_heads`

## BUG-017：`llama2_c::Llama::load` 中的 `unwrap()`

受影响函数：

- [candle-transformers/src/models/llama2_c.rs](../candle-transformers/src/models/llama2_c.rs) 的 `Llama::load`

问题类型：

- `Llama::load` 返回 `Result<Self>`。
- block 加载路径中使用 `unwrap()`。
- checkpoint 缺少第 0 层 block 权重时，函数 panic，而不是返回缺失 tensor 路径。

### 修复策略

```rust
let blocks: Vec<_> = (0..cfg.n_layers)
    .map(|i| Block::load(vb.pp(format!("model.layers.{i}")), &cfg))
    .collect::<Result<Vec<_>>>()?;
```

### 新增测试

测试名：

- `models::llama2_c::tests::load_returns_error_when_block_weights_are_missing`

测试只提供顶层 embedding、lm_head 和 final norm 权重，故意缺 block 权重。修复前 panic；修复后返回包含
`model.layers.0` 的错误。

## BUG-018：`Granite::Cache::new` head 配置

受影响函数：

- [candle-transformers/src/models/granite.rs](../candle-transformers/src/models/granite.rs) 的 `Cache::new`

问题类型：

- Granite RoPE 频率构造会计算 `hidden_size / num_attention_heads`。
- `num_attention_heads = 0` 会除零 panic。
- 不可整除配置会静默截断 head dim。

### 修复策略

在 `Cache::new` 入口检查：

- `num_attention_heads > 0`
- `hidden_size % num_attention_heads == 0`

### 新增测试

测试名：

- `models::granite::tests::cache_rejects_zero_attention_heads`

## BUG-019：`Granite::load` 中的 `unwrap()`

受影响函数：

- [candle-transformers/src/models/granite.rs](../candle-transformers/src/models/granite.rs) 的 `Granite::load`

问题类型：

- block 加载失败时使用 `unwrap()`。
- 缺权重、shape 错误等 checkpoint 问题会变成 panic。

### 修复策略

和 LLaMA 系列一致，使用 `collect::<Result<Vec<_>>>()?` 保留错误传播。

### 新增测试

测试名：

- `models::granite::tests::load_returns_error_when_block_weights_are_missing`

## BUG-020：`VoxtralLlama::load` 中的 `unwrap()`

受影响函数：

- [candle-transformers/src/models/voxtral/voxtral_llama.rs](../candle-transformers/src/models/voxtral/voxtral_llama.rs) 的 `VoxtralLlama::load`

问题类型：

- public `load` 返回 `Result<Self>`，但 block 加载用 `unwrap()`。
- 当 checkpoint 只有顶层权重而缺 block 权重时，构造过程 panic。

### 修复策略

```rust
let blocks: Vec<_> = (0..cfg.num_hidden_layers)
    .map(|i| Block::load(vb.pp(format!("model.layers.{i}")), cfg))
    .collect::<Result<Vec<_>>>()?;
```

### 新增测试

测试名：

- `models::voxtral::voxtral_llama::tests::load_returns_error_when_block_weights_are_missing`

已运行：

```bash
cargo test -p candle-transformers models::llama2_c::tests
cargo test -p candle-transformers models::granite::tests
cargo test -p candle-transformers models::voxtral::voxtral_llama::tests
```

结果：5 个模型加载/cache 配置测试全部通过。

## 二十个案例教什么

这些都不是复杂算法 bug，但很适合训练源码审计能力：

- Public API 返回 `Result`，就要主动检查普通非法输入是否会绕过 `Result` 变成 panic。
- `usize` 除法要先排除 0 factor。
- `usize` 的 `len - 1` 是 Rust 代码里常见边界风险点。
- Tensor rank 正确不代表每个维度都非空。
- 文档注释引用外部 API 时，行为应该尽量匹配读者预期。
- 模型加载代码里不应该用 `unwrap()` 处理 checkpoint 错误。
- 有状态组件的 fallible append 必须考虑失败原子性，不能让 `?` 之前已经污染 `self`。
- Batch 级输入必须显式校验长度；“短了静默错形状，长了越界 panic”比单纯报错更危险。
- 构造函数内部的配置派生值要先验合法性，再计算除法、取模、reshape 或 position embedding 长度。
- 模型加载函数中同一个 `unwrap()` 模式会在不同模型里重复出现；每个 public loader 都要独立测试。
- 分组卷积这类看似底层的配置字段，必须先校验再参与 shape 计算。
- 修复时不只补“报错测试”，还要补“正常扩展语义测试”，防止改成只会拒绝输入。

后续可以用同样方法继续审计：

- 模型加载函数内部是否还有 `unwrap()` 绕过 `Result`。
- cache 代码里是否存在把 head 维、seq 维、head_dim 维混淆的风险。
- 模型 config 里是否还有 `num_heads = 0`、`head_dim = 0`、`hidden_size % heads != 0` 这类可提前拒绝的边界。
