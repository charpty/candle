# Candle bug hunt 报告：一百一十六个 public API 边界修复

本报告记录一次真实源码审计：从 public API 合同出发，找到可复现问题，补测试并修复。

## 总结

本轮累计修复一百一十六个问题。BUG-001 到 BUG-020 覆盖通用 ops、KV cache、conv、ViT 和 loader
错误传播；BUG-021 到 BUG-044 继续扩展到 BatchNorm、loss、Mimi transformer、Gemma4 vision/text
这些更贴近模型配置和训练/推理边界的路径；BUG-045 到 BUG-053 继续覆盖 Gemma4 audio 的
Conformer attention 和 SSCP conv 配置；BUG-054 到 BUG-056 覆盖 Gemma4 multimodal embedding
和 mask 对齐语义；BUG-057 到 BUG-059 覆盖 Gemma4 audio forward 输入合同；BUG-060 到 BUG-062
继续覆盖 Gemma4 multimodal/vision 的运行期数量和 pooling 边界；BUG-063 到 BUG-069 覆盖
Qwen3-VL vision 构造期配置；BUG-070 到 BUG-074 覆盖 Qwen3-VL text attention 配置和空序列
forward 边界；BUG-075 到 BUG-080 覆盖 Qwen3-VL 外层 forward 的 per-batch 元数据和 image/video
placeholder span 合同；BUG-081 到 BUG-086 覆盖 Qwen3-VL vision runtime `grid_thw` 合同；
BUG-087 到 BUG-092 覆盖 PaddleOCR-VL vision 构造期配置；BUG-093 到 BUG-097 覆盖
PaddleOCR-VL text attention 配置和空序列 forward 合同；BUG-098 到 BUG-107 覆盖 PaddleOCR-VL
vision runtime `grid_thw` 合同；BUG-108 到 BUG-116 覆盖 PaddleOCR-VL text M-RoPE 运行期输入合同。

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
| BUG-021 | [candle-nn/src/batch_norm.rs](../candle-nn/src/batch_norm.rs) | BatchNorm training 每通道只有一个样本时会计算非有限 running variance | 训练路径拒绝 `batch_size <= 1`，失败前不污染状态 |
| BUG-022 | [candle-nn/src/loss.rs](../candle-nn/src/loss.rs) | `nll` / `cross_entropy` 空 batch 可能除零或先触发间接 reduce 错误 | 在 loss 入口拒绝空 target batch |
| BUG-023 | [candle-nn/src/loss.rs](../candle-nn/src/loss.rs) | `huber` 接收 `delta <= 0`，可能产生无意义甚至负 loss | 要求 `delta > 0` |
| BUG-024 | [candle-transformers/src/models/mimi/transformer.rs](../candle-transformers/src/models/mimi/transformer.rs) | Mimi self-attention `num_heads = 0` 构造期除零 | attention 配置先校验 heads 非零 |
| BUG-025 | [candle-transformers/src/models/mimi/transformer.rs](../candle-transformers/src/models/mimi/transformer.rs) | Mimi self-attention `kv_repeat = 0` 构造期除零 | attention 配置先校验 `kv_repeat > 0` |
| BUG-026 | [candle-transformers/src/models/mimi/transformer.rs](../candle-transformers/src/models/mimi/transformer.rs) | Mimi `d_model % num_heads != 0` 时 head dim 静默截断或后续 reshape 错 | 构造阶段要求 `d_model` 可被 heads 整除 |
| BUG-027 | [candle-transformers/src/models/mimi/transformer.rs](../candle-transformers/src/models/mimi/transformer.rs) | Mimi `num_heads % kv_repeat != 0` 时 KV head 数静默截断 | 构造阶段要求 heads 可被 `kv_repeat` 整除 |
| BUG-028 | [candle-transformers/src/models/mimi/transformer.rs](../candle-transformers/src/models/mimi/transformer.rs) | Mimi cross-attention `kv_repeat = 0` 同样除零 | cross-attention 复用 attention 配置校验 |
| BUG-029 | [candle-transformers/src/models/mimi/transformer.rs](../candle-transformers/src/models/mimi/transformer.rs) | `StreamingTransformer::new` 接受 0 层，forward 时访问 `layers[0]` panic | 构造阶段拒绝 `num_layers = 0` |
| BUG-030 | [candle-transformers/src/models/mimi/transformer.rs](../candle-transformers/src/models/mimi/transformer.rs) | Mimi Sin 位置编码输入通道 `< 2` 时 `half_dim - 1` 下溢 | forward 入口拒绝过小通道数 |
| BUG-031 | [candle-transformers/src/models/mimi/transformer.rs](../candle-transformers/src/models/mimi/transformer.rs) | Mimi Sin 位置编码 2 通道时出现 `0/0`，输出 NaN | `half_dim == 1` 使用有限频率 |
| BUG-032 | [candle-transformers/src/models/gemma4/vision.rs](../candle-transformers/src/models/gemma4/vision.rs) | Gemma4 vision attention `num_key_value_heads = 0` 构造期除零 | 构造前校验 KV heads 非零 |
| BUG-033 | [candle-transformers/src/models/gemma4/vision.rs](../candle-transformers/src/models/gemma4/vision.rs) | Gemma4 vision attention heads/KV heads 不整除时静默丢组 | 要求 `num_attention_heads % num_key_value_heads == 0` |
| BUG-034 | [candle-transformers/src/models/gemma4/vision.rs](../candle-transformers/src/models/gemma4/vision.rs) | Gemma4 `patch_size = 0` 构造可通过，forward 时除零 | `VisionTower::new` / `PatchEmbedder::new` 拒绝零 patch |
| BUG-035 | [candle-transformers/src/models/gemma4/vision.rs](../candle-transformers/src/models/gemma4/vision.rs) | Gemma4 `pooling_kernel_size = 0` forward 时除零 | 构造阶段拒绝零 pooling kernel |
| BUG-036 | [candle-transformers/src/models/gemma4/vision.rs](../candle-transformers/src/models/gemma4/vision.rs) | Gemma4 2D RoPE `head_dim` 非 4 的倍数时位置编码维度错配 | 要求 `head_dim` 是 `2 * ndim` 的非零倍数 |
| BUG-037 | [candle-transformers/src/models/gemma4/vision.rs](../candle-transformers/src/models/gemma4/vision.rs) | `VisionTower::forward(&[])` 直接访问第 0 张图 panic | 空 image batch 返回 `Err` |
| BUG-038 | [candle-transformers/src/models/gemma4/text.rs](../candle-transformers/src/models/gemma4/text.rs) | Gemma4 text sliding attention `num_key_value_heads = 0` 除零 | 按 layer 类型校验选中的 KV heads |
| BUG-039 | [candle-transformers/src/models/gemma4/text.rs](../candle-transformers/src/models/gemma4/text.rs) | Gemma4 text heads/KV heads 不整除时静默截断 GQA group | 要求 attention heads 可被 KV heads 整除 |
| BUG-040 | [candle-transformers/src/models/gemma4/text.rs](../candle-transformers/src/models/gemma4/text.rs) | Gemma4 text global KV heads 为 0 时 full attention 除零 | 校验 `num_global_key_value_heads` |
| BUG-041 | [candle-transformers/src/models/gemma4/text.rs](../candle-transformers/src/models/gemma4/text.rs) | Gemma4 text local `head_dim = 0` 可构造无效 RoPE/attention | 构造前校验 local head dim |
| BUG-042 | [candle-transformers/src/models/gemma4/text.rs](../candle-transformers/src/models/gemma4/text.rs) | Gemma4 text global `head_dim = 0` 可构造无效 global RoPE | 构造前校验 global head dim |
| BUG-043 | [candle-transformers/src/models/gemma4/text.rs](../candle-transformers/src/models/gemma4/text.rs) | `partial_rotary_factor > 1` 使 `half_dim - rope_angles` 下溢 panic | 要求 rotary factor 有限且在 `[0, 1]` |
| BUG-044 | [candle-transformers/src/models/gemma4/text.rs](../candle-transformers/src/models/gemma4/text.rs) | `TextModel::forward` 空 token 序列在 `seq_len - 1` 下溢 | forward / forward_embeds 拒绝空序列 |
| BUG-045 | [candle-transformers/src/models/gemma4/audio.rs](../candle-transformers/src/models/gemma4/audio.rs) | Gemma4 audio `conf_num_attention_heads = 0` 在 head dim 计算中除零 | audio attention 构造前校验 heads 非零 |
| BUG-046 | [candle-transformers/src/models/gemma4/audio.rs](../candle-transformers/src/models/gemma4/audio.rs) | `hidden_size % conf_num_attention_heads != 0` 时 head dim 静默截断 | 要求 hidden size 可被 heads 整除 |
| BUG-047 | [candle-transformers/src/models/gemma4/audio.rs](../candle-transformers/src/models/gemma4/audio.rs) | odd `hidden_size` 让 relative position sin/cos 维度少一列 | 要求 hidden size 为偶数 |
| BUG-048 | [candle-transformers/src/models/gemma4/audio.rs](../candle-transformers/src/models/gemma4/audio.rs) | `conf_attention_chunk_size = 0` 使 block 转换路径后续 `div_ceil(0)` | 构造阶段拒绝零 chunk size |
| BUG-049 | [candle-transformers/src/models/gemma4/audio.rs](../candle-transformers/src/models/gemma4/audio.rs) | `sscp_conv_channel_size` 少于两层时构造直接索引越界 | 校验 SSCP channel 配置长度 |
| BUG-050 | [candle-transformers/src/models/gemma4/audio.rs](../candle-transformers/src/models/gemma4/audio.rs) | `sscp_conv_kernel_size` 少于两层或内层少 time/freq 时索引越界 | 校验 SSCP kernel 配置形状 |
| BUG-051 | [candle-transformers/src/models/gemma4/audio.rs](../candle-transformers/src/models/gemma4/audio.rs) | SSCP stride 为 0 时 frequency 输出计算除零 | 校验 stride time/freq 都大于 0 |
| BUG-052 | [candle-transformers/src/models/gemma4/audio.rs](../candle-transformers/src/models/gemma4/audio.rs) | SSCP frequency kernel 大于 padded frequency 时 `usize` 下溢 | 构造阶段检查 kernel 能放入 padded frequency |
| BUG-053 | [candle-transformers/src/models/gemma4/audio.rs](../candle-transformers/src/models/gemma4/audio.rs) | `conf_conv_kernel_size = 0` 使 light conv causal padding 下溢 | 构造阶段拒绝零 conformer conv kernel |
| BUG-054 | [candle-transformers/src/models/gemma4/mod.rs](../candle-transformers/src/models/gemma4/mod.rs) | 单 batch multimodal embeddings 从序列开头铺开，mask 在后面时会取错或取 0 | 按 mask true 位置顺序 gather embeddings |
| BUG-055 | [candle-transformers/src/models/gemma4/mod.rs](../candle-transformers/src/models/gemma4/mod.rs) | 多 batch multimodal embedding helper 直接返回全 0 | 支持 row-major 跨 batch 放置 |
| BUG-056 | [candle-transformers/src/models/gemma4/mod.rs](../candle-transformers/src/models/gemma4/mod.rs) | embedding 数量和 mask token 数不一致时静默 pad/truncate | 数量不一致直接返回错误 |
| BUG-057 | [candle-transformers/src/models/gemma4/audio.rs](../candle-transformers/src/models/gemma4/audio.rs) | `audio_mel_mask` batch 为 1 时可广播到多个 audio batch | forward 要求 mask batch 精确匹配 audio batch |
| BUG-058 | [candle-transformers/src/models/gemma4/audio.rs](../candle-transformers/src/models/gemma4/audio.rs) | `audio_mel_mask` time 为 1 时可广播到多个 audio frame | forward 要求 mask time 精确匹配 audio time |
| BUG-059 | [candle-transformers/src/models/gemma4/audio.rs](../candle-transformers/src/models/gemma4/audio.rs) | 空 audio time 维会进入后续 conv/subsample 边界 | forward 入口拒绝空 time |
| BUG-060 | [candle-transformers/src/models/gemma4/mod.rs](../candle-transformers/src/models/gemma4/mod.rs) | encoder 产生 multimodal embedding 但 prompt 没有对应特殊 token 时静默丢特征 | 即使 mask token 数为 0，也校验 embedding 数量一致 |
| BUG-061 | [candle-transformers/src/models/gemma4/vision.rs](../candle-transformers/src/models/gemma4/vision.rs) | 图像太小导致 `num_patches / (pooling_kernel_size^2) == 0`，pooling 输出无效 | `VisionTower::encode_single` 拒绝零输出 token |
| BUG-062 | [candle-transformers/src/models/gemma4/vision.rs](../candle-transformers/src/models/gemma4/vision.rs) | `VisionPooler::forward(..., Some(0))` 会进入除零/无效 pooling | pooler 入口拒绝 `output_length = 0` |
| BUG-063 | [candle-transformers/src/models/qwen3_vl/vision.rs](../candle-transformers/src/models/qwen3_vl/vision.rs) | Qwen3-VL vision `num_heads = 0` 构造期除零 | 构造阶段拒绝零 heads |
| BUG-064 | [candle-transformers/src/models/qwen3_vl/vision.rs](../candle-transformers/src/models/qwen3_vl/vision.rs) | `hidden_size % num_heads != 0` 时 head dim 静默截断 | 要求 hidden size 可被 heads 整除 |
| BUG-065 | [candle-transformers/src/models/qwen3_vl/vision.rs](../candle-transformers/src/models/qwen3_vl/vision.rs) | RoPE head dim 非 4 的倍数时 cos/sin 维度与 Q/K head dim 错配 | 要求 vision head_dim 是 4 的非零倍数 |
| BUG-066 | [candle-transformers/src/models/qwen3_vl/vision.rs](../candle-transformers/src/models/qwen3_vl/vision.rs) | `patch_size = 0` 会进入 Conv3D stride/reshape 非法配置 | 构造阶段拒绝零 patch |
| BUG-067 | [candle-transformers/src/models/qwen3_vl/vision.rs](../candle-transformers/src/models/qwen3_vl/vision.rs) | `temporal_patch_size = 0` 会进入 Conv3D kernel/reshape 非法配置 | 构造阶段拒绝零 temporal patch |
| BUG-068 | [candle-transformers/src/models/qwen3_vl/vision.rs](../candle-transformers/src/models/qwen3_vl/vision.rs) | `spatial_merge_size = 0` 使 merger 后续取模/除法为 0 | 构造阶段拒绝零 spatial merge |
| BUG-069 | [candle-transformers/src/models/qwen3_vl/vision.rs](../candle-transformers/src/models/qwen3_vl/vision.rs) | `num_position_embeddings = 0` 被当作平方数接受，后续 grid 插值会下溢 | 构造阶段拒绝零 position embeddings |
| BUG-070 | [candle-transformers/src/models/qwen3_vl/text.rs](../candle-transformers/src/models/qwen3_vl/text.rs) | Qwen3-VL text `num_attention_heads = 0` 被构造函数接受，产生无效 zero-head attention | 构造阶段拒绝零 attention heads |
| BUG-071 | [candle-transformers/src/models/qwen3_vl/text.rs](../candle-transformers/src/models/qwen3_vl/text.rs) | `num_key_value_heads = 0` 在 `num_attention_heads / num_key_value_heads` 处 panic | 构造阶段拒绝零 KV heads |
| BUG-072 | [candle-transformers/src/models/qwen3_vl/text.rs](../candle-transformers/src/models/qwen3_vl/text.rs) | `num_attention_heads % num_key_value_heads != 0` 时 GQA group 数静默截断 | 要求 attention heads 可被 KV heads 整除 |
| BUG-073 | [candle-transformers/src/models/qwen3_vl/text.rs](../candle-transformers/src/models/qwen3_vl/text.rs) | `head_dim = 0` 会构造空 RoPE/RmsNorm，并让 softmax scale 失去语义 | 构造阶段拒绝零 head dim |
| BUG-074 | [candle-transformers/src/models/qwen3_vl/text.rs](../candle-transformers/src/models/qwen3_vl/text.rs) | `[B, 0, H]` 空序列会进入 attention/last-token 路径，得到间接 reshape 错误 | `forward_embeds` 入口拒绝空序列 |
| BUG-075 | [candle-transformers/src/models/qwen3_vl/mod.rs](../candle-transformers/src/models/qwen3_vl/mod.rs) | `seqlen_offsets = []` 且 `seqlen <= 1` 时直接索引 `seqlen_offsets[0]` panic | forward 入口拒绝空 offsets |
| BUG-076 | [candle-transformers/src/models/qwen3_vl/mod.rs](../candle-transformers/src/models/qwen3_vl/mod.rs) | `seqlen_offsets.len() != batch_size` 被静默接受，后续 RoPE/attention batch 语义不闭合 | 要求 offsets 数量匹配 batch |
| BUG-077 | [candle-transformers/src/models/qwen3_vl/mod.rs](../candle-transformers/src/models/qwen3_vl/mod.rs) | `seqlens = []` 在 `seqlens.iter().max().unwrap()` 处 panic | forward 入口拒绝空 seqlens |
| BUG-078 | [candle-transformers/src/models/qwen3_vl/mod.rs](../candle-transformers/src/models/qwen3_vl/mod.rs) | `seqlens.len() != batch_size` 被静默接受，per-batch 元数据和输入 batch 脱节 | 要求 seqlens 数量匹配 batch |
| BUG-079 | [candle-transformers/src/models/qwen3_vl/mod.rs](../candle-transformers/src/models/qwen3_vl/mod.rs) | image placeholder span `start > end` 在 `end - start` 处下溢 panic | 计算长度前校验 image span 顺序和范围 |
| BUG-080 | [candle-transformers/src/models/qwen3_vl/mod.rs](../candle-transformers/src/models/qwen3_vl/mod.rs) | video placeholder span `start > end` 在 `end - start` 处下溢 panic | 计算长度前校验 video span 顺序和范围 |
| BUG-081 | [candle-transformers/src/models/qwen3_vl/vision.rs](../candle-transformers/src/models/qwen3_vl/vision.rs) | `grid_thw` 只有两列时直接访问 `g[2]` panic | 要求 `grid_thw` shape 为 `[N, 3]` |
| BUG-082 | [candle-transformers/src/models/qwen3_vl/vision.rs](../candle-transformers/src/models/qwen3_vl/vision.rs) | `grid_thw` 多于三列时额外列被静默忽略 | 同样用 `[N, 3]` 精确 shape 校验拒绝额外列 |
| BUG-083 | [candle-transformers/src/models/qwen3_vl/vision.rs](../candle-transformers/src/models/qwen3_vl/vision.rs) | grid height 不能被 `spatial_merge_size` 整除时延迟成 reshape 错误 | 插值/reshape 前校验 height 整除关系 |
| BUG-084 | [candle-transformers/src/models/qwen3_vl/vision.rs](../candle-transformers/src/models/qwen3_vl/vision.rs) | grid width 不能被 `spatial_merge_size` 整除时延迟成 reshape 错误 | 插值/reshape 前校验 width 整除关系 |
| BUG-085 | [candle-transformers/src/models/qwen3_vl/vision.rs](../candle-transformers/src/models/qwen3_vl/vision.rs) | pixel token 行数和 `grid_thw` 派生 token 数不一致时只得到 add shape mismatch | forward 入口校验二者 token count 一致 |
| BUG-086 | [candle-transformers/src/models/qwen3_vl/vision.rs](../candle-transformers/src/models/qwen3_vl/vision.rs) | `h * w` 用 `u32` 乘法，超大 grid 在 debug 下溢出 panic | 使用 checked arithmetic 并返回 `Err` |
| BUG-087 | [candle-transformers/src/models/paddleocr_vl/vision.rs](../candle-transformers/src/models/paddleocr_vl/vision.rs) | PaddleOCR-VL vision `num_attention_heads = 0` 在 `head_dim()` 中除零 panic | 构造阶段拒绝零 attention heads |
| BUG-088 | [candle-transformers/src/models/paddleocr_vl/vision.rs](../candle-transformers/src/models/paddleocr_vl/vision.rs) | `hidden_size % num_attention_heads != 0` 时 head dim 静默截断 | 要求 hidden size 可被 heads 整除 |
| BUG-089 | [candle-transformers/src/models/paddleocr_vl/vision.rs](../candle-transformers/src/models/paddleocr_vl/vision.rs) | 2D RoPE head dim 非 4 的倍数时 cos/sin 维度和 Q/K head dim 不匹配 | 要求 vision head_dim 是 4 的非零倍数 |
| BUG-090 | [candle-transformers/src/models/paddleocr_vl/vision.rs](../candle-transformers/src/models/paddleocr_vl/vision.rs) | `patch_size = 0` 在 position embedding base grid 计算时除零 | 构造阶段拒绝零 patch |
| BUG-091 | [candle-transformers/src/models/paddleocr_vl/vision.rs](../candle-transformers/src/models/paddleocr_vl/vision.rs) | `image_size % patch_size != 0` 时 base position grid 静默截断 | 要求 image size 可被 patch size 整除 |
| BUG-092 | [candle-transformers/src/models/paddleocr_vl/vision.rs](../candle-transformers/src/models/paddleocr_vl/vision.rs) | `spatial_merge_size = 0` 被 Projector 构造接受，forward 后续除零 | Projector/VisionModel 构造阶段拒绝零 spatial merge |
| BUG-093 | [candle-transformers/src/models/paddleocr_vl/text.rs](../candle-transformers/src/models/paddleocr_vl/text.rs) | PaddleOCR-VL text `num_attention_heads = 0` 被构造函数接受，产生无效 zero-head attention | 构造阶段拒绝零 attention heads |
| BUG-094 | [candle-transformers/src/models/paddleocr_vl/text.rs](../candle-transformers/src/models/paddleocr_vl/text.rs) | `num_key_value_heads = 0` 在 GQA repeat 计算中除零 panic | 构造阶段拒绝零 KV heads |
| BUG-095 | [candle-transformers/src/models/paddleocr_vl/text.rs](../candle-transformers/src/models/paddleocr_vl/text.rs) | `num_attention_heads % num_key_value_heads != 0` 时 GQA group 数静默截断 | 要求 attention heads 可被 KV heads 整除 |
| BUG-096 | [candle-transformers/src/models/paddleocr_vl/text.rs](../candle-transformers/src/models/paddleocr_vl/text.rs) | `head_dim = 0` 会构造空 RoPE/attention，并让 softmax scale 失去语义 | 构造阶段拒绝零 head dim |
| BUG-097 | [candle-transformers/src/models/paddleocr_vl/text.rs](../candle-transformers/src/models/paddleocr_vl/text.rs) | `[B, 0, H]` 空序列进入 forward 后得到间接 reshape 错误 | `forward_embeds_with_mrope` 入口拒绝空序列 |
| BUG-098 | [candle-transformers/src/models/paddleocr_vl/vision.rs](../candle-transformers/src/models/paddleocr_vl/vision.rs) | `grid_thw` 只有两列时在 RoPE 路径直接切片越界 panic | 要求 `grid_thw` shape 为 `[N, 3]` |
| BUG-099 | [candle-transformers/src/models/paddleocr_vl/vision.rs](../candle-transformers/src/models/paddleocr_vl/vision.rs) | `grid_thw` 多于三列时额外列被静默忽略 | 同样用精确 `[N, 3]` 校验拒绝额外列 |
| BUG-100 | [candle-transformers/src/models/paddleocr_vl/vision.rs](../candle-transformers/src/models/paddleocr_vl/vision.rs) | 空 `grid_thw` 返回间接 `stack expects at least one tensor` 错误 | forward 入口返回明确 `grid_thw` 错误 |
| BUG-101 | [candle-transformers/src/models/paddleocr_vl/vision.rs](../candle-transformers/src/models/paddleocr_vl/vision.rs) | `t = 0` 进入后续空 stack/reshape 路径 | 校验每行 temporal 必须大于 0 |
| BUG-102 | [candle-transformers/src/models/paddleocr_vl/vision.rs](../candle-transformers/src/models/paddleocr_vl/vision.rs) | `h = 0` 进入后续空 stack/reshape 路径 | 校验每行 height 必须大于 0 |
| BUG-103 | [candle-transformers/src/models/paddleocr_vl/vision.rs](../candle-transformers/src/models/paddleocr_vl/vision.rs) | `w = 0` 进入后续空 stack/reshape 路径 | 校验每行 width 必须大于 0 |
| BUG-104 | [candle-transformers/src/models/paddleocr_vl/vision.rs](../candle-transformers/src/models/paddleocr_vl/vision.rs) | grid height 不能被 `spatial_merge_size` 整除时 projector 静默丢掉尾部 patch 行 | 入口校验 height 整除关系 |
| BUG-105 | [candle-transformers/src/models/paddleocr_vl/vision.rs](../candle-transformers/src/models/paddleocr_vl/vision.rs) | grid width 不能被 `spatial_merge_size` 整除时 projector 静默丢掉尾部 patch 列 | 入口校验 width 整除关系 |
| BUG-106 | [candle-transformers/src/models/paddleocr_vl/vision.rs](../candle-transformers/src/models/paddleocr_vl/vision.rs) | pixel token 行数和 `grid_thw` 派生 token 数不一致时只得到间接 narrow/reshape 错误 | forward 入口校验二者 token count 一致 |
| BUG-107 | [candle-transformers/src/models/paddleocr_vl/vision.rs](../candle-transformers/src/models/paddleocr_vl/vision.rs) | `build_cu_seqlens` 用 `u32` 计算 `h * w`，超大 grid 在 debug 下溢出 panic | 使用 `usize` 和 checked arithmetic 计算面积与累计长度 |
| BUG-108 | [candle-transformers/src/models/paddleocr_vl/text.rs](../candle-transformers/src/models/paddleocr_vl/text.rs) | `apply_multimodal_rotary_emb` 对 `position_ids[0] != 3` 使用 `assert_eq!` panic | 返回明确 `position_ids` shape 错误 |
| BUG-109 | [candle-transformers/src/models/paddleocr_vl/text.rs](../candle-transformers/src/models/paddleocr_vl/text.rs) | export 版 M-RoPE 同样对错误第一维 `assert_eq!` panic | export 路径复用同一 shape 校验 |
| BUG-110 | [candle-transformers/src/models/paddleocr_vl/text.rs](../candle-transformers/src/models/paddleocr_vl/text.rs) | `position_ids` batch 维不匹配时会把 q/k batch 静默广播扩大 | 要求 `position_ids` batch 精确匹配 q/k |
| BUG-111 | [candle-transformers/src/models/paddleocr_vl/text.rs](../candle-transformers/src/models/paddleocr_vl/text.rs) | `position_ids` seq_len 不匹配时只得到间接 broadcast 错误 | 要求 `position_ids` seq_len 精确匹配 q/k |
| BUG-112 | [candle-transformers/src/models/paddleocr_vl/text.rs](../candle-transformers/src/models/paddleocr_vl/text.rs) | 单图 M-RoPE `grid_h = 0` 被接受，image token 被当成普通文本推进 | 拒绝零 grid height |
| BUG-113 | [candle-transformers/src/models/paddleocr_vl/text.rs](../candle-transformers/src/models/paddleocr_vl/text.rs) | 单图 M-RoPE `grid_w = 0` 被接受，image token 被当成普通文本推进 | 拒绝零 grid width |
| BUG-114 | [candle-transformers/src/models/paddleocr_vl/text.rs](../candle-transformers/src/models/paddleocr_vl/text.rs) | image token 少于 `grid_h * grid_w` 时 helper 仍返回 Ok | 要求 image token 数等于 grid 面积 |
| BUG-115 | [candle-transformers/src/models/paddleocr_vl/text.rs](../candle-transformers/src/models/paddleocr_vl/text.rs) | image token 多于 `grid_h * grid_w` 时多余 token 被当成文本 | 同样校验 token count |
| BUG-116 | [candle-transformers/src/models/paddleocr_vl/text.rs](../candle-transformers/src/models/paddleocr_vl/text.rs) | image token 非连续时仍被按同一图像位置编码 | 要求 image token span 连续 |

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

## BUG-021：BatchNorm training 单样本状态污染

受影响函数：

- [candle-nn/src/batch_norm.rs](../candle-nn/src/batch_norm.rs) 的 `BatchNorm::forward_train`

问题类型：

- BatchNorm 训练态需要每个 channel 至少两个值才能估计无偏方差。
- 原实现把输入 flatten 成 `[C, N]` 后直接计算 `N / (N - 1)`。
- 当 `N = 1` 时会得到非有限系数，running variance 被污染。

### 为什么这是 bug

`BatchNorm::forward_train` 返回 `Result<Tensor>`，而且内部会更新 running mean/var。这里不只是“输入非法”：
如果函数在报错前已经把状态改坏，后续合法 batch 也会被污染。

训练态 BatchNorm 的关键合同是：

```text
输入合法性先于状态更新
```

也就是说，所有会影响 running statistics 的边界都必须在 mutation 前完成校验。

### 最小复现

```rust
let mut bn = candle_nn::batch_norm(1, Default::default(), vb)?;
let xs = Tensor::new(&[[1f32]], &Device::Cpu)?;
let err = bn.forward_train(&xs).unwrap_err().to_string();
assert!(err.contains("more than one value per channel"));
```

修复前，函数不会稳定返回这个错误，而且 running variance 可能被非有限值污染。

### 修复策略

在 flatten 后读取每通道样本数：

```rust
let batch_size = x.dim(1)?;
if batch_size <= 1 {
    candle::bail!("batch-norm training requires more than one value per channel");
}
```

然后再计算 mean、variance 和 running statistics。

新增测试：

- `batch_norm_train_rejects_single_value_per_channel`

已运行：

```bash
cargo test -p candle-nn --test batch_norm
```

分支：

- `agent/bug-021-batch-norm-single-value`

## BUG-022 到 BUG-023：loss 函数非法输入

受影响函数：

- [candle-nn/src/loss.rs](../candle-nn/src/loss.rs) 的 `nll`
- [candle-nn/src/loss.rs](../candle-nn/src/loss.rs) 的 `cross_entropy`
- [candle-nn/src/loss.rs](../candle-nn/src/loss.rs) 的 `huber`

### BUG-022：空 batch loss

`nll` 中有这类缩放：

```rust
let scale = -1.0 / b_sz as f64;
```

当 target batch 为空时，`b_sz = 0`。这不是模型训练中的正常数学对象，应返回清晰错误。
`cross_entropy` 更隐蔽：它会先进入 `log_softmax`，得到一个较间接的 reduce 错误，而不是告诉用户 target batch 为空。

修复后：

- `nll` 直接拒绝空 target。
- `cross_entropy` 在 `log_softmax` 之前检查输入 rank 和 batch 长度。

### BUG-023：Huber delta

Huber loss 的 `delta` 是阈值，必须大于 0。原实现接受 `delta <= 0`，会让分段公式失去语义，甚至生成负 loss。

修复后：

```rust
if delta <= 0. {
    candle::bail!("huber delta must be greater than zero");
}
```

新增测试：

- `nll_and_cross_entropy_reject_empty_batch`
- `huber_rejects_non_positive_delta`

已运行：

```bash
cargo test -p candle-nn --test loss
```

分支：

- `agent/bug-022-nll-empty-batch`
- `agent/bug-023-huber-delta-validation`

## BUG-024 到 BUG-031：Mimi transformer 配置与 Sin 位置编码

受影响文件：

- [candle-transformers/src/models/mimi/transformer.rs](../candle-transformers/src/models/mimi/transformer.rs)

这组问题来自同一个审计模式：构造函数返回 `Result<Self>`，但派生配置时先做除法、取模或索引。

### Attention head 配置

原 self-attention 和 cross-attention 构造路径都会计算：

```rust
let num_kv = cfg.num_heads / cfg.kv_repeat;
let kv_dim = num_kv * (embed_dim / cfg.num_heads);
```

这里有四类问题：

- `num_heads = 0`：`embed_dim / cfg.num_heads` 除零。
- `kv_repeat = 0`：`cfg.num_heads / cfg.kv_repeat` 除零。
- `d_model % num_heads != 0`：head dim 被整数除法截断。
- `num_heads % kv_repeat != 0`：KV head 数被截断，GQA 语义错。

修复策略是新增统一校验：

```rust
validate_attention_config(cfg)?;
```

这个 helper 在 self-attention、cross-attention 和 `StreamingTransformer::new` 都会调用。

### 0 层 StreamingTransformer

`StreamingTransformer::new` 原来允许 `num_layers = 0`，但 forward 中直接访问：

```rust
let pos = self.layers[0].self_attn.kv_cache.current_seq_len();
```

这是 public 构造成功、第一次 forward 才 panic 的典型边界 bug。修复为构造阶段拒绝 0 层。

### Sin 位置编码

Sin 分支按 runtime channel 数计算：

```rust
let half_dim = c / 2;
theta.powf(i as f32 / (half_dim - 1) as f32)
```

两个边界都不安全：

- `c < 2` 时 `half_dim - 1` 下溢。
- `c = 2` 时 `half_dim = 1`，分母是 0，`0 / 0` 生成 NaN。

修复后：

- `c < 2` 返回错误。
- 奇数 channel 返回错误，因为生成的 sin/cos 拼接维度无法匹配原 channel。
- `half_dim == 1` 使用 `vec![1f32]`，避免 NaN。

新增测试：

- `self_attention_rejects_zero_num_heads`
- `self_attention_rejects_zero_kv_repeat`
- `self_attention_rejects_non_divisible_model_width`
- `self_attention_rejects_non_divisible_kv_repeat`
- `cross_attention_rejects_zero_kv_repeat`
- `streaming_transformer_rejects_zero_layers`
- `sinusoidal_embedding_rejects_too_few_channels`
- `sinusoidal_embedding_with_two_channels_stays_finite`

已运行：

```bash
cargo test -p candle-transformers models::mimi::transformer::tests
```

分支：

- `agent/bug-024-mimi-zero-num-heads`
- `agent/bug-025-mimi-zero-kv-repeat`
- `agent/bug-026-mimi-dmodel-head-divisibility`
- `agent/bug-027-mimi-kvrepeat-head-divisibility`
- `agent/bug-028-mimi-cross-attn-zero-kvrepeat`
- `agent/bug-029-mimi-zero-layers`
- `agent/bug-030-mimi-sin-small-channel-panic`
- `agent/bug-031-mimi-sin-two-channel-nan`

## BUG-032 到 BUG-037：Gemma4 vision 输入与配置

受影响文件：

- [candle-transformers/src/models/gemma4/vision.rs](../candle-transformers/src/models/gemma4/vision.rs)

### Attention KV heads

Vision attention 原来直接保存：

```rust
num_kv_groups: num_heads / num_kv_heads
```

`num_kv_heads = 0` 会 panic；`num_heads` 不能被 `num_kv_heads` 整除时，GQA group 会被截断。
修复后构造前校验：

- `num_attention_heads > 0`
- `num_key_value_heads > 0`
- `num_attention_heads % num_key_value_heads == 0`

### patch / pooling 配置

`patch_size = 0` 和 `pooling_kernel_size = 0` 都能通过构造，但 forward 会分别在 patchify 和 pooling 输出长度计算中除零。
修复后 `VisionTower::new` 阶段拒绝它们；`PatchEmbedder::new` 也单独拒绝零 patch size。

### 2D RoPE head_dim

Gemma4 vision 的 2D RoPE 会把 head dim 分配给两个空间维度，并在每个维度里做 rotate-half。
所以 `head_dim` 必须是 `2 * ndim` 的非零倍数；在当前 2D 场景下就是 4 的倍数。

否则 cos/sin 的最后一维和 Q/K 的最后一维无法对齐，错误会延迟到 forward。修复为构造期报错。

### 空 image batch

`VisionTower::forward` 原来第一行就读取：

```rust
let device = pixel_values_list[0].device().clone();
```

空切片会 panic。修复为：

```rust
if pixel_values_list.is_empty() {
    candle::bail!("pixel_values_list must not be empty")
}
```

新增测试：

- `vision_attention_rejects_zero_key_value_heads`
- `vision_attention_rejects_non_divisible_key_value_heads`
- `vision_tower_rejects_zero_patch_size`
- `vision_tower_rejects_zero_pooling_kernel_size`
- `vision_tower_rejects_invalid_rope_head_dim`
- `vision_tower_rejects_empty_image_batch`

已运行：

```bash
cargo test -p candle-transformers models::gemma4::vision::tests
```

分支：

- `agent/bug-032-gemma4-vision-zero-kv-heads`
- `agent/bug-033-gemma4-vision-kv-head-divisibility`
- `agent/bug-034-gemma4-vision-zero-patch-size`
- `agent/bug-035-gemma4-vision-zero-pooling-kernel`
- `agent/bug-036-gemma4-vision-invalid-rope-head-dim`
- `agent/bug-037-gemma4-vision-empty-image-batch`

## BUG-038 到 BUG-044：Gemma4 text 配置与空序列

受影响文件：

- [candle-transformers/src/models/gemma4/text.rs](../candle-transformers/src/models/gemma4/text.rs)

### Layer 类型决定 KV heads

Gemma4 text 有 sliding attention 和 full/global attention 两类 layer。原代码在 `Attention::new` 中根据 layer 类型选择：

```rust
let (head_dim, num_kv_heads) = if is_sliding {
    (cfg.head_dim, cfg.num_key_value_heads)
} else {
    (cfg.global_head_dim, global_kv)
};
let num_kv_groups = num_heads / num_kv_heads;
```

风险点：

- sliding KV heads 为 0 会除零。
- global KV heads 为 0 也会除零。
- heads/KV heads 不整除会让 GQA group 被截断。
- local/global head dim 为 0 会构造无效 RoPE 和 attention scale。

修复策略是把“根据 layer 类型选 head_dim/KV heads”抽成 helper，再统一校验。

### partial_rotary_factor

global RoPE 会计算：

```rust
let rope_angles = (partial_rotary_factor * head_dim as f64 / 2.0) as usize;
inv_freq_vec.extend(std::iter::repeat_n(0f32, half_dim - rope_angles));
```

如果 `partial_rotary_factor > 1`，`rope_angles > half_dim`，`half_dim - rope_angles` 下溢 panic。
修复后要求 factor 有限且在 `[0, 1]`。

### 空 token 序列

`TextModel::forward_embeds` 最后取最后一个 token 的 logits：

```rust
xs.narrow(1, seq_len - 1, 1)?
```

当 `seq_len = 0` 时会 `usize` 下溢。修复后 `forward` 和 `forward_embeds` 都先拒绝空序列。

新增测试：

- `text_model_rejects_zero_sliding_key_value_heads`
- `text_model_rejects_non_divisible_key_value_heads`
- `text_model_rejects_zero_global_key_value_heads`
- `text_model_rejects_zero_local_head_dim`
- `text_model_rejects_zero_global_head_dim`
- `text_model_rejects_invalid_partial_rotary_factor`
- `text_model_rejects_empty_input_sequence`

已运行：

```bash
cargo test -p candle-transformers models::gemma4::text::tests
```

分支：

- `agent/bug-038-gemma4-text-zero-sliding-kv-heads`
- `agent/bug-039-gemma4-text-kv-head-divisibility`
- `agent/bug-040-gemma4-text-zero-global-kv-heads`
- `agent/bug-041-gemma4-text-zero-local-head-dim`
- `agent/bug-042-gemma4-text-zero-global-head-dim`
- `agent/bug-043-gemma4-text-invalid-rotary-factor`
- `agent/bug-044-gemma4-text-empty-input-sequence`

## BUG-045 到 BUG-053：Gemma4 audio Conformer 配置

受影响文件：

- [candle-transformers/src/models/gemma4/audio.rs](../candle-transformers/src/models/gemma4/audio.rs)

Gemma4 audio 的结构比 text/vision 更容易出边界问题，因为它同时有：

- SSCP 两层 2D conv projection。
- Conformer attention 的 chunk/block/context 计算。
- relative position embedding。
- light conv1d 的 causal padding。

这些组件都从 config 派生 shape，因此 public `AudioModel::new` 必须先校验配置，再进入权重 shape
构造或后续 forward。

### Attention 与 relative position embedding

原实现直接计算：

```rust
let head_dim = channels / num_heads;
```

这带来三个边界：

- `num_heads = 0` 除零。
- `hidden_size % num_heads != 0` 时 head dim 被截断，Q/K/V reshape 语义错误。
- `hidden_size` 为奇数时，relative position embedding 的 sin/cos 拼接只有 `hidden_size - 1` 列，
  但 `pos_proj` 期望输入维度仍是 `hidden_size`。

修复后 `validate_audio_attention_config` 要求：

- `hidden_size > 0`
- `hidden_size` 为偶数
- `conf_num_attention_heads > 0`
- `hidden_size % conf_num_attention_heads == 0`
- `conf_attention_chunk_size > 0`

### SSCP conv projection

SSCP projection 写死使用两层 conv：

```rust
for i in 0..2 {
    let kernel_w = cfg.sscp_conv_kernel_size[i][1];
    let stride_w = cfg.sscp_conv_stride_size[i][1];
    let f_out = (f_in_padded - kernel_w) / stride_w + 1;
}
```

所以配置数组的长度、内层长度、stride 和 kernel 都必须提前检查：

- channel/kernel/stride 配置至少有两层。
- kernel/stride 每层至少包含 time 和 frequency 两个值。
- kernel 和 stride 都大于 0。
- frequency kernel 不能大于当前 padded frequency，否则 `f_in_padded - kernel_w` 下溢。

### Light conv kernel

Conformer light conv 原来计算：

```rust
causal_padding: cfg.conf_conv_kernel_size - 1
```

`conf_conv_kernel_size = 0` 会直接下溢。修复后构造阶段拒绝零 kernel。

新增测试：

- `audio_model_rejects_zero_attention_heads`
- `audio_model_rejects_non_divisible_attention_heads`
- `audio_model_rejects_odd_hidden_size`
- `audio_model_rejects_zero_attention_chunk_size`
- `audio_model_rejects_short_sscp_channel_config`
- `audio_model_rejects_short_sscp_kernel_config`
- `audio_model_rejects_zero_sscp_stride`
- `audio_model_rejects_oversized_sscp_kernel`
- `audio_model_rejects_zero_conformer_conv_kernel`

已运行：

```bash
cargo test -p candle-transformers models::gemma4::audio::tests
```

分支：

- `agent/bug-045-gemma4-audio-zero-attention-heads`
- `agent/bug-046-gemma4-audio-head-divisibility`
- `agent/bug-047-gemma4-audio-odd-hidden-size`
- `agent/bug-048-gemma4-audio-zero-chunk-size`
- `agent/bug-049-gemma4-audio-short-sscp-channels`
- `agent/bug-050-gemma4-audio-short-sscp-kernels`
- `agent/bug-051-gemma4-audio-zero-sscp-stride`
- `agent/bug-052-gemma4-audio-oversized-sscp-kernel`
- `agent/bug-053-gemma4-audio-zero-conv-kernel`

## BUG-054 到 BUG-056：Gemma4 multimodal embedding 与 mask 对齐

受影响文件：

- [candle-transformers/src/models/gemma4/mod.rs](../candle-transformers/src/models/gemma4/mod.rs)

`forward_multimodal` 会先得到 text token embeddings，然后把 image/audio encoder 输出投影到 text hidden size，
再替换输入序列中的特殊 token 位置。关键 helper 是：

```rust
fn broadcast_embed_to_mask(embeds: &Tensor, mask: &Tensor) -> Result<Tensor>
```

它的合同应该是：

```text
按 mask 中 true 的位置，从左到右、从 batch 0 到 batch N，依次放入 embeds 的第 0、1、2... 行。
```

原实现有三类问题：

- 单 batch 时只是把 embeds 从序列开头 pad/truncate 到 `seq_len`，并没有放到 mask 为 true 的位置。
- 多 batch 时直接返回全 0，导致 multimodal features 被静默丢掉。
- `embeds.len()` 和 mask token 数不一致时静默 pad 或 truncate，调用者无法发现 prompt 中特殊 token 数错了。

修复策略：

1. 把 mask flatten 到 host，统计 true token 数。
2. 要求 embedding 行数和 mask true 数完全一致。
3. 构造一个 row-major gather index：mask true 的位置依次拿下一个 embed，mask false 的位置临时拿 0。
4. `index_select` 后 reshape 成 `[B, S, H]`，再乘 mask，把 false 位置清零。

新增测试：

- `broadcast_embed_to_mask_places_single_batch_tokens_at_mask_positions`
- `broadcast_embed_to_mask_places_multi_batch_tokens_in_row_major_order`
- `broadcast_embed_to_mask_rejects_embedding_count_mismatch`

已运行：

```bash
cargo test -p candle-transformers models::gemma4::tests
```

分支：

- `agent/bug-054-gemma4-single-batch-embed-placement`
- `agent/bug-055-gemma4-multi-batch-embed-placement`
- `agent/bug-056-gemma4-embed-mask-count-mismatch`

## BUG-057 到 BUG-059：Gemma4 audio forward 输入合同

受影响文件：

- [candle-transformers/src/models/gemma4/audio.rs](../candle-transformers/src/models/gemma4/audio.rs)

`AudioModel::forward` 的文档合同是：

```text
audio_mel      : [batch, time, mel_bins]
audio_mel_mask : [batch, time]
```

这两个 batch/time 维不能利用 Tensor 广播规则。mask 的语义是一帧一位，如果 mask batch 或 time
为 1 被广播到更大的 audio tensor，就会把一条 mask 错套到多个样本或多个 frame 上。

修复策略是在 public forward 第一行校验：

- `audio_mel` 是 3D。
- `audio_mel_mask` 是 2D。
- time 维非空。
- mask batch 等于 audio batch。
- mask time 等于 audio time。

新增测试：

- `audio_model_rejects_mask_batch_mismatch`
- `audio_model_rejects_mask_time_mismatch`
- `audio_model_rejects_empty_time_dimension`

已运行：

```bash
cargo test -p candle-transformers models::gemma4::audio::tests
```

分支：

- `agent/bug-057-gemma4-audio-mask-batch-broadcast`
- `agent/bug-058-gemma4-audio-mask-time-broadcast`
- `agent/bug-059-gemma4-audio-empty-time-input`

## BUG-060：Gemma4 modality 输入没有对应特殊 token

受影响文件：

- [candle-transformers/src/models/gemma4/mod.rs](../candle-transformers/src/models/gemma4/mod.rs)

BUG-054 到 BUG-056 修复了 `broadcast_embed_to_mask` 的主要放置逻辑，但还剩一个边界：

```text
embeds.len() > 0，但 mask 中没有任何 image/audio special token。
```

这代表调用者给了 image/audio encoder 输出，却没有在 prompt 中留出承载位置。原逻辑在 `token_count == 0`
时直接返回全 0，等价于静默丢掉这些多模态特征。

修复策略：

- 先计算 `embed_len` 和 `token_count`。
- 无论 `token_count` 是否为 0，都要求二者相等。
- 只有 `embed_len == token_count == 0` 时才返回全 0。

新增测试：

- `broadcast_embed_to_mask_rejects_embeddings_without_mask_tokens`

已运行：

```bash
cargo test -p candle-transformers models::gemma4::tests
```

分支：

- `agent/bug-060-gemma4-modality-without-token`

## BUG-061 到 BUG-062：Gemma4 vision pooling 零输出长度

受影响文件：

- [candle-transformers/src/models/gemma4/vision.rs](../candle-transformers/src/models/gemma4/vision.rs)

`VisionTower::encode_single` 根据图像 patch 数和 pooling kernel 算输出 token 数：

```rust
let output_length = num_patches / (k * k);
```

如果输入图像太小，例如只有 1 个 patch，而 `pooling_kernel_size = 3`，则 `output_length = 0`。
后续 pooler 会把 0 当成输出长度参与除法和 scatter 输出 shape，语义已经失效。

修复策略：

- `pooling_kernel_size^2` 使用 `checked_mul`，防溢出。
- pooling kernel area 为 0 时返回错误。
- `output_length == 0` 时在 `VisionTower` 入口返回清晰错误。
- `VisionPooler` 自己也拒绝 `output_length = 0`，避免其他调用绕过。

新增测试：

- `vision_tower_rejects_zero_pooling_output_length`
- `vision_pooler_rejects_zero_output_length`

已运行：

```bash
cargo test -p candle-transformers models::gemma4::vision::tests
```

分支：

- `agent/bug-061-gemma4-vision-empty-pooling-output`
- `agent/bug-062-gemma4-vision-zero-pooler-output-length`

## BUG-063 到 BUG-069：Qwen3-VL vision 构造期配置

受影响文件：

- [candle-transformers/src/models/qwen3_vl/vision.rs](../candle-transformers/src/models/qwen3_vl/vision.rs)

Qwen3-VL vision 有多处从 config 派生 shape 的代码：

```rust
let head_dim = cfg.hidden_size / cfg.num_heads;
VisionRotaryEmbedding::new(head_dim / 2, device)?;
cfg.spatial_merge_size.pow(2)
Conv3dNoBias::new(..., [temporal_patch_size, patch_size, patch_size], stride = patch_size)
```

这些字段如果不在构造期校验，会导致三类问题：

- 直接除零或取模零。
- 整数除法静默截断 head dim。
- RoPE cos/sin 维度和 Q/K head dim 对不上，错误延迟到 forward。

修复策略：

- `hidden_size > 0`
- `num_heads > 0`
- `hidden_size % num_heads == 0`
- `head_dim` 是 4 的非零倍数，因为 vision RoPE 会先构造 `head_dim / 2` 的二维频率，再拼回 Q/K head。
- `patch_size > 0`
- `temporal_patch_size > 0`
- `spatial_merge_size > 0`，并检查平方不溢出。
- `num_position_embeddings > 0`，原有 perfect square 检查继续保留。

新增测试：

- `vision_model_rejects_zero_num_heads`
- `vision_model_rejects_hidden_size_not_divisible_by_heads`
- `vision_model_rejects_invalid_rotary_head_dim`
- `vision_model_rejects_zero_patch_size`
- `vision_model_rejects_zero_temporal_patch_size`
- `vision_model_rejects_zero_spatial_merge_size`
- `vision_model_rejects_zero_position_embeddings`

已运行：

```bash
cargo test -p candle-transformers models::qwen3_vl::vision::tests
```

分支：

- `agent/bug-063-qwen3-vl-vision-zero-heads`
- `agent/bug-064-qwen3-vl-vision-head-divisibility`
- `agent/bug-065-qwen3-vl-vision-invalid-rope-head-dim`
- `agent/bug-066-qwen3-vl-vision-zero-patch-size`
- `agent/bug-067-qwen3-vl-vision-zero-temporal-patch`
- `agent/bug-068-qwen3-vl-vision-zero-spatial-merge`
- `agent/bug-069-qwen3-vl-vision-zero-position-embeddings`

## BUG-070 到 BUG-074：Qwen3-VL text attention 配置和空序列

受影响文件：

- [candle-transformers/src/models/qwen3_vl/text.rs](../candle-transformers/src/models/qwen3_vl/text.rs)

Qwen3-VL text 的 attention 是典型 GQA 结构。构造函数会从 config 派生下面这些值：

```rust
let num_heads = cfg.num_attention_heads;
let num_kv_heads = cfg.num_key_value_heads;
let q_proj = linear_b(hidden_sz, num_heads * cfg.head_dim, false, vb.pp("q_proj"))?;
let k_proj = linear_b(hidden_sz, num_kv_heads * cfg.head_dim, false, vb.pp("k_proj"))?;
let n_kv_groups = cfg.num_attention_heads / cfg.num_key_value_heads;
let softmax_scale = 1.0 / (cfg.head_dim as f64).sqrt();
```

这里的边界不应该等到 forward 里靠 reshape、matmul 或 softmax 间接暴露。原因很直接：

- `num_attention_heads = 0` 会构造 zero-head attention，`num_attn_heads` 还会被外层 mask expand 使用。
- `num_key_value_heads = 0` 会在构造阶段直接除零 panic，绕过 `Result<Self>`。
- `num_attention_heads % num_key_value_heads != 0` 时，`n_kv_groups` 使用整数除法静默截断，Q heads 和 K/V heads 组数合同已经不成立。
- `head_dim = 0` 会构造空 RoPE/RmsNorm，并让 `1.0 / sqrt(0)` 变成非有限 scale 语义。
- `forward_embeds` 输入 `[B, 0, H]` 时，调用方拿到的是底层 reshape 错误，旧代码还存在后续 `seq_len - 1` 这类 last-token 边界风险。

我先只加测试、不加修复，在官方 main 基线上跑：

```bash
cargo test -p candle-transformers models::qwen3_vl::text::tests
```

修复前 5 个测试全部失败，失败形态分别是：

- `num_attention_heads = 0`：构造函数返回 `Ok`，没有拒绝非法 zero-head attention。
- `num_key_value_heads = 0`：在 `num_attention_heads / num_key_value_heads` 处 panic，信息是 `attempt to divide by zero`。
- heads/KV heads 不整除：构造函数返回 `Ok`，GQA group 数被静默截断。
- `head_dim = 0`：构造函数返回 `Ok`，接受了无效 head 维度。
- 空序列：没有得到明确的空序列错误，而是 `cannot reshape tensor of 0 elements to (1, 0, ())`。

修复策略：

1. 新增 `validate_attention_config`，在构造期拒绝：
   - `num_attention_heads == 0`
   - `num_key_value_heads == 0`
   - `num_attention_heads % num_key_value_heads != 0`
   - `head_dim == 0`
2. `Qwen3VLTextModel::new` 和 `Attention::new` 都调用校验。前者保护 public 构造 API，后者保护模块内部未来新增构造路径。
3. `forward_embeds` 在进入 decoder layers 前拒绝 `seq_len == 0`。
4. 顺手把 attention mask 的 device 迁移从 `unwrap()` 改成 `?`，保持错误传播风格一致。

新增测试：

- `text_model_rejects_zero_attention_heads`
- `text_model_rejects_zero_key_value_heads`
- `text_model_rejects_non_divisible_key_value_heads`
- `text_model_rejects_zero_head_dim`
- `text_model_rejects_empty_input_sequence`

修复后验证：

```bash
cargo test -p candle-transformers models::qwen3_vl::text::tests
cargo fmt --all --check
git diff --check
```

分支：

- `agent/bug-070-qwen3-vl-text-zero-attention-heads`
- `agent/bug-071-qwen3-vl-text-zero-kv-heads`
- `agent/bug-072-qwen3-vl-text-kv-head-divisibility`
- `agent/bug-073-qwen3-vl-text-zero-head-dim`
- `agent/bug-074-qwen3-vl-text-empty-sequence`

## BUG-075 到 BUG-080：Qwen3-VL forward glue 输入合同

受影响文件：

- [candle-transformers/src/models/qwen3_vl/mod.rs](../candle-transformers/src/models/qwen3_vl/mod.rs)

这一组问题不在单个 attention 算子里，而在多模态模型的外层 glue。`Qwen3VLModel::forward`
同时接收文本 token、image/video pixel tensor、grid metadata、placeholder span、`seqlens` 和
`seqlen_offsets`。这些参数共同定义“每个 batch 的 token 序列和视觉 embedding 如何对齐”。

原代码里有三个典型边界点：

```rust
seqlen_offsets[0]
seqlens.iter().max().unwrap()
spans.iter().map(|(s, e)| e - s)
```

这些写法本身不一定错，但前提是调用入口已经证明：

- `seqlen_offsets` 非空，且长度等于 batch size。
- `seqlens` 非空，且长度等于 batch size。
- 每个 placeholder span 满足 `start <= end <= seq_len`。

修复前我先只加测试、不加修复，跑：

```bash
cargo test -p candle-transformers models::qwen3_vl::tests
```

6 个测试全部失败，失败形态分别是：

- `seqlen_offsets = []`：`seqlen_offsets[0]` 直接 index out of bounds。
- `seqlen_offsets.len() != batch_size`：模型返回 `Ok`，静默接受了和 batch 不一致的 offsets。
- `seqlens = []`：`seqlens.iter().max().unwrap()` 对 `None` unwrap。
- `seqlens.len() != batch_size`：模型返回 `Ok`，静默接受了和 batch 不一致的 per-batch 元数据。
- image placeholder span `(1, 0)`：计算 `end - start` 时 `usize` 下溢 panic。
- video placeholder span `(1, 0)`：同样在 `end - start` 处下溢 panic。

修复策略：

1. 新增 `validate_sequence_metadata`，在 forward 入口先校验 `seqlens` 和 `seqlen_offsets`。
2. 新增 `validate_placeholder_spans`，在调用 vision encoder 前先校验 image/video spans。
3. `validate_placeholder_spans` 返回已校验的 `total_expected`，替代原来的 `map(|(s, e)| e - s).sum()`。
4. span 总长度使用 `checked_add`，避免极端情况下累计长度溢出。

新增测试：

- `model_rejects_empty_seqlen_offsets`
- `model_rejects_seqlen_offsets_batch_mismatch`
- `model_rejects_empty_seqlens`
- `model_rejects_seqlens_batch_mismatch`
- `model_rejects_reversed_image_placeholder_span`
- `model_rejects_reversed_video_placeholder_span`

测试夹具里有一个容易忽略的细节：Qwen3-VL 的 `conv3d_temporal_2` 明确假设 temporal patch size
为 2，因此最小 vision fixture 也要用 `temporal_patch_size = 2`，pixel row 长度是
`in_chans * temporal_patch_size * patch_size * patch_size = 6`。否则测试会在模型构造阶段失败，
还没进入要验证的 forward 边界。

修复后验证：

```bash
cargo test -p candle-transformers models::qwen3_vl::tests
cargo fmt --all --check
git diff --check
```

分支：

- `agent/bug-075-qwen3-vl-empty-seqlen-offsets`
- `agent/bug-076-qwen3-vl-seqlen-offset-batch-mismatch`
- `agent/bug-077-qwen3-vl-empty-seqlens`
- `agent/bug-078-qwen3-vl-seqlens-batch-mismatch`
- `agent/bug-079-qwen3-vl-reversed-image-span`
- `agent/bug-080-qwen3-vl-reversed-video-span`

## BUG-081 到 BUG-086：Qwen3-VL vision runtime grid 合同

受影响文件：

- [candle-transformers/src/models/qwen3_vl/vision.rs](../candle-transformers/src/models/qwen3_vl/vision.rs)

`Qwen3VLVisionModel::forward` 的核心输入不是只有 `pixel_values`，还有 `grid_thw`。这个 tensor
描述每个视觉样本的 temporal、height、width grid，它会同时影响：

- positional embedding 插值。
- rotary position embedding 坐标生成。
- attention 的 cumulative sequence lengths。
- patch merger 的 reshape 分组。

原实现直接把 `grid_thw` 转成 `Vec<Vec<u32>>`，然后假设每行至少三列：

```rust
let grid = grid_thw.to_vec2::<u32>()?;
let h = g[1] as usize;
let w = g[2] as usize;
let area = (g[1] * g[2]) as usize;
```

这些假设没有在入口证明，就会出现两类问题：短列 panic，长列静默忽略；不满足 merge 的 grid
延迟到 reshape 才报错；超大 `h * w` 在 debug 构建下直接整数溢出 panic。

修复前 6 个测试的失败形态：

- `[1, 2]` 两列 grid：访问 `g[2]` 时 index out of bounds。
- `[1, 2, 2, 99]` 四列 grid：模型返回 `Ok`，第 4 列被完全忽略。
- `h = 3, spatial_merge_size = 2`：最终得到 `shape mismatch in reshape`。
- `w = 3, spatial_merge_size = 2`：同样延迟成 reshape 错误。
- pixel rows 为 3、`grid_thw = [1,2,2]` 派生 token 数为 4：只得到 `shape mismatch in add`。
- `h = u32::MAX, w = u32::MAX`：`g[1] * g[2]` 溢出 panic。

修复策略：

1. 新增 `validate_grid_thw`，要求 `grid_thw` 是精确 `[N, 3]`，既不短也不多。
2. 拒绝空 grid row、`t/h/w = 0`、`spatial_merge_size = 0`。
3. 要求 `h` 和 `w` 都能被 `spatial_merge_size` 整除。
4. 用 checked arithmetic 计算每行 area、每行 token 数和总 token 数。
5. 在 `forward` 中用 `xs.dim(0)` 校验 pixel token 行数必须等于 `grid_thw` 派生 token 总数。
6. `build_cu_seqlens` 自己也调用同一个校验，避免后续维护者绕过 public forward 时重新引入溢出。

新增测试：

- `vision_model_rejects_grid_with_too_few_columns`
- `vision_model_rejects_grid_with_extra_columns`
- `vision_model_rejects_grid_height_not_divisible_by_merge`
- `vision_model_rejects_grid_width_not_divisible_by_merge`
- `vision_model_rejects_pixel_grid_token_count_mismatch`
- `vision_model_rejects_grid_area_overflow`

修复后验证：

```bash
cargo test -p candle-transformers models::qwen3_vl::vision::tests
cargo fmt --all --check
git diff --check
```

分支：

- `agent/bug-081-qwen3-vl-grid-too-few-columns`
- `agent/bug-082-qwen3-vl-grid-extra-columns`
- `agent/bug-083-qwen3-vl-grid-height-merge`
- `agent/bug-084-qwen3-vl-grid-width-merge`
- `agent/bug-085-qwen3-vl-grid-token-count`
- `agent/bug-086-qwen3-vl-grid-area-overflow`

## BUG-087 到 BUG-092：PaddleOCR-VL vision 构造期配置

受影响文件：

- [candle-transformers/src/models/paddleocr_vl/vision.rs](../candle-transformers/src/models/paddleocr_vl/vision.rs)

这一组是从 Qwen3-VL 迁移出来的同类审计：PaddleOCR-VL vision 也是 NaViT 风格视觉 encoder，
同样有 patch embedding、2D RoPE、spatial merge projector。它的构造路径里存在这些派生值：

```rust
let head_dim = cfg.hidden_size / cfg.num_attention_heads;
let base_grid_size = cfg.image_size / cfg.patch_size;
let merged_hidden_size = cfg.hidden_size * cfg.spatial_merge_size.pow(2);
VisionRotaryEmbedding::new(head_dim / 2, device)?;
```

修复前 6 个测试的失败形态：

- `num_attention_heads = 0`：`VisionConfig::head_dim()` 直接除零 panic。
- `hidden_size = 10, num_attention_heads = 4`：构造函数返回 `Ok`，head dim 被静默截断。
- `hidden_size = 6, num_attention_heads = 2`：构造函数返回 `Ok`，但 2D RoPE 的 cos/sin 维度后续无法和 3 维 head 对齐。
- `patch_size = 0`：`image_size / patch_size` 直接除零 panic。
- `image_size = 5, patch_size = 2`：构造函数返回 `Ok`，base position grid 从 2.5 静默截断成 2。
- `spatial_merge_size = 0`：Projector 构造函数返回 `Ok`，但 forward 中 `h / m`、`w / m` 会除零。

修复策略：

1. 新增 `validate_vision_config`，在 `VisionModel::new` 最前面拒绝非法 heads、patch/image size 和 RoPE head dim。
2. 新增 `checked_merged_hidden_size`，用 checked arithmetic 计算 `hidden_size * spatial_merge_size^2`。
3. `Projector::new` 也使用 `checked_merged_hidden_size`，避免未来直接构造 projector 时绕过 `VisionModel::new`。
4. 明确要求 PaddleOCR-VL vision head dim 是 4 的非零倍数；这是 2D RoPE 拼接回 Q/K head dim 的必要 shape 合同。

新增测试：

- `vision_model_rejects_zero_attention_heads`
- `vision_model_rejects_hidden_size_not_divisible_by_heads`
- `vision_model_rejects_invalid_rotary_head_dim`
- `vision_model_rejects_zero_patch_size`
- `vision_model_rejects_image_size_not_divisible_by_patch_size`
- `vision_model_rejects_zero_spatial_merge_size`

修复后验证：

```bash
cargo test -p candle-transformers models::paddleocr_vl::vision::tests
cargo fmt --all --check
git diff --check
```

分支：

- `agent/bug-087-paddleocr-vl-zero-vision-heads`
- `agent/bug-088-paddleocr-vl-vision-head-divisibility`
- `agent/bug-089-paddleocr-vl-vision-rope-head-dim`
- `agent/bug-090-paddleocr-vl-zero-patch-size`
- `agent/bug-091-paddleocr-vl-image-patch-divisibility`
- `agent/bug-092-paddleocr-vl-zero-spatial-merge`

## BUG-093 到 BUG-097：PaddleOCR-VL text 配置和空序列

受影响文件：

- [candle-transformers/src/models/paddleocr_vl/text.rs](../candle-transformers/src/models/paddleocr_vl/text.rs)

PaddleOCR-VL text decoder 是 ERNIE-4.5 风格的 causal decoder。它的 attention 构造会从
`TextConfig` 派生这些值：

```rust
let num_kv_groups = num_heads / num_kv_heads;
let q_proj = linear_b(hidden_sz, num_heads * head_dim, ...)?;
let k_proj = linear_b(hidden_sz, num_kv_heads * head_dim, ...)?;
let softmax_scale = 1.0 / (head_dim as f64).sqrt();
```

这些不是“可选优化参数”，而是 attention shape 合同。只要 heads、KV heads 或 head_dim 为 0，
或者 GQA 分组不能整除，模型就无法定义一条可靠的 forward 路径。

修复前 5 个测试的失败形态：

- `num_attention_heads = 0`：构造函数返回 `Ok`，产生语义无效的 zero-head attention。
- `num_key_value_heads = 0`：`num_heads / num_kv_heads` 直接除零 panic。
- `num_attention_heads = 3, num_key_value_heads = 2`：构造函数返回 `Ok`，GQA group 数从 1.5 静默截断成 1。
- `head_dim = 0`：构造函数返回 `Ok`，后续 attention/softmax scale 没有有效 head 维语义。
- 输入 embedding shape 为 `[B, 0, H]`：forward 进入最后 token 选择路径，得到间接 reshape/索引错误。

修复策略：

1. 新增 `validate_text_config`，在 `TextModel::new` 和 `Attention::new` 都调用，避免直接构造
   attention 时绕过 text model 总入口。
2. 构造期拒绝零 attention heads、零 KV heads、heads/KV heads 不整除、零 head_dim。
3. 同一校验函数也收紧 `head_dim` 的偶数性和 `mrope_section` 总宽度，防止 M-RoPE section
   与 head 维度不一致；本批编号只按已经独立复现的 5 个失败形态计数。
4. `forward_embeds_with_mrope` 在读取最后 token 前先拒绝 `seq_len == 0`，把间接错误变成入口合同错误。

新增测试：

- `text_model_rejects_zero_attention_heads`
- `text_model_rejects_zero_key_value_heads`
- `text_model_rejects_non_divisible_key_value_heads`
- `text_model_rejects_zero_head_dim`
- `text_model_rejects_empty_input_sequence`

修复后验证：

```bash
cargo test -p candle-transformers models::paddleocr_vl::text::tests
cargo fmt --all --check
git diff --check
```

分支：

- `agent/bug-093-paddleocr-vl-text-zero-attention-heads`
- `agent/bug-094-paddleocr-vl-text-zero-kv-heads`
- `agent/bug-095-paddleocr-vl-text-kv-head-divisibility`
- `agent/bug-096-paddleocr-vl-text-zero-head-dim`
- `agent/bug-097-paddleocr-vl-text-empty-sequence`

## BUG-098 到 BUG-107：PaddleOCR-VL vision runtime grid 合同

受影响文件：

- [candle-transformers/src/models/paddleocr_vl/vision.rs](../candle-transformers/src/models/paddleocr_vl/vision.rs)

PaddleOCR-VL vision 的 public forward 不只接收图像 Tensor，还接收 `grid_thw`。这个 Tensor
描述每张图的 temporal、height、width patch grid，并同时驱动三条路径：

- `rot_pos_emb` 用它生成 2D RoPE 的行列坐标。
- `build_cu_seqlens` 用它划分每个 frame 的 attention 序列范围。
- `Projector::forward` 用它按 `spatial_merge_size` 做 2D patch merge。

原实现的问题是：三条路径都直接 `to_vec2::<u32>()?`，然后假设每行正好三列、t/h/w 都大于 0、
h/w 能被 merge size 整除、pixel token 总数一定匹配。这些假设没有在入口证明。

修复前 10 个测试的失败形态：

- `[1, 2]` 两列 grid：`v[1..3]` 切片越界 panic。
- `[1, 2, 2, 99]` 四列 grid：模型返回 `Ok`，第 4 列被完全忽略。
- 空 grid：返回 `stack expects at least one tensor` 这种内部错误，而不是指向 `grid_thw`。
- `t = 0`、`h = 0`、`w = 0`：进入空 stack/reshape 路径，错误和调用方输入合同脱节。
- `h = 3, spatial_merge_size = 2`：projector 只处理前两行 patch，最后一行被静默丢掉。
- `w = 3, spatial_merge_size = 2`：projector 只处理前两列 patch，最后一列被静默丢掉。
- pixel rows 为 4、`grid_thw = [1, 2, 4]` 派生 token 数为 8：只得到间接 narrow 错误。
- `h = 65536, w = 65536`：`build_cu_seqlens` 中 `u32` 乘法在 debug 下溢出 panic。

修复策略：

1. 新增 `validate_grid_thw`，要求 `grid_thw` 精确为 `[N, 3]` 且至少一行。
2. 拒绝 `t/h/w = 0`，拒绝 `spatial_merge_size = 0`。
3. 要求 h/w 都能被 `spatial_merge_size` 整除，避免 projector 静默丢 patch。
4. 使用 `usize` + checked arithmetic 计算每行 area、每行 token 数和总 token 数。
5. `VisionModel` 在 RoPE、cu_seqlens、projector 前先校验 pixel token 行数等于 `grid_thw` 派生 token 总数。
6. `Projector::forward` 和 `forward_multi` 也调用同一个 helper，防止直接构造 projector 时绕过总入口。

新增测试：

- `vision_model_rejects_grid_with_too_few_columns`
- `vision_model_rejects_grid_with_extra_columns`
- `vision_model_rejects_empty_grid`
- `vision_model_rejects_zero_temporal_grid`
- `vision_model_rejects_zero_grid_height`
- `vision_model_rejects_zero_grid_width`
- `vision_model_rejects_grid_height_not_divisible_by_merge`
- `vision_model_rejects_grid_width_not_divisible_by_merge`
- `vision_model_rejects_pixel_grid_token_count_mismatch`
- `build_cu_seqlens_uses_usize_grid_area`

修复后验证：

```bash
cargo test -p candle-transformers models::paddleocr_vl::vision::tests
cargo fmt --all --check
git diff --check
```

分支：

- `agent/bug-098-paddleocr-vl-grid-too-few-columns`
- `agent/bug-099-paddleocr-vl-grid-extra-columns`
- `agent/bug-100-paddleocr-vl-empty-grid`
- `agent/bug-101-paddleocr-vl-zero-temporal-grid`
- `agent/bug-102-paddleocr-vl-zero-grid-height`
- `agent/bug-103-paddleocr-vl-zero-grid-width`
- `agent/bug-104-paddleocr-vl-grid-height-merge`
- `agent/bug-105-paddleocr-vl-grid-width-merge`
- `agent/bug-106-paddleocr-vl-grid-token-count`
- `agent/bug-107-paddleocr-vl-grid-area-overflow`

## BUG-108 到 BUG-116：PaddleOCR-VL text M-RoPE 输入合同

受影响文件：

- [candle-transformers/src/models/paddleocr_vl/text.rs](../candle-transformers/src/models/paddleocr_vl/text.rs)

这一组继续看 PaddleOCR-VL text，但关注点从构造期配置转到运行期 M-RoPE 输入。M-RoPE
有两个容易被低估的协议：

- `RotaryEmbedding::apply_multimodal_rotary_emb` 的 `position_ids` 必须是 `[3, batch, seq_len]`。
- `compute_mrope_position_ids` 里 image token span 必须和 `grid_h * grid_w` 完全一致。

原实现的问题是，第一条协议用 `assert_eq!` 和 Tensor broadcast 间接处理，第二条协议只找第一个
image token，然后继续扫描后续 token，没有证明 token 数量和连续性。

修复前 9 个测试的失败形态：

- `position_ids` 第一维为 2：普通 M-RoPE 路径 `assert_eq!` panic。
- export 版 M-RoPE 第一维为 2：同样 `assert_eq!` panic。
- `position_ids` batch 为 2、q/k batch 为 1：broadcast 把输出 batch 静默扩大。
- `position_ids` seq_len 为 3、q/k seq_len 为 2：只得到间接 `broadcast_mul` shape 错误。
- `grid_h = 0` 或 `grid_w = 0`：单图 helper 返回 `Ok`，image token 被当成普通文本位置推进。
- image token 少于 grid 面积：helper 返回 `Ok`，缺失的视觉 token 没被发现。
- image token 多于 grid 面积：多余 image token 被当成文本 token。
- image token 非连续：两个 image token 被当成同一张图分散编码。

修复策略：

1. 新增 `validate_position_ids`，同时校验 q/k batch、seq_len、head_dim，以及 `position_ids` 精确 shape。
2. 普通 M-RoPE 和 export M-RoPE 共用这个校验，删除 public 方法中的 `assert_eq!`。
3. 新增 `validate_image_grid`，拒绝零 `grid_h/grid_w`，并用 checked arithmetic 计算 grid token 数。
4. `compute_mrope_position_ids` 对每个 batch 收集 image token 位置，要求数量等于 grid 面积且位置连续。
5. 额外补一个正向测试，确认连续 image token 的位置编码仍保持原语义。

新增测试：

- `rotary_embedding_rejects_position_ids_with_wrong_first_dim`
- `rotary_embedding_export_rejects_position_ids_with_wrong_first_dim`
- `rotary_embedding_rejects_position_ids_batch_mismatch`
- `rotary_embedding_rejects_position_ids_seq_len_mismatch`
- `compute_mrope_position_ids_rejects_zero_grid_height`
- `compute_mrope_position_ids_rejects_zero_grid_width`
- `compute_mrope_position_ids_rejects_too_few_image_tokens`
- `compute_mrope_position_ids_rejects_too_many_image_tokens`
- `compute_mrope_position_ids_rejects_non_contiguous_image_tokens`
- `compute_mrope_position_ids_accepts_contiguous_image_tokens`

修复后验证：

```bash
cargo test -p candle-transformers models::paddleocr_vl::text::tests
cargo fmt --all --check
git diff --check
```

分支：

- `agent/bug-108-paddleocr-vl-position-ids-first-dim`
- `agent/bug-109-paddleocr-vl-position-ids-export-first-dim`
- `agent/bug-110-paddleocr-vl-position-ids-batch-mismatch`
- `agent/bug-111-paddleocr-vl-position-ids-seq-mismatch`
- `agent/bug-112-paddleocr-vl-mrope-zero-grid-height`
- `agent/bug-113-paddleocr-vl-mrope-zero-grid-width`
- `agent/bug-114-paddleocr-vl-mrope-too-few-image-tokens`
- `agent/bug-115-paddleocr-vl-mrope-too-many-image-tokens`
- `agent/bug-116-paddleocr-vl-mrope-non-contiguous-image-tokens`

## 一百一十六个案例教什么

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
