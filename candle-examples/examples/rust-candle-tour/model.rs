//! # Tiny causal language model
//!
//! 本文件不是为了实现一个有语言能力的模型，而是用最少代码保留真实 LLaMA forward 的
//! 关键结构和 Tensor shape。对照阅读：
//!
//! - 真正的 LLaMA 模型：`candle-transformers/src/models/llama.rs`
//! - Linear：`candle-nn/src/linear.rs`
//! - Embedding：`candle-nn/src/embedding.rs`
//! - Tensor：`candle-core/src/tensor.rs`

use std::collections::HashMap;

use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{embedding, linear_no_bias, Embedding, Linear, Module, VarBuilder};

/// 小配置只保留本例需要的三个维度。
///
/// `Copy` 表示按值复制不会转移后让原变量失效；这里三个字段都是 usize，可以安全 Copy。
#[derive(Debug, Clone, Copy)]
pub struct TinyConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
}

impl TinyConfig {
    fn head_dim(self) -> usize {
        self.hidden_size / self.num_heads
    }

    fn validate(self) -> Result<()> {
        if self.num_heads == 0 || self.hidden_size % self.num_heads != 0 {
            candle::bail!(
                "hidden_size {} must be divisible by non-zero num_heads {}",
                self.hidden_size,
                self.num_heads
            )
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpectedWeight {
    pub name: &'static str,
    pub dims: (usize, usize),
}

pub fn expected_demo_weight_shapes(config: TinyConfig) -> Result<Vec<ExpectedWeight>> {
    config.validate()?;
    let h = config.hidden_size;
    let v = config.vocab_size;
    Ok(vec![
        ExpectedWeight {
            name: "model.embed_tokens.weight",
            dims: (v, h),
        },
        ExpectedWeight {
            name: "model.layers.0.self_attn.q_proj.weight",
            dims: (h, h),
        },
        ExpectedWeight {
            name: "model.layers.0.self_attn.k_proj.weight",
            dims: (h, h),
        },
        ExpectedWeight {
            name: "model.layers.0.self_attn.v_proj.weight",
            dims: (h, h),
        },
        ExpectedWeight {
            name: "model.layers.0.self_attn.o_proj.weight",
            dims: (h, h),
        },
        ExpectedWeight {
            name: "lm_head.weight",
            dims: (v, h),
        },
    ])
}

/// 每个请求/会话都应持有自己的 KV Cache。
///
/// `Option` 明确表达两种状态：
///
/// - None：还没有执行 prefill。
/// - Some((k, v))：已经保存历史 K/V。
#[derive(Debug, Default)]
pub struct KvCache {
    kv: Option<(Tensor, Tensor)>,
}

impl KvCache {
    pub fn len(&self) -> usize {
        // `as_ref()` 把 Option<(Tensor, Tensor)> 变成 Option<&(Tensor, Tensor)>，
        // 因此这里只借用 cache，不会移动里面的 Tensor。
        self.kv.as_ref().map_or(0, |(key, _value)| key.dims()[2])
    }
}

#[derive(Debug, Clone)]
struct TinyAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl TinyAttention {
    fn load(vb: VarBuilder, config: TinyConfig) -> Result<Self> {
        let hidden = config.hidden_size;

        // `vb.pp("q_proj")` 返回带新路径前缀的 VarBuilder，不修改原来的 vb。
        // linear_no_bias 最终读取 `{prefix}.weight`。
        Ok(Self {
            q_proj: linear_no_bias(hidden, hidden, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(hidden, hidden, vb.pp("k_proj"))?,
            v_proj: linear_no_bias(hidden, hidden, vb.pp("v_proj"))?,
            o_proj: linear_no_bias(hidden, hidden, vb.pp("o_proj"))?,
            num_heads: config.num_heads,
            head_dim: config.head_dim(),
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        cache: &mut KvCache,
        trace_shapes: bool,
    ) -> Result<Tensor> {
        let (batch, seq_len, hidden) = x.dims3()?;
        trace_tensor(trace_shapes, "attn input [B,S,H]", x);

        // [B, S, H] -> [B, S, H]，然后拆成 head 并转置：
        // [B, S, H] -> [B, S, N, D] -> [B, N, S, D]
        let q = self
            .q_proj
            .forward(x)?
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        trace_tensor(trace_shapes, "q [B,N,S,D]", &q);
        let k_new = self
            .k_proj
            .forward(x)?
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        trace_tensor(trace_shapes, "k new [B,N,S,D]", &k_new);
        let v_new = self
            .v_proj
            .forward(x)?
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        trace_tensor(trace_shapes, "v new [B,N,S,D]", &v_new);

        // `match &cache.kv` 只借用旧 cache。
        // prefill 走 None；decode 用 Tensor::cat 在 sequence 维（dim=2）追加新 K/V。
        let (k, v) = match &cache.kv {
            None => (k_new, v_new),
            Some((k_cached, v_cached)) => (
                Tensor::cat(&[k_cached, &k_new], 2)?.contiguous()?,
                Tensor::cat(&[v_cached, &v_new], 2)?.contiguous()?,
            ),
        };

        // Tensor::clone() 对 Candle Tensor 是共享引用计数句柄，不是复制整块 K/V 数据。
        // cache 获得共享句柄，下面的 attention 仍可继续使用局部 k/v。
        cache.kv = Some((k.clone(), v.clone()));
        trace_tensor(trace_shapes, "k cache [B,N,T,D]", &k);
        trace_tensor(trace_shapes, "v cache [B,N,T,D]", &v);

        // Q [B, N, S, D]
        // Kᵀ [B, N, D, T]
        // att [B, N, S, T]
        let scale = (self.head_dim as f64).sqrt();
        let attention_scores = (q.matmul(&k.t()?)? / scale)?;
        trace_tensor(trace_shapes, "scores [B,N,S,T]", &attention_scores);

        let attention_scores = if seq_len == 1 {
            // decode 时只有当前 query，没有“当前输入内部的未来 token”。
            attention_scores
        } else {
            // mask shape [S, T]，1 表示该位置必须被屏蔽。
            let mask =
                candle_transformers::utils::build_causal_mask(seq_len, index_pos, x.device())?
                    .broadcast_as(attention_scores.shape())?;
            let minus_infinity =
                Tensor::new(f32::NEG_INFINITY, x.device())?.broadcast_as(mask.shape())?;
            mask.where_cond(&minus_infinity, &attention_scores)?
        };

        let probabilities = candle_nn::ops::softmax_last_dim(&attention_scores)?;
        trace_tensor(trace_shapes, "probs [B,N,S,T]", &probabilities);

        // [B, N, S, T] @ [B, N, T, D] -> [B, N, S, D]
        // 再转回 [B, S, N, D] -> [B, S, H]。
        let context = probabilities.matmul(&v)?;
        trace_tensor(trace_shapes, "context heads", &context);
        let context = context.transpose(1, 2)?.reshape((batch, seq_len, hidden))?;
        trace_tensor(trace_shapes, "context [B,S,H]", &context);
        let output = self.o_proj.forward(&context)?;
        trace_tensor(trace_shapes, "attn output", &output);
        Ok(output)
    }
}

/// 真实 LLaMA 还有 RMSNorm、MLP、RoPE 和多层 Block。
/// 本例只保留 Embedding、一个 Attention residual 和 lm_head。
#[derive(Debug, Clone)]
pub struct TinyCausalLm {
    token_embedding: Embedding,
    attention: TinyAttention,
    lm_head: Linear,
}

impl TinyCausalLm {
    fn load(vb: VarBuilder, config: TinyConfig) -> Result<Self> {
        config.validate()?;

        // 最终参数路径：
        // model.embed_tokens.weight
        // model.layers.0.self_attn.{q,k,v,o}_proj.weight
        // lm_head.weight
        Ok(Self {
            token_embedding: embedding(
                config.vocab_size,
                config.hidden_size,
                vb.pp("model.embed_tokens"),
            )?,
            attention: TinyAttention::load(vb.pp("model.layers.0.self_attn"), config)?,
            lm_head: linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?,
        })
    }

    /// 创建确定性小权重，然后仍然通过 VarBuilder 加载。
    ///
    /// 这样既不需要网络/模型文件，又能真实演示“参数名 -> VarBuilder -> Layer”的路径。
    pub fn from_demo_weights(config: TinyConfig, device: &Device) -> Result<Self> {
        config.validate()?;
        let mut weights = HashMap::new();
        for (index, weight) in expected_demo_weight_shapes(config)?.into_iter().enumerate() {
            let (rows, cols) = weight.dims;
            insert_weight(&mut weights, weight.name, rows, cols, index + 1, device)?;
        }

        // from_tensors 取得 HashMap 的所有权；builder 内部用 trait object 统一权重来源。
        let vb = VarBuilder::from_tensors(weights, DType::F32, device);
        Self::load(vb, config)
    }

    pub fn forward(
        &self,
        token_ids: &Tensor,
        index_pos: usize,
        cache: &mut KvCache,
        trace_shapes: bool,
    ) -> Result<Tensor> {
        let (_batch, seq_len) = token_ids.dims2()?;
        trace_tensor(trace_shapes, "token ids [B,S]", token_ids);

        // [B, S] -> [B, S, H]
        let hidden = self.token_embedding.forward(token_ids)?;
        trace_tensor(trace_shapes, "embedding [B,S,H]", &hidden);
        let residual = &hidden;
        let attended = self
            .attention
            .forward(&hidden, index_pos, cache, trace_shapes)?;
        let hidden = (attended + residual)?;
        trace_tensor(trace_shapes, "residual [B,S,H]", &hidden);

        // 生成下一个 token 只需要最后一个位置：
        // [B, S, H] -> [B, H] -> [B, V]
        let last_hidden = hidden.i((.., seq_len - 1, ..))?.contiguous()?;
        trace_tensor(trace_shapes, "last hidden [B,H]", &last_hidden);
        let logits = self.lm_head.forward(&last_hidden)?.to_dtype(DType::F32)?;
        trace_tensor(trace_shapes, "logits [B,V]", &logits);
        Ok(logits)
    }
}

fn trace_tensor(enabled: bool, label: &str, tensor: &Tensor) {
    if enabled {
        println!("{}", tensor_trace_line(label, tensor));
    }
}

fn tensor_trace_line(label: &str, tensor: &Tensor) -> String {
    format!(
        "    {label:<22} shape={:?} dtype={:?}",
        tensor.dims(),
        tensor.dtype()
    )
}

/// 生成一个小而确定的矩阵，避免下载权重，也避免依赖后端随机数实现。
fn insert_weight(
    weights: &mut HashMap<String, Tensor>,
    name: &str,
    rows: usize,
    cols: usize,
    salt: usize,
    device: &Device,
) -> Result<()> {
    let values = (0..rows * cols)
        .map(|index| {
            let centered = ((index * 7 + salt * 11) % 29) as f32 - 14.0;
            centered / 32.0
        })
        .collect::<Vec<_>>();
    let tensor = Tensor::from_vec(values, (rows, cols), device)?;
    weights.insert(name.to_string(), tensor);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_head_config() {
        let config = TinyConfig {
            vocab_size: 8,
            hidden_size: 10,
            num_heads: 4,
        };
        let error = config.validate().unwrap_err().to_string();
        assert!(error.contains("must be divisible"));
    }

    #[test]
    fn reports_expected_weight_names_and_shapes() -> Result<()> {
        let config = TinyConfig {
            vocab_size: 32,
            hidden_size: 16,
            num_heads: 4,
        };
        let weights = expected_demo_weight_shapes(config)?;
        assert_eq!(
            weights
                .iter()
                .map(|weight| (weight.name, weight.dims))
                .collect::<Vec<_>>(),
            vec![
                ("model.embed_tokens.weight", (32, 16)),
                ("model.layers.0.self_attn.q_proj.weight", (16, 16)),
                ("model.layers.0.self_attn.k_proj.weight", (16, 16)),
                ("model.layers.0.self_attn.v_proj.weight", (16, 16)),
                ("model.layers.0.self_attn.o_proj.weight", (16, 16)),
                ("lm_head.weight", (32, 16)),
            ]
        );
        Ok(())
    }

    #[test]
    fn tensor_trace_line_reports_shape_and_dtype() -> Result<()> {
        let tensor = Tensor::new(&[1u32, 2], &Device::Cpu)?;
        let line = tensor_trace_line("token ids [S]", &tensor);
        assert!(line.contains("shape=[2]"));
        assert!(line.contains(&format!("dtype={:?}", tensor.dtype())));
        Ok(())
    }

    #[test]
    fn forward_extends_cache_from_prefill_to_decode() -> Result<()> {
        let device = Device::Cpu;
        let config = TinyConfig {
            vocab_size: 32,
            hidden_size: 16,
            num_heads: 4,
        };
        let model = TinyCausalLm::from_demo_weights(config, &device)?;
        let mut cache = KvCache::default();

        let prompt = Tensor::new(&[1u32, 5, 9], &device)?.unsqueeze(0)?;
        let prefill_logits = model.forward(&prompt, 0, &mut cache, false)?;
        assert_eq!(prefill_logits.dims(), &[1, config.vocab_size]);
        assert_eq!(cache.len(), 3);

        let next = Tensor::new(&[2u32], &device)?.unsqueeze(0)?;
        let decode_logits = model.forward(&next, 3, &mut cache, false)?;
        assert_eq!(decode_logits.dims(), &[1, config.vocab_size]);
        assert_eq!(cache.len(), 4);

        Ok(())
    }
}
