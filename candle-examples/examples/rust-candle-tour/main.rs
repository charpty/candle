//! # Candle 源码导览：从应用入口一路看到 Attention
//!
//! 这是一个可运行的教学示例。它没有下载真正的 LLaMA 权重，而是用很小的确定性权重
//! 搭出一条与 LLaMA 相同形状的主干：
//!
//! ```text
//! CLI 参数
//!    │
//!    ├── Rust: struct / derive / Option / Result
//!    ▼
//! token ids: Vec<u32>
//!    │
//!    ├── Rust: 所有权、slice 借用、&T 与 &mut T
//!    ▼
//! Tensor [batch, seq]
//!    │
//!    ├── Embedding
//!    ▼
//! hidden [batch, seq, hidden]
//!    │
//!    ├── Q/K/V projection → causal attention → KV Cache
//!    ▼
//! hidden [batch, seq, hidden]
//!    │
//!    ├── 只选择最后一个位置 → lm_head
//!    ▼
//! logits [batch, vocab]
//!    │
//!    ├── argmax / temperature / top-k / top-p
//!    ▼
//! next token
//!    │
//!    └── 可选 EOS 检查
//! ```
//!
//! Candle 的分层：
//!
//! ```text
//! 本文件 main.rs                         应用与生成循环
//!          │
//!          ▼
//! model.rs                               模型、Attention、KV Cache
//!          │
//!          ▼
//! candle-nn                              Embedding、Linear、Module、VarBuilder
//!          │
//!          ▼
//! candle-core                            Tensor、Shape、Layout、Storage
//!          │
//!          ▼
//! CPU / CUDA / Metal                     具体算子后端
//! ```
//!
//! 建议阅读顺序：
//!
//! 1. 先顺着本文件的 `main` 往下看。
//! 2. 看到 `TinyCausalLm::forward` 时跳到 `model.rs`。
//! 3. 看到 `LogitsProcessor` 时跳到 `sampling.rs`。
//! 4. 最后用 `--show-backend-path` 进入 `backend_walkthrough.rs`。
//! 5. 用 `--show-layout-path` 观察 shape、stride 和 contiguous。
//! 6. 加上 `--trace-shapes` 观察 prefill/decode 中每个关键 Tensor 的 shape 和 dtype。
//! 7. 加上 `--show-weight-paths` 观察 VarBuilder 会查找哪些参数名。

mod backend_walkthrough;
mod layout_walkthrough;
mod model;
mod sampling;

use std::time::{Duration, Instant};

use anyhow::{bail, Result};
use candle::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use clap::Parser;

use model::{expected_demo_weight_shapes, KvCache, TinyCausalLm, TinyConfig};
use sampling::build_sampling;

/// `derive(Parser)` 是过程宏：它根据字段和 `#[arg(...)]` 属性生成命令行解析代码。
///
/// 对 C++ 开发者来说，可以把它理解为“编译期代码生成 + 强类型参数结构”，但它不是
/// C 预处理器的文本替换。
#[derive(Debug, Parser)]
#[command(about = "A fully annotated tiny Candle causal-LM code tour")]
struct Args {
    /// 强制使用 CPU。未指定时，Candle 会尝试已编译且可用的 CUDA/Metal 后端。
    #[arg(long)]
    cpu: bool,

    /// 教学模型没有 tokenizer，所以直接使用逗号分隔的 token id。
    #[arg(long, default_value = "1,5,9,2")]
    prompt: String,

    /// 生成多少个新 token。
    #[arg(short = 'n', long, default_value_t = 8)]
    sample_len: usize,

    /// 小于等于 0 时使用 greedy/argmax。
    #[arg(long, default_value_t = 0.0)]
    temperature: f64,

    #[arg(long)]
    top_k: Option<usize>,

    #[arg(long)]
    top_p: Option<f64>,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// 可选 EOS token。采样到该 token 后停止，并把它保留在输出序列中。
    #[arg(long)]
    eos_token: Option<u32>,

    /// 额外运行一个 2x2 matmul，按源码层次讲解后端分发。
    #[arg(long)]
    show_backend_path: bool,

    /// 额外展示 transpose/narrow/contiguous 如何改变 Tensor layout。
    #[arg(long)]
    show_layout_path: bool,

    /// 打印 demo 模型通过 VarBuilder 查找的权重路径和 shape。
    #[arg(long)]
    show_weight_paths: bool,

    /// 打印每次 forward 内部关键 Tensor 的 shape 和 dtype。
    #[arg(long)]
    trace_shapes: bool,

    /// 同时运行 KV cache 解码和全量重算，比较两者生成结果。
    #[arg(long)]
    compare_cache: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GenerationMode {
    Cached,
    RecomputeAll,
}

impl GenerationMode {
    fn label(self) -> &'static str {
        match self {
            Self::Cached => "cached",
            Self::RecomputeAll => "recompute-all",
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct GenerationSummary {
    tokens: Vec<u32>,
    cache_lens: Vec<usize>,
    stopped_on_eos: Option<u32>,
    elapsed: Duration,
}

/// 把 `"1,5,9"` 转成 `Vec<u32>`。
///
/// Rust 语法观察：
///
/// - 参数用 `&str`，表示函数只借用字符串，不取得所有权。
/// - `Result<Vec<u32>>` 表示成功返回 Vec，失败返回错误。
/// - iterator 的每一项都是 `Result<u32>`，`collect` 可以把它们汇总成一个 Result。
fn parse_prompt(prompt: &str, vocab_size: usize) -> Result<Vec<u32>> {
    if prompt.trim().is_empty() {
        bail!("prompt must contain at least one token id")
    }

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

    if let Some(token) = tokens
        .iter()
        .copied()
        .find(|&token| token as usize >= vocab_size)
    {
        bail!("token {token} is outside vocabulary 0..{vocab_size}")
    }
    Ok(tokens)
}

fn main() -> Result<()> {
    // `Args::parse()` 返回拥有全部命令行数据的 Args。
    let args = Args::parse();

    // `?`：成功时取出 Device，失败时立刻把错误从 main 返回。
    // `device` 默认不可变，因为后面不会把它重新绑定到别的 Device。
    let device = candle_examples::device(args.cpu)?;

    if args.show_backend_path {
        backend_walkthrough::run(&device)?;
    }
    if args.show_layout_path {
        layout_walkthrough::run(&device)?;
    }

    // TinyConfig 实现了 Copy，所以这个按值变量复制成本很低。
    // 真正 LLaMA 的对应配置在 candle-transformers/src/models/llama.rs。
    let config = TinyConfig {
        vocab_size: 32,
        hidden_size: 16,
        num_heads: 4,
    };
    if args.show_weight_paths {
        print_weight_paths(config)?;
    }

    // 这里模拟真实的：
    // SafeTensors -> VarBuilder -> Llama::load。
    // 区别只是本示例用确定性小矩阵代替磁盘权重。
    let model = TinyCausalLm::from_demo_weights(config, &device)?;

    let prompt_tokens = parse_prompt(&args.prompt, config.vocab_size)?;
    validate_optional_token("eos token", args.eos_token, config.vocab_size)?;
    let sampling = build_sampling(args.temperature, args.top_k, args.top_p)?;

    println!("device: {device:?}");
    println!("prompt token ids: {prompt_tokens:?}");

    if args.compare_cache {
        compare_cache_modes(
            &model,
            &device,
            &prompt_tokens,
            args.sample_len,
            args.seed,
            sampling,
            args.eos_token,
            args.trace_shapes,
        )?;
    } else {
        println!("generation:");
        let summary = run_generation(
            &model,
            &device,
            &prompt_tokens,
            args.sample_len,
            args.seed,
            sampling,
            args.eos_token,
            GenerationMode::Cached,
            args.trace_shapes,
            true,
        )?;
        if let Some(eos_token) = summary.stopped_on_eos {
            println!("stopped on eos token: {eos_token}");
        }
        println!("all token ids: {:?}", summary.tokens);
    }
    Ok(())
}

fn validate_optional_token(label: &str, token: Option<u32>, vocab_size: usize) -> Result<()> {
    if let Some(token) = token {
        if token as usize >= vocab_size {
            bail!("{label} {token} is outside vocabulary 0..{vocab_size}")
        }
    }
    Ok(())
}

fn print_weight_paths(config: TinyConfig) -> Result<()> {
    println!("weight path tour:");
    for weight in expected_demo_weight_shapes(config)? {
        let (rows, cols) = weight.dims;
        println!("  {:<44} [{rows}, {cols}]", weight.name);
    }
    println!(
        "  path rule: vb.pp(\"model.layers.0.self_attn\").pp(\"q_proj\") -> model.layers.0.self_attn.q_proj.weight"
    );
    Ok(())
}

fn generation_step_context(
    mode: GenerationMode,
    step: usize,
    token_count: usize,
    index_pos: usize,
) -> (usize, usize, &'static str) {
    match mode {
        GenerationMode::Cached => {
            let context_size = if step == 0 { token_count } else { 1 };
            let stage = if step == 0 { "prefill" } else { "decode" };
            (context_size, index_pos, stage)
        }
        GenerationMode::RecomputeAll => (token_count, 0, "recompute"),
    }
}

fn compare_cache_modes(
    model: &TinyCausalLm,
    device: &Device,
    prompt_tokens: &[u32],
    sample_len: usize,
    seed: u64,
    sampling: Sampling,
    eos_token: Option<u32>,
    trace_shapes: bool,
) -> Result<()> {
    if sampling != Sampling::ArgMax {
        bail!("--compare-cache currently requires greedy sampling, keep --temperature 0.0")
    }

    println!("cache comparison:");
    let cached = run_generation(
        model,
        device,
        prompt_tokens,
        sample_len,
        seed,
        sampling.clone(),
        eos_token,
        GenerationMode::Cached,
        trace_shapes,
        true,
    )?;
    let recomputed = run_generation(
        model,
        device,
        prompt_tokens,
        sample_len,
        seed,
        sampling,
        eos_token,
        GenerationMode::RecomputeAll,
        trace_shapes,
        true,
    )?;
    print_generation_stats("cached", &cached);
    print_generation_stats("recompute-all", &recomputed);

    if cached.tokens != recomputed.tokens {
        bail!(
            "cached and recompute-all generated different tokens: {:?} vs {:?}",
            cached.tokens,
            recomputed.tokens
        )
    }
    if cached.stopped_on_eos != recomputed.stopped_on_eos {
        bail!(
            "cached and recompute-all stopped differently: {:?} vs {:?}",
            cached.stopped_on_eos,
            recomputed.stopped_on_eos
        )
    }

    println!("cache comparison result: token sequences match");
    Ok(())
}

fn print_generation_stats(label: &str, summary: &GenerationSummary) {
    let seconds = summary.elapsed.as_secs_f64();
    let generated_len = summary.cache_lens.len();
    let tokens_per_second = if generated_len == 0 {
        0.0
    } else if seconds > 0.0 {
        generated_len as f64 / seconds
    } else {
        f64::INFINITY
    };
    println!(
        "  {label} generated={generated_len} elapsed={elapsed:?} token/s={tokens_per_second:.2}",
        elapsed = summary.elapsed
    );
}

fn run_generation(
    model: &TinyCausalLm,
    device: &Device,
    prompt_tokens: &[u32],
    sample_len: usize,
    seed: u64,
    sampling: Sampling,
    eos_token: Option<u32>,
    mode: GenerationMode,
    trace_shapes: bool,
    print_steps: bool,
) -> Result<GenerationSummary> {
    // `mut` 必须显式写出：tokens 会 push，cache 会更新，sampler 的 RNG 状态会推进。
    let mut tokens = prompt_tokens.to_vec();
    let mut cache = KvCache::default();
    let mut sampler = LogitsProcessor::from_sampling(seed, sampling);
    let mut cache_lens = Vec::with_capacity(sample_len);
    let mut stopped_on_eos = None;

    // index_pos 是当前输入在完整序列中的起点，RoPE 会使用同一概念。
    let mut index_pos = 0usize;
    let started = Instant::now();

    for step in 0..sample_len {
        // 第一次处理完整 prompt，叫 prefill。
        // 后续只处理最新 token，叫 decode。
        let (context_size, forward_index_pos, stage) =
            generation_step_context(mode, step, tokens.len(), index_pos);

        // `ctxt` 的类型是 `&[u32]`：它借用 Vec 的一段，不复制 token。
        // `saturating_sub` 防止 usize 下溢。
        let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];

        // Tensor::new 把 host slice 转成目标 Device 上的 Tensor。
        // unsqueeze(0) 加 batch 维：[seq] -> [1, seq]。
        let input = Tensor::new(ctxt, device)?.unsqueeze(0)?;
        if trace_shapes {
            println!(
                "  {mode} {stage}: index_pos={forward_index_pos} context_size={context_size} input_shape={input_shape:?}",
                mode = mode.label(),
                input_shape = input.dims(),
            );
        }

        // 权重是只读的，所以 model 用 `&self`。
        // KV Cache 会变化，所以必须传独占可变借用 `&mut cache`。
        let mut recompute_cache;
        let active_cache = match mode {
            GenerationMode::Cached => &mut cache,
            GenerationMode::RecomputeAll => {
                recompute_cache = KvCache::default();
                &mut recompute_cache
            }
        };
        let logits = model.forward(&input, forward_index_pos, active_cache, trace_shapes)?;
        let cache_len = active_cache.len();

        // 模型已处理 ctxt 中的全部位置；下一轮从新的绝对位置开始。
        if mode == GenerationMode::Cached {
            index_pos += ctxt.len();
        }

        // logits shape 是 [1, vocab]。去掉 batch 维得到 [vocab]。
        let logits = logits.squeeze(0)?;
        let next_token = sampler.sample(&logits)?;

        // 先完成 ctxt 的最后一次使用。如果把这条 println 放到 push 后面，编译器会报
        // E0502：push 可能重新分配 Vec，而 ctxt 仍借用旧内存。
        if print_steps {
            println!(
                "  {mode} step={step:02} input={ctxt:?} -> token={next_token:02} cache_len={cache_len}",
                mode = mode.label()
            );
        }
        cache_lens.push(cache_len);

        // 到这里 `ctxt` 已经不再使用，借用结束；因此可以安全 push。
        // push 可能让 Vec 重新分配，如果 ctxt 仍会使用，编译器会拒绝这段代码。
        tokens.push(next_token);
        if eos_token == Some(next_token) {
            stopped_on_eos = Some(next_token);
            break;
        }
    }

    Ok(GenerationSummary {
        tokens,
        cache_lens,
        stopped_on_eos,
        elapsed: started.elapsed(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_prompt_accepts_whitespace() -> Result<()> {
        assert_eq!(parse_prompt(" 1, 5 ,9 ", 32)?, vec![1, 5, 9]);
        Ok(())
    }

    #[test]
    fn parse_prompt_rejects_blank_prompt() {
        let error = parse_prompt("   ", 32).unwrap_err().to_string();
        assert!(error.contains("at least one token"));
    }

    #[test]
    fn parse_prompt_rejects_missing_token_between_commas() {
        let error = parse_prompt("1,,2", 32).unwrap_err().to_string();
        assert!(error.contains("empty token id"));
    }

    #[test]
    fn parse_prompt_rejects_out_of_vocab_token() {
        let error = parse_prompt("1,32", 32).unwrap_err().to_string();
        assert!(error.contains("outside vocabulary"));
    }

    #[test]
    fn accepts_missing_eos_token() -> Result<()> {
        validate_optional_token("eos token", None, 32)
    }

    #[test]
    fn rejects_out_of_vocab_eos_token() {
        let error = validate_optional_token("eos token", Some(32), 32)
            .unwrap_err()
            .to_string();
        assert!(error.contains("outside vocabulary"));
    }

    #[test]
    fn cached_first_step_uses_full_prompt() {
        assert_eq!(
            generation_step_context(GenerationMode::Cached, 0, 4, 0),
            (4, 0, "prefill")
        );
    }

    #[test]
    fn cached_decode_uses_one_token_and_current_position() {
        assert_eq!(
            generation_step_context(GenerationMode::Cached, 2, 6, 5),
            (1, 5, "decode")
        );
    }

    #[test]
    fn recompute_mode_uses_full_context_from_position_zero() {
        assert_eq!(
            generation_step_context(GenerationMode::RecomputeAll, 3, 7, 6),
            (7, 0, "recompute")
        );
    }

    #[test]
    fn zero_length_generation_returns_prompt_without_cache_steps() -> Result<()> {
        let device = Device::Cpu;
        let config = TinyConfig {
            vocab_size: 32,
            hidden_size: 16,
            num_heads: 4,
        };
        let model = TinyCausalLm::from_demo_weights(config, &device)?;
        let prompt = vec![1, 5, 9, 2];

        let summary = run_generation(
            &model,
            &device,
            &prompt,
            0,
            42,
            Sampling::ArgMax,
            None,
            GenerationMode::Cached,
            false,
            false,
        )?;

        assert_eq!(summary.tokens, prompt);
        assert!(summary.cache_lens.is_empty());
        assert_eq!(summary.stopped_on_eos, None);
        Ok(())
    }

    #[test]
    fn eos_stops_generation_after_sampled_token_is_appended() -> Result<()> {
        let device = Device::Cpu;
        let config = TinyConfig {
            vocab_size: 32,
            hidden_size: 16,
            num_heads: 4,
        };
        let model = TinyCausalLm::from_demo_weights(config, &device)?;
        let prompt = vec![1, 5, 9, 2];

        let summary = run_generation(
            &model,
            &device,
            &prompt,
            4,
            42,
            Sampling::ArgMax,
            Some(22),
            GenerationMode::Cached,
            false,
            false,
        )?;

        assert_eq!(summary.tokens, vec![1, 5, 9, 2, 23, 22]);
        assert_eq!(summary.cache_lens, vec![4, 5]);
        assert_eq!(summary.stopped_on_eos, Some(22));
        Ok(())
    }

    #[test]
    fn cached_generation_matches_full_recompute_when_eos_stops_early() -> Result<()> {
        let device = Device::Cpu;
        let config = TinyConfig {
            vocab_size: 32,
            hidden_size: 16,
            num_heads: 4,
        };
        let model = TinyCausalLm::from_demo_weights(config, &device)?;
        let prompt = vec![1, 5, 9, 2];

        let cached = run_generation(
            &model,
            &device,
            &prompt,
            8,
            42,
            Sampling::ArgMax,
            Some(22),
            GenerationMode::Cached,
            false,
            false,
        )?;
        let recomputed = run_generation(
            &model,
            &device,
            &prompt,
            8,
            42,
            Sampling::ArgMax,
            Some(22),
            GenerationMode::RecomputeAll,
            false,
            false,
        )?;

        assert_eq!(cached.tokens, recomputed.tokens);
        assert_eq!(cached.cache_lens, recomputed.cache_lens);
        assert_eq!(cached.stopped_on_eos, Some(22));
        assert_eq!(recomputed.stopped_on_eos, Some(22));
        Ok(())
    }

    #[test]
    fn cached_generation_matches_full_recompute_for_greedy() -> Result<()> {
        let device = Device::Cpu;
        let config = TinyConfig {
            vocab_size: 32,
            hidden_size: 16,
            num_heads: 4,
        };
        let model = TinyCausalLm::from_demo_weights(config, &device)?;
        let prompt = vec![1, 5, 9, 2];

        let cached = run_generation(
            &model,
            &device,
            &prompt,
            4,
            42,
            Sampling::ArgMax,
            None,
            GenerationMode::Cached,
            false,
            false,
        )?;
        let recomputed = run_generation(
            &model,
            &device,
            &prompt,
            4,
            42,
            Sampling::ArgMax,
            None,
            GenerationMode::RecomputeAll,
            false,
            false,
        )?;

        assert_eq!(cached.tokens, recomputed.tokens);
        assert_eq!(cached.cache_lens, vec![4, 5, 6, 7]);
        assert_eq!(recomputed.cache_lens, vec![4, 5, 6, 7]);
        Ok(())
    }
}
