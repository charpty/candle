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

mod backend_walkthrough;
mod model;
mod sampling;

use anyhow::{bail, Result};
use candle::Tensor;
use candle_transformers::generation::LogitsProcessor;
use clap::Parser;

use model::{KvCache, TinyCausalLm, TinyConfig};
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

    /// 额外运行一个 2x2 matmul，按源码层次讲解后端分发。
    #[arg(long)]
    show_backend_path: bool,
}

/// 把 `"1,5,9"` 转成 `Vec<u32>`。
///
/// Rust 语法观察：
///
/// - 参数用 `&str`，表示函数只借用字符串，不取得所有权。
/// - `Result<Vec<u32>>` 表示成功返回 Vec，失败返回错误。
/// - iterator 的每一项都是 `Result<u32>`，`collect` 可以把它们汇总成一个 Result。
fn parse_prompt(prompt: &str, vocab_size: usize) -> Result<Vec<u32>> {
    let tokens = prompt
        .split(',')
        .map(str::trim)
        .map(|piece| {
            piece
                .parse::<u32>()
                .map_err(|error| anyhow::anyhow!("invalid token id {piece:?}: {error}"))
        })
        .collect::<Result<Vec<_>>>()?;

    if tokens.is_empty() {
        bail!("prompt must contain at least one token id")
    }
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

    // TinyConfig 实现了 Copy，所以这个按值变量复制成本很低。
    // 真正 LLaMA 的对应配置在 candle-transformers/src/models/llama.rs。
    let config = TinyConfig {
        vocab_size: 32,
        hidden_size: 16,
        num_heads: 4,
    };

    // 这里模拟真实的：
    // SafeTensors -> VarBuilder -> Llama::load。
    // 区别只是本示例用确定性小矩阵代替磁盘权重。
    let model = TinyCausalLm::from_demo_weights(config, &device)?;

    // `mut` 必须显式写出：tokens 会 push，cache 会更新，sampler 的 RNG 状态会推进。
    let mut tokens = parse_prompt(&args.prompt, config.vocab_size)?;
    let mut cache = KvCache::default();
    let sampling = build_sampling(args.temperature, args.top_k, args.top_p)?;
    let mut sampler = LogitsProcessor::from_sampling(args.seed, sampling);

    println!("device: {device:?}");
    println!("prompt token ids: {tokens:?}");
    println!("generation:");

    // index_pos 是当前输入在完整序列中的起点，RoPE 会使用同一概念。
    let mut index_pos = 0usize;

    for step in 0..args.sample_len {
        // 第一次处理完整 prompt，叫 prefill。
        // 后续只处理最新 token，叫 decode。
        let context_size = if step == 0 { tokens.len() } else { 1 };

        // `ctxt` 的类型是 `&[u32]`：它借用 Vec 的一段，不复制 token。
        // `saturating_sub` 防止 usize 下溢。
        let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];

        // Tensor::new 把 host slice 转成目标 Device 上的 Tensor。
        // unsqueeze(0) 加 batch 维：[seq] -> [1, seq]。
        let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;

        // 权重是只读的，所以 model 用 `&self`。
        // KV Cache 会变化，所以必须传独占可变借用 `&mut cache`。
        let logits = model.forward(&input, index_pos, &mut cache)?;

        // 模型已处理 ctxt 中的全部位置；下一轮从新的绝对位置开始。
        index_pos += ctxt.len();

        // logits shape 是 [1, vocab]。去掉 batch 维得到 [vocab]。
        let logits = logits.squeeze(0)?;
        let next_token = sampler.sample(&logits)?;

        // 先完成 ctxt 的最后一次使用。如果把这条 println 放到 push 后面，编译器会报
        // E0502：push 可能重新分配 Vec，而 ctxt 仍借用旧内存。
        println!(
            "  step={step:02} input={ctxt:?} -> token={next_token:02} cache_len={}",
            cache.len()
        );

        // 到这里 `ctxt` 已经不再使用，借用结束；因此可以安全 push。
        // push 可能让 Vec 重新分配，如果 ctxt 仍会使用，编译器会拒绝这段代码。
        tokens.push(next_token);
    }

    println!("all token ids: {tokens:?}");
    Ok(())
}
