//! # Sampling 与 Rust enum
//!
//! 真正实现位于 `candle-transformers/src/generation/mod.rs`。
//! 这个文件负责把多个命令行 Option 组合成一个合法的 Sampling enum。

use anyhow::{bail, Result};
use candle_transformers::generation::Sampling;

pub fn build_sampling(
    temperature: f64,
    top_k: Option<usize>,
    top_p: Option<f64>,
) -> Result<Sampling> {
    if temperature < 0.0 {
        bail!("temperature must be non-negative, got {temperature}")
    }
    if top_k == Some(0) {
        bail!("top-k must be greater than zero")
    }
    if let Some(p) = top_p {
        if !(0.0..=1.0).contains(&p) || p == 0.0 {
            bail!("top-p must be in (0, 1], got {p}")
        }
    }

    // `if` 和 `match` 都是表达式；最后没有分号，所以结果直接作为函数返回值。
    // match 输入是元组，两个 Option 的四种组合被穷尽列出。
    let sampling = if temperature == 0.0 {
        Sampling::ArgMax
    } else {
        match (top_k, top_p) {
            (None, None) => Sampling::All { temperature },
            (Some(k), None) => Sampling::TopK { k, temperature },
            (None, Some(p)) => Sampling::TopP { p, temperature },
            (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
        }
    };
    Ok(sampling)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_temperature_means_argmax() {
        assert_eq!(build_sampling(0.0, None, None).unwrap(), Sampling::ArgMax);
    }

    #[test]
    fn rejects_invalid_top_k() {
        assert!(build_sampling(1.0, Some(0), None).is_err());
    }
}
