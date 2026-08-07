//! # Tensor layout 与 stride 导览
//!
//! Tensor 的 shape 只说明“有多少元素”；layout 还说明这些元素如何映射到底层 Storage。
//! 这个文件用一个很小的 Tensor 展示：
//!
//! - `transpose` 通常只改 stride，不复制 Storage。
//! - `narrow` 会改变 start offset，可能让 view 不再是 contiguous。
//! - `contiguous()` 会在需要时复制成新的 row-major 布局。

use candle::{Device, Result, Tensor};

#[derive(Debug, PartialEq, Eq)]
struct LayoutSnapshot {
    label: &'static str,
    dims: Vec<usize>,
    stride: Vec<usize>,
    start_offset: usize,
    contiguous: bool,
}

pub fn run(device: &Device) -> Result<()> {
    println!("layout tour:");
    for snapshot in demo_layout(device)? {
        println!(
            "  {label:<20} dims={dims:?} stride={stride:?} offset={offset} contiguous={contiguous}",
            label = snapshot.label,
            dims = snapshot.dims,
            stride = snapshot.stride,
            offset = snapshot.start_offset,
            contiguous = snapshot.contiguous
        );
    }
    Ok(())
}

fn demo_layout(device: &Device) -> Result<Vec<LayoutSnapshot>> {
    let base = Tensor::arange(0f32, 24f32, device)?.reshape((2, 3, 4))?;
    let transposed = base.transpose(1, 2)?;
    let narrowed = base.narrow(1, 1, 2)?;
    let packed = transposed.contiguous()?;

    Ok(vec![
        snapshot("base", &base),
        snapshot("transpose(1,2)", &transposed),
        snapshot("narrow(dim=1)", &narrowed),
        snapshot("contiguous()", &packed),
    ])
}

fn snapshot(label: &'static str, tensor: &Tensor) -> LayoutSnapshot {
    LayoutSnapshot {
        label,
        dims: tensor.dims().to_vec(),
        stride: tensor.stride().to_vec(),
        start_offset: tensor.layout().start_offset(),
        contiguous: tensor.is_contiguous(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_demo_exposes_views_and_contiguous_copy() -> Result<()> {
        let snapshots = demo_layout(&Device::Cpu)?;
        assert_eq!(
            snapshots,
            vec![
                LayoutSnapshot {
                    label: "base",
                    dims: vec![2, 3, 4],
                    stride: vec![12, 4, 1],
                    start_offset: 0,
                    contiguous: true,
                },
                LayoutSnapshot {
                    label: "transpose(1,2)",
                    dims: vec![2, 4, 3],
                    stride: vec![12, 1, 4],
                    start_offset: 0,
                    contiguous: false,
                },
                LayoutSnapshot {
                    label: "narrow(dim=1)",
                    dims: vec![2, 2, 4],
                    stride: vec![12, 4, 1],
                    start_offset: 4,
                    contiguous: false,
                },
                LayoutSnapshot {
                    label: "contiguous()",
                    dims: vec![2, 4, 3],
                    stride: vec![12, 3, 1],
                    start_offset: 0,
                    contiguous: true,
                },
            ]
        );
        Ok(())
    }
}
