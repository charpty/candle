//! # 一个 matmul 如何到达硬件
//!
//! 调用下面的 `a.matmul(&b)` 时，沿源码依次经过：
//!
//! ```text
//! 1. candle-core/src/tensor.rs
//!    Tensor::matmul
//!    - 检查 rank、batch、m/k/n
//!    - 取得两个 Storage 的读访问
//!
//! 2. candle-core/src/storage.rs
//!    Storage::matmul
//!    - 检查 Device 和 DType
//!    - match Cpu / Cuda / Metal
//!
//! 3. 具体 backend
//!    cpu_backend::matmul / cuda_backend::matmul / metal_backend::matmul
//!    - 根据 dtype、layout 选择实现或 kernel
//!
//! 4. 回到 Tensor::matmul
//!    - 用结果 Storage + contiguous Layout 创建新 Tensor
//!    - 记录 BackpropOp::Matmul
//! ```
//!
//! Candle 是 eager execution：`matmul` 返回时计算已经被提交/执行，而不是只登记一张未来
//! 才执行的静态图。GPU 本身可能异步，所以精确计时时仍可能需要 synchronize。

use candle::{Device, Result, Tensor};

pub fn run(device: &Device) -> Result<()> {
    let a = Tensor::new(&[[1f32, 2.0], [3.0, 4.0]], device)?;
    let b = Tensor::new(&[[5f32, 6.0], [7.0, 8.0]], device)?;

    // &b 是共享借用。matmul 不消费 a/b，调用后两者仍然可用。
    let c = a.matmul(&b)?;

    // to_vec2 会把结果变成 host Vec<Vec<f32>>；GPU 上这意味着 device-to-host 读取。
    println!(
        "backend tour: {:?} x {:?} = {:?}",
        a.dims(),
        b.dims(),
        c.to_vec2::<f32>()?
    );
    Ok(())
}
