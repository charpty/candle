#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::{DType, Device, Result};
use candle_nn::{
    conv1d, conv1d_no_bias, conv2d, conv2d_no_bias, conv_transpose1d, conv_transpose1d_no_bias,
    Conv1dConfig, Conv2dConfig, ConvTranspose1dConfig, VarBuilder,
};

#[test]
fn conv1d_rejects_zero_groups_cpu() -> Result<()> {
    let device = Device::Cpu;
    let vb = VarBuilder::zeros(DType::F32, &device);
    let cfg = Conv1dConfig {
        groups: 0,
        ..Default::default()
    };

    let err = conv1d(2, 4, 3, cfg, vb.clone()).unwrap_err().to_string();
    assert!(err.contains("groups"));
    let err = conv1d_no_bias(2, 4, 3, cfg, vb).unwrap_err().to_string();
    assert!(err.contains("groups"));

    Ok(())
}

#[test]
fn conv2d_rejects_zero_groups_cpu() -> Result<()> {
    let device = Device::Cpu;
    let vb = VarBuilder::zeros(DType::F32, &device);
    let cfg = Conv2dConfig {
        groups: 0,
        ..Default::default()
    };

    let err = conv2d(2, 4, 3, cfg, vb.clone()).unwrap_err().to_string();
    assert!(err.contains("groups"));
    let err = conv2d_no_bias(2, 4, 3, cfg, vb).unwrap_err().to_string();
    assert!(err.contains("groups"));

    Ok(())
}

#[test]
fn conv_transpose1d_rejects_zero_groups_cpu() -> Result<()> {
    let device = Device::Cpu;
    let vb = VarBuilder::zeros(DType::F32, &device);
    let cfg = ConvTranspose1dConfig {
        groups: 0,
        ..Default::default()
    };

    let err = conv_transpose1d(2, 4, 3, cfg, vb.clone())
        .unwrap_err()
        .to_string();
    assert!(err.contains("groups"));
    let err = conv_transpose1d_no_bias(2, 4, 3, cfg, vb)
        .unwrap_err()
        .to_string();
    assert!(err.contains("groups"));

    Ok(())
}
