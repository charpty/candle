#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::{DType, Device, Result, Tensor};

#[test]
fn kv_cache() -> Result<()> {
    let mut cache = candle_nn::kv_cache::Cache::new(0, 16);
    for _ in [0, 1] {
        assert_eq!(cache.current_seq_len(), 0);
        let data = cache.current_data()?;
        assert!(data.is_none());
        let t = Tensor::new(&[1f32, 2., 3.], &Device::Cpu)?;
        cache.append(&t)?;
        let data = cache.current_data()?.unwrap();
        assert_eq!(data.to_vec1::<f32>()?, [1., 2., 3.]);
        let t = Tensor::new(&[4f32], &Device::Cpu)?;
        cache.append(&t)?;
        let data = cache.current_data()?.unwrap();
        assert_eq!(data.to_vec1::<f32>()?, [1., 2., 3., 4.]);
        let t = Tensor::new(&[0f32, 5., 6., 7.], &Device::Cpu)?;
        cache.append(&t)?;
        let data = cache.current_data()?.unwrap();
        assert_eq!(data.to_vec1::<f32>()?, [1., 2., 3., 4., 0., 5., 6., 7.]);
        assert_eq!(cache.current_seq_len(), 8);
        cache.reset();
    }
    Ok(())
}

#[test]
fn cache_rejects_zero_max_seq_len_cpu() -> Result<()> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let result = (|| -> Result<(String, usize, bool)> {
            let mut cache = candle_nn::kv_cache::Cache::new(0, 0);
            let t = Tensor::new(&[1f32], &Device::Cpu)?;
            let err = cache.append(&t).unwrap_err().to_string();
            Ok((
                err,
                cache.current_seq_len(),
                cache.current_data()?.is_none(),
            ))
        })();
        tx.send(result).ok();
    });

    let (err, current_seq_len, is_empty) = rx
        .recv_timeout(std::time::Duration::from_secs(1))
        .expect("zero-capacity cache append should return instead of hanging")?;
    assert!(err.contains("max_seq_len"));
    assert_eq!(current_seq_len, 0);
    assert!(is_empty);

    Ok(())
}

#[test]
fn kv_cache_append_is_atomic_when_value_append_fails_cpu() -> Result<()> {
    let mut cache = candle_nn::kv_cache::KvCache::new(1, 4);
    let k = Tensor::zeros((1, 1), DType::F32, &Device::Cpu)?;
    let invalid_v = Tensor::zeros(1, DType::F32, &Device::Cpu)?;

    let err = cache.append(&k, &invalid_v).unwrap_err().to_string();
    assert!(err.contains("dimension"));
    assert_eq!(cache.current_seq_len(), 0);
    assert!(cache.k()?.is_none());
    assert!(cache.v()?.is_none());

    Ok(())
}

#[test]
fn rotating_kv_cache() -> Result<()> {
    let mut cache = candle_nn::kv_cache::RotatingCache::new(0, 6);
    for _ in [0, 1] {
        assert_eq!(cache.offset(), 0);
        assert_eq!(cache.current_seq_len(), 0);
        let data = cache.current_data()?;
        assert!(data.is_none());
        assert_eq!(cache.positions(1), &[0]);
        assert_eq!(cache.positions(2), &[0, 1]);
        let t = Tensor::new(&[1., 2., 3.], &Device::Cpu)?;
        let data = cache.append(&t)?;
        assert_eq!(data.to_vec1::<f64>()?, [1., 2., 3.]);
        assert_eq!(cache.positions(0), &[0, 1, 2]);
        assert_eq!(cache.positions(1), &[0, 1, 2, 3]);
        assert_eq!(cache.positions(2), &[0, 1, 2, 3, 4]);
        assert_eq!(cache.positions(3), &[0, 1, 2, 3, 4, 5]);
        assert_eq!(cache.positions(4), &[6, 1, 2, 3, 4, 5]);
        let t = Tensor::new(&[4.], &Device::Cpu)?;
        let data = cache.append(&t)?;
        assert_eq!(data.to_vec1::<f64>()?, [1., 2., 3., 4.]);
        let t = Tensor::new(&[0., 5., 6., 7.], &Device::Cpu)?;
        let data = cache.append(&t)?;
        assert_eq!(data.to_vec1::<f64>()?, [6., 7., 3., 4., 0., 5.]);
        assert_eq!(cache.current_seq_len(), 8);
        assert_eq!(cache.offset(), 2);

        let t = Tensor::new(&[8.], &Device::Cpu)?;
        let data = cache.append(&t)?;
        assert_eq!(data.to_vec1::<f64>()?, [6., 7., 8., 4., 0., 5.]);
        assert_eq!(cache.current_seq_len(), 9);
        assert_eq!(cache.offset(), 3);

        let t = Tensor::new(&[9., 10., 11.], &Device::Cpu)?;
        let data = cache.append(&t)?;
        assert_eq!(data.to_vec1::<f64>()?, [6., 7., 8., 9., 10., 11.]);
        assert_eq!(cache.current_seq_len(), 12);
        assert_eq!(cache.offset(), 0);

        let t = Tensor::new(&[12.], &Device::Cpu)?;
        let data = cache.append(&t)?;
        assert_eq!(data.to_vec1::<f64>()?, [12., 7., 8., 9., 10., 11.]);
        assert_eq!(cache.current_seq_len(), 13);
        assert_eq!(cache.offset(), 1);

        let mask = cache.attn_mask(2, &Device::Cpu)?.unwrap();
        assert_eq!(
            mask.to_vec2::<u8>()?,
            &[[0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
        );
        let mask = cache.attn_mask(3, &Device::Cpu)?.unwrap();
        assert_eq!(
            mask.to_vec2::<u8>()?,
            &[[0, 0, 1, 1, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0]],
        );
        assert_eq!(cache.positions(0), &[12, 7, 8, 9, 10, 11]);
        assert_eq!(cache.positions(2), &[12, 13, 14, 9, 10, 11]);
        assert_eq!(cache.positions(3), &[12, 13, 14, 15, 10, 11]);
        assert_eq!(cache.positions(8), &[13, 14, 15, 16, 17, 18, 19, 20]);
        let t = Tensor::new(&[0., 1., 2., 3., 4., 5., 6., 7., 8.], &Device::Cpu)?;
        let data = cache.append(&t)?;
        assert_eq!(data.to_vec1::<f64>()?, [0., 1., 2., 3., 4., 5., 6., 7., 8.]);
        assert_eq!(cache.current_seq_len(), 22);
        assert_eq!(cache.offset(), 0);
        assert_eq!(cache.positions(0), &[16, 17, 18, 19, 20, 21]);
        assert_eq!(cache.positions(1), &[22, 17, 18, 19, 20, 21]);

        let mask = cache.attn_mask(1, &Device::Cpu)?;
        assert!(mask.is_none());
        let mask = cache.attn_mask(2, &Device::Cpu)?.unwrap();
        assert_eq!(
            mask.to_vec2::<u8>()?,
            &[[0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
        );
        let mask = cache.attn_mask(3, &Device::Cpu)?.unwrap();
        assert_eq!(
            mask.to_vec2::<u8>()?,
            &[[0, 1, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
        );
        let t = Tensor::new(&[42.], &Device::Cpu)?;

        let data = cache.append(&t)?;
        assert_eq!(data.to_vec1::<f64>()?, [42., 4., 5., 6., 7., 8.]);
        assert_eq!(cache.current_seq_len(), 23);
        assert_eq!(cache.offset(), 1);

        cache.reset();
    }
    Ok(())
}

#[test]
fn rotating_cache_zero_max_seq_len_has_fallible_paths_cpu() -> Result<()> {
    let mut cache = candle_nn::kv_cache::RotatingCache::new(0, 0);

    assert!(cache.positions(0).is_empty());
    assert_eq!(cache.positions(2), &[0, 1]);
    assert!(cache.attn_mask(1, &Device::Cpu)?.is_none());
    let err = cache.attn_mask(2, &Device::Cpu).unwrap_err().to_string();
    assert!(err.contains("max_seq_len"));

    let t = Tensor::new(&[1f32], &Device::Cpu)?;
    let err = cache.append(&t).unwrap_err().to_string();
    assert!(err.contains("max_seq_len"));
    assert_eq!(cache.current_seq_len(), 0);
    assert_eq!(cache.offset(), 0);
    assert!(cache.current_data()?.is_none());

    Ok(())
}

#[test]
fn rotating_kv_cache_append_is_atomic_when_value_append_fails_cpu() -> Result<()> {
    let mut cache = candle_nn::kv_cache::RotatingKvCache::new(1, 4);
    let k = Tensor::zeros((1, 1), DType::F32, &Device::Cpu)?;
    let invalid_v = Tensor::zeros(1, DType::F32, &Device::Cpu)?;

    let err = cache.append(&k, &invalid_v).unwrap_err().to_string();
    assert!(err.contains("dimension"));
    assert_eq!(cache.current_seq_len(), 0);
    assert_eq!(cache.offset(), 0);
    assert!(cache.k()?.is_none());
    assert!(cache.v()?.is_none());

    Ok(())
}

#[test]
fn concat_kv_cache_append_is_atomic_when_value_cat_fails_cpu() -> Result<()> {
    let mut cache = candle_nn::kv_cache::ConcatKvCache::new(1);
    let k1 = Tensor::zeros((1, 1, 2), DType::F32, &Device::Cpu)?;
    let v1 = Tensor::zeros((1, 1, 2), DType::F32, &Device::Cpu)?;
    cache.append(&k1, &v1)?;

    let k2 = Tensor::zeros((1, 1, 2), DType::F32, &Device::Cpu)?;
    let invalid_v2 = Tensor::zeros((2, 1, 2), DType::F32, &Device::Cpu)?;
    let err = cache.append(&k2, &invalid_v2).unwrap_err().to_string();
    assert!(err.contains("shape mismatch") || err.contains("shape"));

    assert_eq!(cache.current_seq_len(), 1);
    assert_eq!(cache.k().unwrap().dims(), &[1, 1, 2]);
    assert_eq!(cache.v().unwrap().dims(), &[1, 1, 2]);

    Ok(())
}

#[test]
fn scattered_cache_rejects_batch_mask_len_mismatch_cpu() -> Result<()> {
    let mut cache =
        candle_nn::kv_cache::ScatteredCacheBuilder::new(2, 5, DType::F32, &Device::Cpu)?;

    let err = cache.indices_and_mask(1, &[true]).unwrap_err().to_string();
    assert!(err.contains("batch_mask length mismatch"));

    let err = cache
        .indices_and_mask(1, &[true, false, true])
        .unwrap_err()
        .to_string();
    assert!(err.contains("batch_mask length mismatch"));

    let err = cache.indices_and_mask(5, &[true]).unwrap_err().to_string();
    assert!(err.contains("batch_mask length mismatch"));

    Ok(())
}
