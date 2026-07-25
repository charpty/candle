# Rust + Candle 可执行源码导览

从 [`main.rs`](./main.rs) 开始阅读，然后依次进入：

1. [`model.rs`](./model.rs)：Embedding、Attention、KV Cache、lm_head。
2. [`sampling.rs`](./sampling.rs)：Option、enum、match 和采样参数。
3. [`backend_walkthrough.rs`](./backend_walkthrough.rs)：Tensor 到硬件后端的路径。

运行：

```bash
cd candle
cargo run -p candle-examples --example rust-candle-tour -- --cpu
```

同时观察 matmul 后端路径：

```bash
cargo run -p candle-examples \
  --example rust-candle-tour \
  -- \
  --cpu \
  --show-backend-path
```

测试：

```bash
cargo test -p candle-examples --example rust-candle-tour
```

如果本机 Cargo 报错并显示失效的 `https://rsproxy.cn/crates.io-index` Git 地址，可以临时
切换到已经配置的 sparse 索引：

```bash
cargo --config 'source.crates-io.replace-with="rsproxy-sparse"' \
  run -p candle-examples \
  --example rust-candle-tour \
  -- \
  --cpu \
  --show-backend-path
```
