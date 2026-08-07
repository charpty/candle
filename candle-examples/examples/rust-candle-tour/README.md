# Rust + Candle 可执行源码导览

从 [`main.rs`](./main.rs) 开始阅读，然后依次进入：

1. [`model.rs`](./model.rs)：Embedding、Attention、KV Cache、lm_head。
2. [`sampling.rs`](./sampling.rs)：Option、enum、match 和采样参数。
3. [`backend_walkthrough.rs`](./backend_walkthrough.rs)：Tensor 到硬件后端的路径。
4. [`layout_walkthrough.rs`](./layout_walkthrough.rs)：shape、stride、view 与 contiguous。

配套讲义：

- [Rust + Candle 推理链路讲解站](../../../rust-learning/html/index.html)
- [Candle 项目架构全景图 HTML](../../../rust-learning/html/architecture.html)
- [课程大纲与交付标准 HTML](../../../rust-learning/html/course-syllabus.html)
- [一次生成全链路拆解 HTML 案例课](../../../rust-learning/html/trace-walkthrough.html)
- [Candle 项目级架构讲义](../../../rust-learning/candle-project-architecture.md)
- [Candle bug hunt 报告](../../../rust-learning/bug-hunt-report.md)
- [从 C++ 到 Rust：跟着 Candle 学推理开发](../../../rust-learning/README.md)
- [Rust + Candle 典型用法精讲](../../../rust-learning/typical-patterns.md)
- [Rust + Candle 深度专题：从类型系统到推理系统不变量](../../../rust-learning/deep-inference-systems.md)
- [Rust + Candle 动手实验手册](../../../rust-learning/rust-candle-labs.md)
- [Candle 源码阅读路线图](../../../rust-learning/source-reading-playbook.md)
- [Rust + Candle 学习自检表](../../../rust-learning/checkpoints.md)
- [Rust + Candle 水平测评题](../../../rust-learning/rust-candle-exam.md)
- [Rust + Candle 水平测评题评分参考](../../../rust-learning/rust-candle-exam-grading-guide.md)

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

观察 Tensor layout/stride：

```bash
cargo run -p candle-examples \
  --example rust-candle-tour \
  -- \
  --cpu \
  --show-layout-path \
  -n 0
```

观察 VarBuilder 期望的权重路径：

```bash
cargo run -p candle-examples \
  --example rust-candle-tour \
  -- \
  --cpu \
  --show-weight-paths \
  -n 0
```

观察 prefill/decode 中每个关键 Tensor 的 shape 和 dtype：

```bash
cargo run -p candle-examples \
  --example rust-candle-tour \
  -- \
  --cpu \
  --trace-shapes \
  -n 2
```

比较 KV cache 解码和每步全量重算：

```bash
cargo run -p candle-examples \
  --example rust-candle-tour \
  -- \
  --cpu \
  --compare-cache \
  -n 4
```

观察 EOS 停止逻辑：

```bash
cargo run -p candle-examples \
  --example rust-candle-tour \
  -- \
  --cpu \
  --eos-token 22 \
  -n 8
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
