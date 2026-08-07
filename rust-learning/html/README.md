# Rust + Candle 推理链路讲解站

入口：

- [index.html](./index.html)
- [architecture.html](./architecture.html)
- [course-syllabus.html](./course-syllabus.html)
- [trace-walkthrough.html](./trace-walkthrough.html)

这是一个纯静态 HTML 学习站，不依赖网络、不需要构建。可以直接用浏览器打开：

```bash
xdg-open rust-learning/html/index.html
```

在不能打开 GUI 的环境里，可以用本地静态服务器预览：

```bash
python3 -m http.server 8000
```

然后访问：

```text
http://127.0.0.1:8000/rust-learning/html/
```

内容结构：

- 第一屏：CLI 到 backend dispatch 的完整推理链路图。
- 项目架构全景图：workspace、crate 边界、Tensor 合同、后端分发、定位表和 bug 案例。
- 课程大纲：模块目标、学习节奏、交付物和验收证据。
- 案例课：把 `--trace-shapes` 和 EOS 输出逐行拆成系统不变量。
- 阶段检查器：每个阶段的 Rust 语义、Candle 合同、常见误解和验证命令。
- 状态分层：模型态、请求态、临时 Tensor 态。
- shape 实验台：切换 cached/recompute，观察 `S`、`T`、cache_len、scores shape。
- 源码地图：直接跳到 tour 和真实 Candle 源码。
- 售卖级验收：用 checklist 检查学习包是否真正能教会人。

对应验证命令：

```bash
cargo test -p candle-examples --example rust-candle-tour
cargo test -p candle-nn --test ops replication_pad2d_cpu
cargo run -p candle-examples --example rust-candle-tour -- --cpu --trace-shapes -n 1
cargo run -p candle-examples --example rust-candle-tour -- --cpu --compare-cache --eos-token 22 -n 8
```
