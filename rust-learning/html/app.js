const stages = [
  {
    id: "cli",
    title: "CLI 参数",
    kind: "request",
    tag: "入口",
    summary: "把用户输入收束成强类型 Args，先决定 device、prompt、采样参数和可选 EOS。",
    metrics: [
      ["Rust 类型", "struct Args + derive(Parser)"],
      ["请求状态", "prompt / sample_len / seed"],
      ["错误边界", "参数解析与 vocab 校验"],
      ["源码", "main.rs"],
    ],
    rust: "`Option<T>` 表达参数是否出现，`Result<()>` 让初始化错误沿 `?` 返回。",
    candle: "此阶段还不创建模型 Tensor，但会决定后续 Tensor 的 device 和请求级状态。",
    pitfall: "不要把 Cargo feature 参数写到 `--` 后面；那会变成 example CLI 参数。",
    commandNote: "运行基础 tour，确认 Args 到生成循环可以闭环。",
    command: "cargo run -p candle-examples --example rust-candle-tour -- --cpu -n 2",
  },
  {
    id: "tokens",
    title: "Prompt tokens",
    kind: "request",
    tag: "输入",
    summary: "把逗号分隔字符串解析成 `Vec<u32>`，并在进入 Tensor 前检查空值和词表范围。",
    metrics: [
      ["输入", "\"1,5,9,2\""],
      ["输出", "Vec<u32>"],
      ["失败", "空 prompt / 空 token / 越界"],
      ["源码", "parse_prompt"],
    ],
    rust: "`collect::<Result<Vec<_>>>()?` 遇到第一处解析错误会短路，不需要 `unwrap()`。",
    candle: "Embedding 只接受合法 token id；越界输入应在构造模型输入前失败。",
    pitfall: "空 prompt 不是小问题，它会让 `[B,S]` 中的 `S` 变成 0，后面取最后位置会失去语义。",
    commandNote: "故意传入越界 token，看错误是否在早期暴露。",
    command: "cargo run -p candle-examples --example rust-candle-tour -- --cpu --prompt 1,32",
  },
  {
    id: "tensor",
    title: "Tensor 输入",
    kind: "tensor",
    tag: "shape",
    summary: "`Tensor::new(ctxt, device)?.unsqueeze(0)?` 把 host slice 变成 `[B,S]` Tensor。",
    metrics: [
      ["shape", "[1, S]"],
      ["dtype", "U32"],
      ["device", "Cpu / Cuda / Metal"],
      ["借用", "ctxt: &[u32]"],
    ],
    rust: "`ctxt` 是借用 `tokens` 的 slice；在 `tokens.push` 前必须结束最后一次使用。",
    candle: "Tensor 同时携带 shape、dtype、device、layout。`--trace-shapes` 会把 shape/dtype 打出来。",
    pitfall: "少了 `unsqueeze(0)`，后续 `dims2()` 和 Embedding 的 batch 约定会对不上。",
    commandNote: "观察 token ids 的 shape 和 dtype。",
    command: "cargo run -p candle-examples --example rust-candle-tour -- --cpu --trace-shapes -n 1",
  },
  {
    id: "weights",
    title: "VarBuilder 权重路径",
    kind: "model",
    tag: "ABI",
    summary: "模型代码和 checkpoint 通过字符串路径对齐，路径和 shape 就是轻量 ABI。",
    metrics: [
      ["Embedding", "model.embed_tokens.weight"],
      ["Attention", "q/k/v/o_proj.weight"],
      ["lm_head", "lm_head.weight"],
      ["来源", "HashMap demo weights"],
    ],
    rust: "`vb.pp(\"...\")` 返回带前缀的新 builder，不修改原 builder。",
    candle: "Layer 只按路径取权重，不关心底层来自 SafeTensors、mmap、HashMap 还是 VarMap。",
    pitfall: "权重名写错是运行期加载错误，不是编译期错误。",
    commandNote: "打印 demo 模型期望的全部权重路径。",
    command: "cargo run -p candle-examples --example rust-candle-tour -- --cpu --show-weight-paths -n 0",
  },
  {
    id: "attention",
    title: "Attention forward",
    kind: "tensor",
    tag: "核心",
    summary: "Embedding 输出 `[B,S,H]`，Attention 拆成 Q/K/V heads，再计算 scores、softmax 和 context。",
    metrics: [
      ["hidden", "[B,S,H]"],
      ["Q/K/V", "[B,N,S,D]"],
      ["scores", "[B,N,S,T]"],
      ["logits", "[B,V]"],
    ],
    rust: "`&self` 读取权重，`&mut KvCache` 独占更新请求态，二者边界必须清楚。",
    candle: "shape 不变量是 `H = N * D`，scores 最后一维 `T` 来自 cache 总长度。",
    pitfall: "本例 shape 数字可能相同，但 `[B,S,N,D]` 和 `[B,N,S,D]` 的维度语义完全不同。",
    commandNote: "打印完整 prefill 过程的 shape/dtype。",
    command: "cargo run -p candle-examples --example rust-candle-tour -- --cpu --trace-shapes -n 1",
  },
  {
    id: "cache",
    title: "KV Cache",
    kind: "request",
    tag: "状态",
    summary: "prefill 后缓存历史 K/V；decode 每步只输入最新 token，并沿序列维追加 K/V。",
    metrics: [
      ["None", "还未 prefill"],
      ["Some", "已有 K/V"],
      ["拼接维度", "dim=2"],
      ["每请求", "独立持有"],
    ],
    rust: "`Option<(Tensor, Tensor)>` 显式表达 cache 是否存在，`as_ref()` 可以只借用不移动。",
    candle: "`Tensor::clone()` 通常复制句柄，`Tensor::cat` 在教学里直观，生产里通常要预分配或分页。",
    pitfall: "cache 不能跨请求共享；那会把一个用户的上下文混进另一个请求。",
    commandNote: "比较 cached decode 和每步全量重算。",
    command: "cargo run -p candle-examples --example rust-candle-tour -- --cpu --compare-cache -n 4",
  },
  {
    id: "sampling",
    title: "Sampling",
    kind: "request",
    tag: "输出",
    summary: "把 `[V]` logits 交给 `LogitsProcessor`，按 greedy、temperature、top-k/top-p 选 next token。",
    metrics: [
      ["greedy", "temperature = 0"],
      ["随机性", "seed + sampler 状态"],
      ["top-k", "候选数量"],
      ["top-p", "累计概率阈值"],
    ],
    rust: "`match (top_k, top_p)` 穷尽四种组合，参数状态不靠默认魔法值表达。",
    candle: "采样前 logits 转成 F32，避免低精度 logits 直接进入概率计算。",
    pitfall: "随机采样下 cache/recompute 对比不一定稳定，所以 `--compare-cache` 要求 greedy。",
    commandNote: "运行一个 top-k/top-p 组合，观察参数能否合法构造。",
    command: "cargo run -p candle-examples --example rust-candle-tour -- --cpu --temperature 0.7 --top-k 5 --top-p 0.9 -n 2",
  },
  {
    id: "eos",
    title: "EOS 停止",
    kind: "request",
    tag: "停止",
    summary: "可选 `--eos-token` 在采样后检查；命中时把 EOS token 保留在输出序列中再停止。",
    metrics: [
      ["状态", "Option<u32>"],
      ["校验", "token < vocab_size"],
      ["停止点", "tokens.push 后"],
      ["测试", "早停 + compare 一致"],
    ],
    rust: "`Option<u32>` 区分未设置和设置了某个 token；不要用 0 假装 None。",
    candle: "EOS 不改变模型 forward，只改变生成循环的请求状态推进。",
    pitfall: "停止检查放在 push 前还是 push 后，会直接改变输出是否包含 EOS。",
    commandNote: "默认 greedy 第二步生成 22，设置 EOS 后会提前停止。",
    command: "cargo run -p candle-examples --example rust-candle-tour -- --cpu --eos-token 22 -n 8",
  },
  {
    id: "backend",
    title: "Backend dispatch",
    kind: "backend",
    tag: "执行",
    summary: "`Tensor::matmul` 检查 shape/dtype/device，再通过 Storage 分发到 CPU/CUDA/Metal。",
    metrics: [
      ["入口", "Tensor::matmul"],
      ["分发", "Storage::matmul"],
      ["后端", "CPU / CUDA / Metal"],
      ["执行", "eager"],
    ],
    rust: "运行时 enum dispatch 和 trait 边界共同存在，错误仍通过 `Result` 回到调用者。",
    candle: "eager execution 意味着算子调用时已经提交或执行，GPU 精确计时要考虑同步。",
    pitfall: "不要一开始就钻 kernel；先确认 Tensor 层的合同。",
    commandNote: "运行 2x2 matmul 后端路径导览。",
    command: "cargo run -p candle-examples --example rust-candle-tour -- --cpu --show-backend-path -n 0",
  },
];

const checklist = [
  ["能从 CLI 讲到 Tensor 输入", "解释 Args、parse_prompt、Tensor::new 和 unsqueeze。"],
  ["能讲清项目架构", "说明 examples、transformers、nn、core 和后端之间的依赖方向。"],
  ["能手算 Attention shape", "写出 hidden、Q/K/V、scores、context、logits 的形状。"],
  ["能区分三类状态", "模型态、请求态、临时 Tensor 态不混放。"],
  ["能说明 KV Cache 正确性", "cached decode 和 recompute-all 在 greedy 下应一致。"],
  ["能定位权重路径错误", "从 VarBuilder 路径推导 checkpoint key。"],
  ["能讲清 dtype/layout", "知道 token ids、hidden、logits 的 dtype 和 contiguous 成本。"],
  ["能解释 EOS 语义", "停止检查位置和输出是否保留 EOS。"],
  ["能复盘真实 bug", "解释 ops、conv groups、loader unwrap、KV cache 原子性、loss/BatchNorm、Mimi 和 Gemma4 配置边界。"],
  ["能跑完整验证", "测试、trace、layout、backend、cache compare 都能跑。"],
];

const generated = [23, 22, 14, 13];
const prompt = [1, 5, 9, 2];
const dims = {
  batch: 1,
  heads: 4,
  headDim: 4,
  hidden: 16,
  vocab: 32,
};

let currentStage = stages[0].id;
const checksStorageKey = "candle-learning-checks";

function el(id) {
  return document.getElementById(id);
}

function stageById(id) {
  return stages.find((stage) => stage.id === id) ?? stages[0];
}

function setStage(id) {
  currentStage = id;
  const stage = stageById(id);

  el("stage-kind").textContent = stage.tag;
  el("stage-title").textContent = stage.title;
  el("stage-summary").textContent = stage.summary;
  el("stage-rust").textContent = stage.rust;
  el("stage-candle").textContent = stage.candle;
  el("stage-pitfall").textContent = stage.pitfall;
  el("stage-command-note").textContent = stage.commandNote;
  el("stage-command").textContent = stage.command;

  el("stage-metrics").replaceChildren(
    ...stage.metrics.map(([label, value]) => {
      const item = document.createElement("div");
      item.className = "metric";
      item.innerHTML = `<span>${label}</span><strong>${value}</strong>`;
      return item;
    }),
  );

  document.querySelectorAll("[data-stage]").forEach((node) => {
    node.setAttribute("aria-current", node.dataset.stage === id ? "true" : "false");
  });
}

function renderStages() {
  const list = el("stage-list");
  const flow = el("flow-grid");

  list.replaceChildren(
    ...stages.map((stage, index) => {
      const button = document.createElement("button");
      button.className = "stage-button";
      button.type = "button";
      button.dataset.stage = stage.id;
      button.innerHTML = `
        <span class="stage-index">${index + 1}</span>
        <span><strong>${stage.title}</strong><span>${stage.summary}</span></span>
      `;
      button.addEventListener("click", () => setStage(stage.id));
      return button;
    }),
  );

  flow.replaceChildren(
    ...stages.map((stage, index) => {
      const button = document.createElement("button");
      button.className = "flow-node";
      button.type = "button";
      button.dataset.stage = stage.id;
      button.innerHTML = `
        <div class="node-meta">
          <span>${String(index + 1).padStart(2, "0")}</span>
          <span class="badge ${stage.kind}">${stage.kind}</span>
        </div>
        <div>
          <h3>${stage.title}</h3>
          <p>${stage.summary}</p>
        </div>
      `;
      button.addEventListener("click", () => setStage(stage.id));
      return button;
    }),
  );
}

function contextFor(mode, step) {
  const tokensBefore = prompt.concat(generated.slice(0, step));
  if (mode === "cached") {
    const contextTokens = step === 0 ? prompt : [generated[step - 1]];
    const indexPos = step === 0 ? 0 : prompt.length + step - 1;
    return {
      stage: step === 0 ? "prefill" : "decode",
      contextTokens,
      contextSize: contextTokens.length,
      indexPos,
      cacheLen: prompt.length + step,
      tokensBefore,
    };
  }

  return {
    stage: "recompute",
    contextTokens: tokensBefore,
    contextSize: tokensBefore.length,
    indexPos: 0,
    cacheLen: tokensBefore.length,
    tokensBefore,
  };
}

function renderDefinitionList(target, rows) {
  target.replaceChildren(
    ...rows.flatMap(([key, value]) => {
      const dt = document.createElement("dt");
      const dd = document.createElement("dd");
      dt.textContent = key;
      dd.textContent = value;
      return [dt, dd];
    }),
  );
}

function updateLab() {
  const mode = el("mode-select").value;
  const step = Number(el("step-range").value);
  const context = contextFor(mode, step);
  const s = context.contextSize;
  const t = context.cacheLen;

  el("step-output").textContent = `step ${step}`;
  renderDefinitionList(el("context-table"), [
    ["stage", context.stage],
    ["tokens before", `[${context.tokensBefore.join(", ")}]`],
    ["input tokens", `[${context.contextTokens.join(", ")}]`],
    ["context_size", String(context.contextSize)],
    ["index_pos", String(context.indexPos)],
    ["cache_len", String(context.cacheLen)],
  ]);

  renderDefinitionList(el("shape-table"), [
    ["input", `[${dims.batch}, ${s}]`],
    ["hidden", `[${dims.batch}, ${s}, ${dims.hidden}]`],
    ["Q", `[${dims.batch}, ${dims.heads}, ${s}, ${dims.headDim}]`],
    ["K/V cache", `[${dims.batch}, ${dims.heads}, ${t}, ${dims.headDim}]`],
    ["scores", `[${dims.batch}, ${dims.heads}, ${s}, ${t}]`],
    ["logits", `[${dims.batch}, ${dims.vocab}]`],
  ]);

  const next = generated[step];
  if (next === 22) {
    el("eos-note").textContent = "本 step 采样到 22；如果设置 --eos-token 22，push 后立即停止。";
  } else {
    el("eos-note").textContent = `本 step 采样到 ${next}；未命中 --eos-token 22，继续下一轮。`;
  }
}

function renderChecklist() {
  const container = el("checklist");
  const saved = readSavedChecks();

  container.replaceChildren(
    ...checklist.map(([title, detail], index) => {
      const label = document.createElement("label");
      label.className = "check-item";
      label.innerHTML = `
        <input type="checkbox" ${saved.has(String(index)) ? "checked" : ""} />
        <span><strong>${title}</strong><span>${detail}</span></span>
      `;
      label.querySelector("input").addEventListener("change", () => {
        const next = readSavedChecks();
        if (label.querySelector("input").checked) {
          next.add(String(index));
        } else {
          next.delete(String(index));
        }
        writeSavedChecks(next);
        updateScore();
      });
      return label;
    }),
  );
  updateScore();
}

function readSavedChecks() {
  try {
    return new Set(JSON.parse(localStorage.getItem(checksStorageKey) || "[]"));
  } catch {
    return new Set();
  }
}

function writeSavedChecks(next) {
  try {
    localStorage.setItem(checksStorageKey, JSON.stringify([...next]));
  } catch {
    // The page remains usable when opened from stricter file:// environments.
  }
}

function updateScore() {
  const checked = document.querySelectorAll(".check-item input:checked").length;
  const percent = Math.round((checked / checklist.length) * 100);
  el("score-ring").textContent = `${percent}%`;
}

function setupCopyButtons() {
  document.querySelectorAll("[data-copy-target]").forEach((button) => {
    button.addEventListener("click", async () => {
      const target = el(button.dataset.copyTarget);
      const text = target?.textContent ?? "";
      try {
        await navigator.clipboard.writeText(text);
        button.textContent = "已复制";
      } catch {
        const area = document.createElement("textarea");
        area.value = text;
        document.body.appendChild(area);
        area.select();
        document.execCommand("copy");
        area.remove();
        button.textContent = "已复制";
      }
      setTimeout(() => {
        button.textContent = "复制";
      }, 1200);
    });
  });
}

function init() {
  renderStages();
  setStage(currentStage);
  updateLab();
  renderChecklist();
  setupCopyButtons();

  el("mode-select").addEventListener("change", updateLab);
  el("step-range").addEventListener("input", updateLab);
}

init();
