# Patch Policy + TTT 按钮密码任务

> 本文件是项目顶层唯一的实验说明和操作入口。更新时间：2026-08-18。

## 项目概况

本项目把 Patch Policy 的 VQ-BeT 策略应用到 MuJoCo/robosuite 的两按钮密码任务，并加入 RoboTTT 的在线快权重记忆。

- 按钮符号：`1` = 左按钮，`2` = 右按钮。
- 密码长度：最多 6 位；输入是 agentview 和 robot0_eye_in_hand 两路图像。
- 动作是 7-D OSC_POSE，act_scale=1.0。
- 主路线：6 个逐位置密码 token、cls@224 encoder、per-layer TTT。
- 训练和推理始终输入完整密码；禁止用“剩余密码”作为推理条件，避免 GT 进度泄漏。
- progress 头可以把梯度回传到 TTT；count 和 next_key 仅使用 detach() 后的观察探针。
- 不使用 TurboVLA 作为策略，也不下载或使用 PushT 数据集。

## 当前状态

B0 前置环境与数据检查、B1 无 TTT 基线、B2/B3 序列和 TTT 代码均已完成。B1 流水线有 4 项回归测试，TTT 流水线有 15 项回归测试。

最近一次完整密码优化实验是 runs/b10_full48_cls_opt：

- cls@224，per-layer TTT online，T=16 carry windows，tbptt=16；
- progress supervision weight=1.0，训练学习率=1e-4；
- 日志最近一次完整评估：holdout 13/16 = 0.8125，抽样 train 2/2 = 1.0；
- GIF 在 runs/b10_full48_cls_opt/gifs/，训练日志在对应 run 的 log.jsonl。

这些是当前代码和参数的实验记录，不代表所有密码都已稳定成功。后续结果只需追加到本节和对应 runs 目录，不再新建计划文档。

## 目录和数据

```text
patch_policy_ttt/
├── button_task/                 # 按钮环境、数据集、密码 token、TTT 层
├── vendor/patch_policy/         # Patch Policy 工作副本
├── vendor/robo_ttt/             # RoboTTT 工作副本
├── scripts/                     # 检查、测试、评估和诊断脚本
├── runs/                        # checkpoint、日志、标签和 GIF
├── train_button.py              # B1：无 TTT 的 VQ-BeT 训练
├── train_button_ttt.py          # B2/B3：序列 + TTT 训练
├── button_task/password_split.json
├── requirements_button.txt
└── requirements_extra.txt
```

默认数据仍在工作区外：

```text
E:\WM\turbovla\data\button_demos\random_pw6_lang_100\demos.h5
E:\WM\turbovla\data\button_demos\random_pw6_lang_1000\demos.h5
E:\WM\turbovla\data\button_demos\random_pw6_lang_small\demos.h5
```

random_pw6_lang_1000 包含 1000 条 demo、64 个密码；当前划分为 48 个 train 密码和 16 个 holdout 密码。划分保存在 button_task/password_split.json。

## 环境要求

推荐使用按钮任务已有的 Python 环境：

```powershell
$py = E:\WM\turbovla\.venv\Scripts\python.exe
$env:MUJOCO_GL = glfw
```

训练和仿真需要 PyTorch、DINOv2 权重、h5py、robosuite、mujoco、einops，以及 TTT 所需的 einx、torch-einops-utils、rotary-embedding-torch 等依赖。当前工作区已经包含 vendor/patch_policy 和 vendor/robo_ttt。

scripts/check_env.py 是历史环境检查脚本，不应作为当前项目的验收入口：它检查工作区外的 patch_policy/robo_ttt，还把 PushT 数据列为必需项，与当前按钮任务约束不一致。

## 常用命令

### 数据检查和密码划分

```powershell
& $py scripts\check_button_data.py `
  --h5 E:/WM/turbovla/data/button_demos/random_pw6_lang_1000/demos.h5

& $py scripts\make_password_split.py `
  --h5 E:/WM/turbovla/data/button_demos/random_pw6_lang_1000/demos.h5 `
  --out button_task/password_split.json `
  --holdout_count 16
```

### B1：无 TTT 基线

```powershell
& $py train_button.py `
  --h5 E:/WM/turbovla/data/button_demos/random_pw6_lang_1000/demos.h5 `
  --split button_task/password_split.json `
  --out runs/b1_cls `
  --encoder-mode cls --image-size 224 `
  --epochs 60 --batch-size 8 --lr 0.0001
```

### B2/B3：TTT 训练

先生成严格回放标签，再按需要合并 best-effort 标签；--labels 只用于进度监督和课程训练。

```powershell
& $py scripts\label_button_demos.py `
  --h5 E:/WM/turbovla/data/button_demos/random_pw6_lang_1000/demos.h5 `
  --out runs/full_labels_strict.npz

& $py train_button_ttt.py `
  --h5 E:/WM/turbovla/data/button_demos/random_pw6_lang_1000/demos.h5 `
  --split button_task/password_split.json `
  --out runs/ttt_cls `
  --load runs/b1_cls/best.pt `
  --encoder-mode cls --image-size 224 `
  --t-window 16 --carry-windows `
  --ttt online --ttt-base-lr 0.01 --tbptt 16 `
  --prog-weight 1.0 --labels runs/full_labels_strict.npz `
  --epochs 60 --batch-size 8 --lr 0.0001
```

### 回归测试

```powershell
& $py scripts\test_button_train_pipeline.py
& $py scripts\test_button_ttt_pipeline.py
```

前者覆盖密码 token、GPT 条件 mask、VQ-BeT 前向/反向和数据切片；后者覆盖 TTT 的 chunked/stepwise 等价性、TBPTT、per-sample 更新、frozen/online、damp、valid mask、per-step attention 和 per-layer TTT。

### checkpoint 评估、GIF 和逐帧诊断

```powershell
& $py scripts\eval_button_checkpoint.py `
  --ckpt runs/ttt_cls/best.pt --args runs/ttt_cls/args.json `
  --passwords 112212,122221,212112 --repeats 3 --ttt

& $py scripts\visualize_button_rollouts.py `
  --ckpt runs/ttt_cls/best.pt --args runs/ttt_cls/args.json `
  --out runs/ttt_cls/gifs --total 8 --fps 20

& $py scripts\diagnose_rollout.py `
  --ckpt runs/ttt_cls/best.pt --args runs/ttt_cls/args.json `
  --password 122221 --max-steps 120
```

评估时不要用与训练配置不同的 --action-window 覆盖模型结构；eval_button_checkpoint.py 已对此做保护。

## `scripts/` 脚本说明

### 应长期保留的工具

| 脚本 | 作用 |
|---|---|
| `check_button_data.py` | 读取 HDF5，检查 group、字段形状、dtype、episode 长度和密码分布。只读诊断。 |
| `make_password_split.py` | 按密码生成可复现的 train/holdout 划分 JSON；不是随机按 episode 切分。 |
| `label_button_demos.py` | 用相同 seed 回放 demo，生成逐帧 `count`/`next_key` 标签；回放结果与 HDF5 最终 `press_count` 不一致时跳过。 |
| `eval_button_checkpoint.py` | 在 ButtonEnv 中批量 rollout checkpoint，报告 success、press_count 和步数；支持 token-swap 诊断和 TTT 快权重。 |
| `visualize_button_rollouts.py` | 生成带密码、step、press_count 标注的 agentview GIF，用于人工检查策略行为。 |
| `summarize_log.py` | 把 `log.jsonl` 压缩成 epoch、loss、成功率和 TTT 门控表格。 |
| `test_button_train_pipeline.py` | B1 回归测试，共 4 项；应在模型或数据管路改动后运行。 |
| `test_button_ttt_pipeline.py` | B2/B3 回归测试，共 15 项；验证训练和推理的 TTT 更新协议一致。 |

### 仅在特定诊断或复现实验中使用

| 脚本 | 作用和状态 |
|---|---|
| `probe_button_features.py` | 冻结 checkpoint，提取 detached GPT/TTT 特征，再单独训练 `AuxProbeHeads` 检查 `next_key`、`count`、`progress` 是否可从记忆中读出；探针 loss 不回传策略。属于分析实验，不是训练入口。 |
| `diagnose_rollout.py` | 单密码逐帧打印动作、末端位置和环境按钮内部状态，用于定位“没有按键”或动作时序错误。属于故障定位工具。 |
| `bench_encoder.py` | 在固定 CUDA batch 大小下比较 DINOv2 fp32/bf16 吞吐；只用于性能 profiling，且要求 CUDA。 |
| `label_missing_best_effort.py` | 针对少数回放在最后一次按键发生物理漂移的 episode，允许 `replay press_count = h5 press_count - 1` 并修正最后一帧。属于历史数据修复脚本，不能替代严格标注。 |
| `check_env.py` | 旧版环境检查，仍包含 PushT 和工作区外路径检查。当前已过时，不能用它判断按钮任务是否可运行。 |

`scripts/__pycache__/` 是 Python 运行时生成的缓存，不属于源代码；本次只整理文档，没有把实验产物或脚本结果删除。

## 关键实现约束

- 密码条件使用 `seq` 模式的 6 个固定长度 token；`sum` 和 `lookup` 只作为 ablation。
- 训练和推理都传完整密码，不能动态替换成剩余后缀。
- 每个 episode 的 TTT fast weights 从空状态开始，推理按 chunk 携带；训练中的 TBPTT 只 detach 计算图，不改变数值更新规则。
- `next_key`/`count` 探针使用 detached 特征；`progress` 监督只允许训练 TTT 和 progress head，不允许泄漏到 GPT/策略主干。
- `ButtonSliceDataset` 的 action window 不得跨 episode；padding 不应更新快权重，也不应进入 outer loss。
- 训练和评估必须使用相同的 encoder、图像尺寸、action window、TTT 参数和精度设置。
- 运行环境设置 `MUJOCO_GL=glfw`；numba 缓存由训练入口重定向到 `runs/.numba_cache`。

## 当前可复用产物

- `button_task/password_split.json`：48 train / 16 holdout 密码划分。
- `runs/full48_labels.npz`：当前完整 train 密码的进度标签，包含严格回放标签和历史 best-effort 修复标签。
- `runs/b10_full48_cls_opt/`：当前最新优化实验的 checkpoint、日志和 GIF。
- `ara/`：研究记录和证据归档，不是顶层操作入口；其中的历史实验说明不再作为当前计划依据。
