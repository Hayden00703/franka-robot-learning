# Franka Panda 机械臂强化学习

这是一个精简的 Franka Emika Panda 机械臂学习仓库，记录了在 NVIDIA Isaac Lab 中训练机械臂完成立方体抓取与抬升任务的核心配置、最终权重和评估结果。仓库不包含 Isaac Lab、Isaac Sim、Python 虚拟环境或缓存。

## 当前结果

- 任务：`Isaac-Lift-Cube-Franka-v0`
- 算法：RSL-RL PPO
- 并行环境：4096
- 训练迭代：1500
- 评估：100 个随机 episode
- 成功标准：物体至少抬升 5 cm，并持续 1 秒
- 成功率：**100%（100/100）**
- 平均最大抬升高度：**0.333 m**
- 最低最大抬升高度：**0.194 m**

演示视频：[`results/franka_lift_demo.mp4`](results/franka_lift_demo.mp4)

## 仓库结构

```text
robot/                      Franka Panda 本体与执行器配置快照
task/                       抓取、抬升任务及奖励配置快照
training/                   PPO 配置和实际训练参数快照
evaluation/                 持续抬升成功率评估脚本
checkpoints/model_1499.pt   最终 RSL-RL checkpoint
checkpoints/exported/       TorchScript 与 ONNX 导出策略
results/                    评估 JSON 与演示视频
patches/                    本机 Isaac Lab 兼容性补丁
```

## 模型文件

| 文件 | 用途 | SHA-256 |
| --- | --- | --- |
| `checkpoints/model_1499.pt` | 完整 RSL-RL checkpoint，可恢复和评估 | `8aaadcb50c53a2d6b2717181f7370a33646bc57d3bc182b5cec260d3ff6e90fe` |
| `checkpoints/exported/policy.pt` | TorchScript 推理策略 | `27c99aa72d1c30fb7eb3d8c829753263b3f3c1a5e70021740f480ca50211c37c` |
| `checkpoints/exported/policy.onnx` | ONNX 模型入口 | `ba3e3230907890a1cd82bb8d79578ebdc49d738e08cb6746d339074b3ae4e27b` |
| `checkpoints/exported/policy.onnx.data` | ONNX 外部权重数据 | `5fb89567f324ded9fb636f45d63916e91e159bd03b5aa8a2395d6ea1ad5f0e8b` |

## 环境

本次训练使用：

- Ubuntu/Linux，Python 3.12
- NVIDIA Isaac Sim 6.0.1
- Isaac Lab `v3.0.0-beta2.patch1`（commit `ffff603ea`）
- RSL-RL 5.0.1
- PyTorch 2.11.0，CUDA 12.8

Isaac Lab 需要单独安装。本机使用的少量兼容性调整记录在 [`patches/isaaclab-local.patch`](patches/isaaclab-local.patch)，其中包括 Franka USD 资源路径调整。

## 复现训练

在已安装并激活的 Isaac Lab 环境中执行：

```bash
cd /path/to/IsaacLab
./isaaclab.sh train --rl_library rsl_rl \
  --task Isaac-Lift-Cube-Franka-v0 \
  --num_envs 4096 \
  --max_iterations 1500 \
  --viz none \
  --device cuda:0
```

训练的完整环境和算法参数快照分别位于 `training/env.yaml` 与 `training/agent.yaml`。4096 个并行环境对显存要求较高，可降低 `--num_envs` 以适配本机硬件。

## 回放最终 checkpoint

```bash
cd /path/to/IsaacLab
./isaaclab.sh play --rl_library rsl_rl \
  --task Isaac-Lift-Cube-Franka-Play-v0 \
  --num_envs 1 \
  --checkpoint /path/to/franka-robot-learning/checkpoints/model_1499.pt \
  --viz kit \
  --device cuda:0
```

## 运行 100 回合评估

评估脚本需要 Isaac Lab 源码目录，以复用其 RSL-RL 命令行模块：

```bash
cd /path/to/franka-robot-learning
export ISAACLAB_ROOT=/path/to/IsaacLab
python evaluation/evaluate.py \
  --headless \
  --episodes 100 \
  --checkpoint checkpoints/model_1499.pt \
  --output results/evaluation_100.json
```

指标实现见 `evaluation/metrics.py`，历史结果见 [`results/evaluation_100.json`](results/evaluation_100.json)。

## 说明

- `robot/`、`task/` 中的文件是训练时所用 Isaac Lab 配置的快照，便于阅读和复现实验；完整框架请从 Isaac Lab 官方仓库安装。
- ONNX 导出由 `.onnx` 与 `.onnx.data` 两个文件共同组成，使用时不要分开。
- 该模型只在仿真环境中评估，尚未验证真实机械臂部署、视觉输入或 sim-to-real 迁移。
- 上游配置的版权与许可信息见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
