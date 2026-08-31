# LAGTCN

本仓库是论文 **LAGTCN: Level-Aware Graph-Temporal Co-Evolution with Multiple
Graph Sources for Hierarchical Electric Load Forecasting** 的公开源码。

仓库只保留两类内容：运行 LAGTCN 必须的源码，以及完整复现论文需要的代码。论文
使用的六个对比模型（DLinear、PatchTST、N-HiTS、iTransformer、DCRNN、MTGNN）
和三种调和方法（BU、TD-FP、MinT-SHR）也保留；探索性模型、旧实验入口、服务器
脚本、原始数据和 checkpoint 不公开在这里。

## 目录结构

```text
lagtcn/                       模型运行源码
├── train.py                  训练、验证与测试入口
├── core/                     数据、图构建、指标和训练逻辑
├── models/                   LAGTCN 与六个论文基线
└── reconciliation/           BU、TD-FP、MinT-SHR 基础实现
reproduction/                 论文复现流程
├── data/                     特征、结构图和合成数据生成
├── manifests/                模型矩阵与图参数搜索清单
├── selection/                仅使用验证集的参数选择
└── evaluation/               调和、正式结果检查和效率测试
configs/                      论文冻结的模型与图参数
data_preparation/             三个数据集的预处理 notebook
docs/                         数据契约与完整复现说明
tests/                        模型、实验协议和调和测试
results/                      小型论文结果文件的预留目录
```

因此，如果只想看模型源代码，直接查看 [`lagtcn/`](lagtcn/)；如果要复现整篇论文，
按 [`reproduction/`](reproduction/) 和
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) 的顺序执行。

## 正式配置与快速检查的区别

论文正式实验固定使用：输入 168 小时、输出未来 24 小时、随机种子 42/43/44、
有效 batch size 128。训练入口也以 `168→24` 和 batch 128 为默认值，其他冻结参数
位于 [`configs/`](configs/)；正式实验应由 manifest 生成，不建议手工拼接参数。

下面的 `24→1`、batch 32 命令只是为了在 CPU 上快速确认代码能运行，并不是为了
提高 README 可读性，也不能用于复现论文数值：

```bash
python -m reproduction.data.make_synthetic

python -m lagtcn.train \
  --data-root Data \
  --dataset GEFCom2012_2level \
  --model-name LAGTCN \
  --graph-mode H \
  --gnn-type gcn \
  --temporal-type patch_transformer \
  --num-timesteps-in 24 \
  --num-timesteps-out 1 \
  --hidden-dim 16 \
  --num-layers 1 \
  --batch-size 32 \
  --epochs 1 \
  --patience 1 \
  --device cpu \
  --no-plots \
  --experiment-stage smoke \
  --experiment-id synthetic_smoke \
  --output-namespace ae/smoke
```

安装环境、测试命令和正式运行示例见 [英文主页](README.md)。数据目录要求见
[`docs/DATA.md`](docs/DATA.md)，完整实验步骤见
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)。合成数据只能做软件检查，
不能用于论文结论。
