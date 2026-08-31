# LAGTCN

本仓库是论文 **LAGTCN: Level-Aware Graph-Temporal Co-Evolution with Multiple
Graph Sources for Hierarchical Electric Load Forecasting** 的公开源码。

当前 `main` 分支已精简为核心实现仓库，只保留 LAGTCN 模型、数据与图构建、训练
评估入口，以及论文使用的 BU、TD-FP、MinT-SHR 三种调和方法。Baseline、批量实验
manifest 和整套测试代码不再放在主分支；精简前的完整复现版本保存在 Git 标签
[`full-reproduction-v1`](https://github.com/HI1003/LAGTCN/tree/full-reproduction-v1)，
需要时可以完整恢复。

## 目录结构

```text
lagtcn/
├── model.py              LAGTCN 模型结构与前向传播
├── data.py               GEFCom 数据读取及无泄漏 80/10/10 切分
├── graphs.py             H/HG 构建与 S/A/D 图源
├── train.py              训练、验证、checkpoint 和测试
├── metrics.py            预测精度与层级一致性指标
└── reconciliation.py     BU、TD-FP、MinT-SHR
data_preparation/          三个论文数据集的预处理 notebook
examples/forward_pass.py  不依赖数据集的前向示例
```

模型实现直接查看 [`lagtcn/model.py`](lagtcn/model.py)，三种调和方法集中在
[`lagtcn/reconciliation.py`](lagtcn/reconciliation.py)。

## 安装

代码面向 Python 3.10 和 PyTorch 2.3，精简后不再依赖 baseline 框架或 PyTorch
Geometric。

```bash
conda create -n lagtcn python=3.10 -y
conda activate lagtcn
pip install -r requirements.txt
```

CUDA 12.1 环境可改用 `requirements-cuda121.txt`。

## 数据准备

每个数据集应整理为：

```text
Data/<dataset>/
├── node_values.npy
├── normalization_params.npy
├── sum_matrix.csv
├── hierarchy_info.json
├── hierarchy.csv
├── adj_hierarchy.npy
├── adj_HGNN.npy
└── <带时间戳的原始 CSV>
```

支持 `GEFCom2012_2level`、`GEFCom2017QualifyingMatch_3level` 和
`GEFCom2017FinalMatch_4level`。先运行 [`data_preparation/`](data_preparation/)
中对应的 notebook，再生成固定 H/HG 图：

```bash
python -m lagtcn.graphs Data/<dataset>
```

原始数据不随仓库分发。

## 训练

训练入口的默认值就是论文协议：输入过去 168 小时、一次输出未来 24 小时、物理
batch size 为 128。

```bash
python -m lagtcn.train \
  --data-root Data \
  --dataset GEFCom2012_2level \
  --graph-mode H \
  --num-timesteps-in 168 \
  --num-timesteps-out 24 \
  --batch-size 128
```

使用 S、A 或 D 时，还要分别传入验证集选定的 `--static-threshold`、
`--adaptive-top-k` 或 `--dynamic-threshold`。训练结果写入
`Data/<dataset>/output/` 下的时间戳目录，包括最佳 checkpoint、配置、指标、验证集
预测和测试集 base 预测。

下面的 `24→1`、batch 32 仅用于快速检查 CPU 上能否运行，不是论文实验参数：

```bash
python -m lagtcn.train \
  --data-root Data \
  --dataset GEFCom2012_2level \
  --graph-mode H \
  --num-timesteps-in 24 \
  --num-timesteps-out 1 \
  --hidden-dim 16 \
  --num-layers 1 \
  --batch-size 32 \
  --epochs 1 \
  --patience 1 \
  --device cpu
```

不依赖数据集的正式形状前向检查：

```bash
python -m examples.forward_pass
```

## BU、TD-FP 与 MinT-SHR

训练结束后，将 `RUN` 指向结果目录。MinT-SHR 会用验证集预测残差估计收缩协方差，
再作用于测试集，不会使用测试集真实值估计权重。

```bash
RUN=Data/GEFCom2012_2level/output/<run>

python -m lagtcn.reconciliation \
  --method bu \
  --base-archive "$RUN/base_predictions.npz" \
  --sum-matrix Data/GEFCom2012_2level/sum_matrix.csv \
  --output "$RUN/reconciled_bu.npz"

python -m lagtcn.reconciliation \
  --method td_fp \
  --base-archive "$RUN/base_predictions.npz" \
  --sum-matrix Data/GEFCom2012_2level/sum_matrix.csv \
  --output "$RUN/reconciled_td_fp.npz"

python -m lagtcn.reconciliation \
  --method mint_shrink \
  --base-archive "$RUN/base_predictions.npz" \
  --validation-archive "$RUN/validation_predictions.npz" \
  --sum-matrix Data/GEFCom2012_2level/sum_matrix.csv \
  --output "$RUN/reconciled_mint_shrink.npz"
```

三种方法都在底层节点空间施加非负约束，再通过 `y = S b` 重建全部层级，因此输出
按构造满足非负与层级一致性。

引用信息见 [`CITATION.cff`](CITATION.cff)，代码采用 [MIT License](LICENSE)。
