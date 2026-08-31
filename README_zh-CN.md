# LAGTCN

本仓库是论文 **LAGTCN: Level-Aware Graph-Temporal Co-Evolution with Multiple
Graph Sources for Hierarchical Electric Load Forecasting** 的独立复现代码库。

仓库包含 LAGTCN、对比模型、层级调和方法、正式实验清单生成与后处理代码。
原始 GEFCom 数据、处理后的大数组、检查点和服务器配置不会进入 Git；数据准备
方式和目录契约见 [docs/DATA.md](docs/DATA.md)。

快速验证：

```bash
python scripts/make_synthetic_dataset.py
python -m pytest -q
```

CPU 一轮训练命令、环境安装方式及目录结构见英文主页 [README.md](README.md)，
完整实验流程见 [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)。合成数据仅用于
检查代码能否运行，不能用于论文数值结论。

