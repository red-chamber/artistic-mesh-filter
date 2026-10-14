# Artistic Mesh Filter

用于研究高质量 Low-poly / Artistic Mesh 自动筛选的最小可复现项目。

当前数据设定：

- 约 1600 个 Toys4K 模型作为正样本；
- 约 3000 个经人工选择的 Objaverse 模型作为负样本；
- 本机只负责代码开发和小型测试；
- 真实模型处理、渲染和训练均在远端数据设备运行。

> 重要：当前标签与数据来源完全相关，因此这些标签属于 `dataset_proxy`（数据集代理标签），不是最终人工金标准。模型必须在面数匹配测试集和人工盲审集上额外评估，避免只学会区分 Toys4K 与 Objaverse。

## 项目能力

当前骨架提供：

1. 从多个只读数据源建立统一 manifest；
2. 使用中性 `asset_id` 管理对象；
3. 解析 Mesh 并检查基础合法性；
4. 计算基础几何与三角形形状特征；
5. 基于 SHA-256 标记精确重复文件；
6. 生成可复现、重复组隔离的数据划分；
7. 训练只使用面数/顶点数的逻辑回归基线；
8. 保存运行配置、预测结果和评估指标。

暂未锁定的内容见 [`docs/OPEN_QUESTIONS.md`](docs/OPEN_QUESTIONS.md)。

## 目录原则

- Git 仓库保存代码、配置模板、小型元数据和测试；
- 真实模型存放在远端 `<DATA_ROOT>`，不进入 Git；
- `raw/` 只读，任何处理结果写入单独目录；
- 标签和数据划分保存在表格中，不通过 `positive/negative` 目录名表达；
- 解析失败只表示当前流水线不可用，不等于 Artistic 负样本。

## 本机开发

推荐 Python 3.10 或 3.11。安装开发依赖：

```bash
python -m pip install -e ".[dev,train]"
pytest
```

本机只使用 `tests/fixtures/` 中的小型 OBJ 文件：

```bash
python -m mesh_filter build-manifest --config configs/dev.yaml
python -m mesh_filter validate --config configs/dev.yaml
python -m mesh_filter deduplicate --config configs/dev.yaml
python -m mesh_filter split --config configs/dev.yaml
```

也可使用 `scripts/` 中按顺序编号的入口脚本。

## 远端运行

1. 复制 `configs/paths.example.yaml` 为远端私有配置，例如 `configs/paths.remote.yaml`；
2. 填写 Toys4K、Objaverse 和输出目录；
3. 不要将真实路径配置提交 Git；
4. 先分别取少量模型做冒烟测试，再处理全量数据；
5. 正式运行时保存同一 Git commit、配置、依赖版本和随机种子。

典型流程：

```text
build-manifest
→ validate
→ deduplicate
→ split
→ train-count-baseline
```

正式数据处理前请阅读：

- `docs/REMOTE_WORKFLOW.md`
- `docs/DATA_CONTRACT.md`
- `docs/LABELING_GUIDE.md`

## 输出说明

- `assets_v1.csv`：资产总账；
- `basic_features_v1.csv`：解析状态、合法性状态和基础特征；
- `assets_deduplicated_v1.csv`：加入精确重复组的 manifest；
- `train.csv`、`val.csv`、`test.csv`：冻结的数据划分；
- `runs/<run-id>/`：实验配置、指标、预测和模型文件。

Parquet 支持通过可选依赖启用；最小版本默认使用 CSV，以减少远端首次部署的依赖问题。
