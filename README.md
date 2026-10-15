# Artistic Mesh Filter

用于研究高质量 Low-poly / Artistic Mesh 自动筛选的最小可复现项目。

当前数据设定：

- 1567 个 Toys4K 模型作为正样本；
- 3000 个 Objaverse 模型作为负样本，由一条独立流水线按种子 `20261014` 随机抽样，只保留
  顶点数与面数都小于 20000 的三角网格，并经 Blender 统一重导出；
- 本机只负责代码开发和小型测试；
- 真实模型处理、渲染和训练均在远端数据设备运行。

负样本的来历、它引入的存活偏差与导出器指纹，以及实测的泄漏通道都记在
[`docs/DATA_CARD.md`](docs/DATA_CARD.md) 和 [`docs/LABELING_GUIDE.md`](docs/LABELING_GUIDE.md)。

> 重要：当前标签与数据来源完全相关，因此这些标签属于 `dataset_proxy`（数据集代理标签），不是最终人工金标准。模型必须在面数匹配测试集和人工盲审集上额外评估，避免只学会区分 Toys4K 与 Objaverse。

## 项目能力

1. 从多个只读数据源建立统一 manifest；
2. 使用中性 `asset_id` 管理对象；
3. 解析 Mesh 并检查基础合法性；
4. 计算基础几何与三角形形状特征；
5. 把重复模型归入同一组：SHA-256 抓字节相同的文件，几何指纹抓旋转/缩放/镜像/顶点重排
   后的同一模型，保证它不会同时出现在训练集和测试集；
6. 生成可复现、重复组隔离的数据划分；
7. 训练只使用面数/顶点数的逻辑回归基线；
8. 保存运行配置、预测结果和评估指标。

另外提供这些能力：

- `probe`：只读扫描真实数据源，报告实际格式、白名单外格式、容器结构、尺度分布和
  解析失败原因，用来回答合成夹具回答不了的问题；
- `smoke`：在全量运行前对少量真实模型跑一遍完整单文件流程，给出机器可读的通过/不通过裁决；
- `build-matched-test`：从冻结划分里构造面数匹配的正负配对，让分数不再可能来自"这一类面数少"；
- `build-audit-gold`：抽一份盲审样本，输出评审者表与对照表两个文件；
- 环境快照：记录 Python、关键库版本、git commit 和加速器信息。

暂未锁定的内容见 [`docs/OPEN_QUESTIONS.md`](docs/OPEN_QUESTIONS.md)。

## 目录原则

- Git 仓库保存代码、配置模板、小型元数据和测试；
- 真实模型存放在远端 `<DATA_ROOT>`，不进入 Git；
- `raw/` 只读，任何处理结果写入单独目录；
- 标签和数据划分保存在表格中，不通过 `positive/negative` 目录名表达；
- 解析失败只表示当前流水线不可用，不等于 Artistic 负样本。

## 本机开发

推荐 Python 3.10 或 3.11。使用 uv：

```bash
uv run --extra dev,train pytest
uv run --extra dev ruff check .
```

或使用 pip：

```bash
python -m pip install -e ".[dev,train]"
pytest
ruff check .
```

本机只使用 `tests/fixtures/` 中的小型 OBJ 文件：

```bash
uv run python scripts/00_build_manifest.py --config configs/dev.yaml
uv run python scripts/01_validate.py --config configs/dev.yaml
uv run python scripts/02_deduplicate.py --config configs/dev.yaml
uv run python scripts/03_split.py --config configs/dev.yaml
uv run python scripts/05_probe_formats.py --config configs/dev.yaml
uv run python scripts/06_smoke_test.py --config configs/dev.yaml
uv run python scripts/07_build_matched_test.py --config configs/dev.yaml
uv run python scripts/08_build_audit_gold.py --config configs/dev.yaml
```

`04_train_count_baseline.py` 需要 `train` 额外依赖（scikit-learn、joblib）：

```bash
uv run --extra train python scripts/04_train_count_baseline.py --config configs/dev.yaml
```

所有脚本只是 `python -m mesh_filter <command> --config ...` 的薄封装，两者等价。

## 本机查看模型

夹具和抽样结果需要肉眼确认时，用 VS Code 的只读预览插件即可，不需要装 Blender：

- `logicossoftware.logicos-3d-viewer`（3D Viewer）：支持 OBJ、GLB/glTF 2.0、STL、PLY，
  带线框切换、坐标轴、模型统计和原始尺寸，且从不修改源文件。线框模式正好用来检查本项目
  关心的边流，是对这些格式的主选。
- `cesium.gltf-vscode`（glTF Tools）：glTF/GLB 预览与校验，并支持 GLB 与文本 glTF 互转。
  需要做格式校验或转换时用它。
- `AssetToolLabs.mesh-viewer-vscode`（3D Mesh Viewer）：覆盖 FBX、USD、STEP、DAE、3MF
  等更多格式，功能更重，需要时再装。

```bash
code --install-extension logicossoftware.logicos-3d-viewer
```

这些扩展只用于人工查看，不参与流水线，也不改变仓库里的模型文件。

## 远端运行

1. 复制 `configs/paths.example.yaml` 为 `configs/paths.remote.yaml`；
2. 填写 Toys4K、Objaverse 和输出目录；
3. 不要将真实路径配置提交 Git；
4. 先跑 `probe` 看真实格式，再跑 `smoke` 拿到通过裁决，然后才处理全量数据；
5. 正式运行时保存同一 Git commit、配置、依赖版本和随机种子。

典型流程：

```text
probe（只读，看真实格式）
→ smoke（门槛，未通过先修）
→ build-manifest
→ validate
→ deduplicate
→ split
→ train-count-baseline
```

命令、检查项和回传清单见 [`docs/REMOTE_WORKFLOW.md`](docs/REMOTE_WORKFLOW.md)。正式数据处理前
请同时阅读 [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md) 与
[`docs/LABELING_GUIDE.md`](docs/LABELING_GUIDE.md)。

## 输出说明

- `assets_v1.csv`：资产总账，字段定义见 `data/schemas/manifest.schema.json`；
- `basic_features_v1.csv`：解析状态、合法性状态和基础特征；
- `assets_deduplicated_v1.csv`：加入精确重复组的 manifest；
- `train.csv`、`val.csv`、`test.csv`：冻结的数据划分；
- `matched_test.csv`：面数匹配的正负配对测试集，只用于诊断；
- `audit_gold.csv` 与 `audit_gold_key.csv`：盲审表与对照表，后者不得交给评审者；
- `runs/<run-id>/`：实验配置、指标、预测和模型文件；
- `reports/probe_report.json`、`reports/smoke_report.json`：探测与冒烟报告。

Parquet 支持通过可选依赖启用；最小版本默认使用 CSV，以减少远端首次部署的依赖问题。
