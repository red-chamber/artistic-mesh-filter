# Remote workflow

本机只写代码、跑夹具；所有真实数据操作都在数据设备上完成。本文按执行顺序排列，
每一步都给出命令、要看的输出和要回传的文件。

核心约束：本机能证明代码逻辑正确，不能证明代码适配真实模型。因此第 3 步（探测）
和第 4 步（冒烟）是全量运行前的硬门槛，不能跳过。

## 1. 准备环境

远端拉取与本机相同的 Git commit，建独立环境：

```bash
git rev-parse HEAD            # 记录 commit，与回传结果一起保存
python -m pip install -e ".[train,dev]"
```

也可以使用 uv：

```bash
uv sync --extra train --extra dev
```

保存环境快照（Python、依赖、CUDA、显卡）：

```bash
python scripts/98_capture_environment.py --output reports/environment.json
```

## 2. 配置路径

```bash
cp configs/paths.example.yaml configs/paths.remote.yaml
```

`configs/paths.remote.yaml` 已被 `.gitignore` 排除，不要提交。需要填写：

- `sources.toys4k.root`、`sources.objaverse.root`：只读原始数据目录；
- `outputs.*`：全部输出位置，正式输出不要与冒烟输出共用路径；
- `formats.extensions`：允许摄入的扩展名；
- `smoke.*`：冒烟门槛；
- `split.*`：正式划分比例与随机种子。

## 3. 格式探测（只读，先跑这一步）

```bash
python -m mesh_filter probe --config configs/paths.remote.yaml --sample-size 20
```

它不改动任何源文件，也不需要先有 manifest。报告里要看的是：

| 字段 | 回答的问题 |
| --- | --- |
| `ingestable_count` | 白名单内实际找到多少文件；为 0 说明 `formats.extensions` 写错了 |
| `unconfigured_extensions` | 源里存在但白名单没覆盖的格式，这些文件对其他命令完全不可见 |
| `unconfigured_probe` | 这些白名单外格式能否被 trimesh 解析 |
| `aggregate.parse_rate` | 抽样解析成功率 |
| `aggregate.containers` / `multi_geometry_share` | 一个文件是单 Mesh 还是多几何体场景 |
| `aggregate.records_with_instancing` | 是否存在非单位变换（实例化、预变换） |
| `aggregate.parts_with_uv` | 是否带 UV/贴图 |
| `aggregate.longest_extent` | 尺度分布，用于判断是否需要统一缩放 |
| `aggregate.quarantine_buckets` | 失败原因分类 |

这些读数直接对应 `docs/OPEN_QUESTIONS.md` 的第 2、3、5 条。回传
`reports/probe_report.json`。

已知限制：trimesh 在加载时就会三角化，因此报告无法说明源文件里原本是不是四边形
或多边形面。需要这个信息时必须另用 Assimp/Open3D 做原始解析，不能从本报告推断。

## 4. 冒烟测试（门槛，未通过不要往下走）

```bash
python -m mesh_filter smoke --config configs/paths.remote.yaml --sample-size 20
```

它按 `runtime.sample_per_source` 或 `--sample-size` 等距抽取文件，对每个文件执行与
`validate` 完全相同的加载与特征计算，然后给出一份裁决。退出码为 0 表示通过，非 0
表示未通过，可以直接用于 shell 门控。

检查项：

| 检查 | 含义 |
| --- | --- |
| `<source>.ingestable_not_empty` | 该来源在白名单内找到了文件 |
| `<source>.sample_not_empty` | 抽到了样本 |
| `<source>.no_silent_drops` | 抽样数 = 解析成功 + 失败，没有任何文件被静默丢弃 |
| `<source>.parse_rate` | 该来源抽样解析率不低于 `smoke.min_parse_rate` |
| `<source>.seconds_per_asset` | 最慢单个模型不超过 `smoke.max_seconds_per_asset` |
| `<source>.peak_memory` | 单模型 Python 堆峰值不超过 `smoke.max_peak_mb` |
| `<source>.selection_limits` | 抽样资产都落在该来源声明的筛选界内（仅当配置里写了 `selection_limits` 时出现） |
| `overall.parse_rate` | 两个来源合计解析率达标 |

`selection_limits` 检查的是负样本的挑选规则：3000 个 Objaverse 模型是提前按"面数、顶点数
小于 20000 且均为三角网格"筛过的（见 `docs/LABELING_GUIDE.md`）。界内的失败说明挂上来的
目录和标签定义不符，此时分数没有意义。注意面数是三角化之后测得的，而"源文件原本是否
为三角网格"无法在流水线内复核。

`peak_memory` 只统计 Python 堆（tracemalloc），不含 C 层与显存，因此是下界，不能当作
真实内存预算。真实峰值请在远端用系统工具另测。

报告还包含 `quarantine_plan`：失败对象按 `parse_error` / `invalid_geometry` /
`unsupported_format` 分类，并给出样本路径。进入 quarantine 不等于它是 Artistic 负样本，
只表示当前流水线用不了它。

抽样是确定性的（按排序等距），因此同一样本量下两次运行的样本集合一致，便于对比。

回传 `reports/smoke_report.json`。未通过时按 `checks[].name` 定位，先修代码或配置，
再重跑本步。

## 5. 全量流程

冒烟通过后：

1. 复制一份配置，把 `outputs.*` 指向正式路径（不要复用冒烟输出）；
2. 把 `limit_per_source` 设为 `null`；
3. 按顺序执行：

```bash
python -m mesh_filter build-manifest  --config configs/paths.remote.yaml
python -m mesh_filter validate        --config configs/paths.remote.yaml
python -m mesh_filter deduplicate     --config configs/paths.remote.yaml
python -m mesh_filter split           --config configs/paths.remote.yaml
python -m mesh_filter build-matched-test --config configs/paths.remote.yaml
python -m mesh_filter build-audit-gold   --config configs/paths.remote.yaml
python -m mesh_filter prepare-review     --config configs/paths.remote.yaml
python -m mesh_filter train-count-baseline --config configs/paths.remote.yaml \
  --experiment configs/experiments/count_baseline.yaml
```

`prepare-review` 写出匿名化的 3D 查看目录（`outputs.review_dir`），是人工盲审的交接物：
把一个扁平目录连同 `REVIEW_PROTOCOL.md` 交给评审者，用任意 3D 查看器逐个打开、按说明填
`human_label`。`audit_gold_key.csv` 不能一起交出去。

`split` 的输出里要确认 `conflicting_groups`：非零说明存在"同一几何体既是正样本又是负样本"
的重复组，它们已被整体移出并记进 `conflicts.csv`。数量应该是零星的；如果很多，说明两个
来源的重叠比预期严重，`label_origin=dataset_proxy` 这个前提本身就有问题了。

`build-matched-test` 与 `build-audit-gold` 必须在 `split` 之后、训练之前跑，**并且要看它们
打印出来的分箱报告**：

- `build-matched-test` 的 `dropped_bins` 里如果是"面数分布不重叠"造成的整档丢弃，那么计数
  基线的高分不能解释为拓扑能力，后面所有结论都要打折扣；如果它直接报错，说明两个来源在
  面数上根本不重叠，这时要先决定怎么办，再谈训练；
- `build-audit-gold` 的 `strata` 表要确认每个分层都抽到了人。拿 `audit_gold_key.csv` 只用于
  事后对账，评审时不能打开。

长跑建议：在配置里设 `runtime.resume: true`，中断后重跑 `validate` 会复用已解析的行，
只处理新增或上次失败的对象。前提是只读源在这期间没有变化；源变了就设回 `false` 全量重算。

`deduplicate` 要在 `validate` 之后跑，因为它的近重复阶段依赖特征表：它先用

```text
(vertex_count, face_count, component_count)
```

挑出可能重复的对象，只对这些对象重新解析并算指纹。这一步选出的候选是精确的而不是启发式的
——计数不同的两个资产按定义就不可能互为近重复，所以跳过它们不会漏掉任何东西。看过
`assets_in_duplicate_groups`、`groups_merged_by_near_duplicate` 和 `oversized_bucket_count`
之后再进 `split`；`oversized_bucket_count` 非零说明桶键对这份数据太粗，需要看
`oversized_buckets` 里的具体键再决定。把 `deduplicate.near_duplicate` 设为 `false`
可以跳过整个解析阶段，但冻结划分给训练用之前不要关。

## 6. 回传清单

回传：

```text
reports/probe_report.json
reports/smoke_report.json
reports/environment.json
data/manifests/assets_v1.csv
data/manifests/basic_features_v1.csv
data/manifests/assets_deduplicated_v1.csv
data/splits/v1/          （含 matched_test.csv）
data/labels/             （含 audit_gold.csv；audit_gold_key.csv 一并回传但不得交给评审者）
review/                  （prepare-review 的匿名化模型目录，交给评审者的那一份）
runs/<run-id>/           （resolved_config、metrics.json、predictions.csv）
logs/
少量错误案例与画廊
```

人工盲审还需要 `audit_gold.csv` 里每个 `asset_id` 的多视角渲染图。渲染在本流水线之外，
所以在真正确认渲染后端之前，盲审这一步只能先准备好样本表，图片要另行生成。

不回传：原始模型、canonical 产物、全量渲染、模型权重。本机通过特征表、指标和失败
日志做分析，不需要拿到大文件。

## 7. 生产注意事项

- 正式特征提取必须在同一个远端环境完成，不要分设备拼接结果；
- 不同 Blender / Assimp / OpenGL 版本会改变导入和渲染结果，渲染若进训练必须固定版本；
- 实际使用的 batch size、混合精度、梯度累积、进程数和采样方式必须记录进
  `runs/<run-id>/resolved_config.json`；
- 在确认真实模型格式之前，不启用自动修复、顶点合并或覆盖式三角化；
- 正式测试集一旦冻结，只能通过新版本替代，不直接修改。
