# Data contract

## Asset identity

每个文件获得一个稳定、中性的 `asset_id`。它由数据源名称和相对路径计算，但 ID 本身不暴露正负标签。

原始文件路径由两部分组成：

```text
sources.<source_dataset>.root + raw_relpath
```

代码不得根据目录名推断标签。

## Manifest columns

| Column | Meaning |
| --- | --- |
| `asset_id` | 项目内部稳定 ID |
| `source_dataset` | 数据来源，仅用于追踪和偏差分析 |
| `source_uid` | 来源内部 UID；当前默认使用相对路径 |
| `raw_relpath` | 相对于只读 source root 的路径 |
| `sha256` | 原始文件内容哈希 |
| `file_format` | 小写扩展名，不含点 |
| `file_size_bytes` | 原始文件大小 |
| `label` | 1 为代理正样本，0 为代理负样本 |
| `label_origin` | `dataset_proxy`、`human_review` 或 `synthetic_fixture` |
| `selection_rule` | 样本如何进入当前数据集 |
| `parse_status` | `not_run`、`parsed` 或 `failed` |
| `validity_status` | `not_run`、`valid`、`valid_with_warnings` 或 `invalid` |
| `error_code` | 机器可统计的错误类型 |
| `duplicate_group_id` | 重复对象必须共享的分组 ID |
| `canonical_version` | 规范化产物版本；当前未启用 |
| `split` | `train`、`val` 或 `test` |

## Invariants

1. `raw/` 永不被覆盖；
2. 一个 `asset_id` 在一个 manifest 版本内只出现一次；
3. 同一 `duplicate_group_id` 不得跨 split；
4. `validity_status=invalid` 不自动推出 `label=0`；
5. 来源、路径、许可证和人工标签不得作为训练特征；
6. 正式测试集冻结后只能通过新版本替代，不直接修改。

## Generated tables

基础特征表以 `asset_id` 为主键。未能解析的对象仍保留一行，填写错误字段，数值特征留空，以便统计失败率而不是静默丢样本。

`deduplicate` 还会写一张几何指纹表（`outputs.fingerprints`），只在需要近重复检测时生成。
每行是 `asset_id` 加三组对旋转、镜像、统一缩放、平移和顶点顺序都不变的量：到质心距离的
64 个分位数、三角面面积的 16 个分位数、以及协方差特征值比。它只是中间产物，拆分阶段不读。

### `duplicate_group_id` 的取值

- 同一哈希的文件：`exact-<sha 前 16 位>`；
- 由近重复合并成的组（组内哈希不同）：`dup-<组内最小 asset_id>`；
- 没有哈希（`runtime.hash_files: false`）且不重复的对象：它自己的 `asset_id`。

合并规则刻意偏向"宁可多合"：错合只是少一点训练数据，漏合会让同一个模型横跨训练与测试，
把评测静默抬高。因此同一形状的镜像、内外翻转版本也算重复。

`validate` 默认每次全量重算。设 `runtime.resume: true` 时，它会复用输出表里 `parse_status=parsed`
的行，只处理新出现的资产，并在返回的计数里用 `resumed` 报告复用了多少行；已经不在当前
manifest 里的旧行会被丢弃。这个开关假定只读源没有变化，源变更后必须全量重算。

## Evaluation sets

由 `build-matched-test` 与 `build-audit-gold` 生成，两者都是以 `asset_id` 为主键的子集表，
不是新的数据划分，永远不参与训练。

### `matched_test.csv`

面数匹配测试集。默认从冻结的 `test` 划分里抽取（`matched_test.pool` 可改），在每个面数
分箱内把正负样本按顶点数排序后一一配对，因此两个标签的面数分布相同。列包括
`face_count_bin` 与 `match_pair_id`，同一对的两行共享 `match_pair_id`。

只有一个标签的分箱会被整体丢弃，并在命令输出与 `dropped_bins` 里列出原因，不做配平。
若一个分箱都不含两个标签，命令直接报错——那说明两个来源的面数分布不重叠，匹配集无法
构造，这本身就是必须知道的结论。

### `audit_gold.csv` 与 `audit_gold_key.csv`

人工盲审样本，输出两个文件：

- `audit_gold.csv`：评审者打开的表。只有 `anonymous_name`、`asset_id` 和空的评审列
  （`human_label`、`human_confidence`、`human_notes`、`reviewer`、`reviewed_at`），
  不含来源、路径、代理标签和面数；
- `audit_gold_key.csv`：把匿名名映射回来源与代理标签的对照表，同时记录分层键。

`human_label` 取值见 `docs/LABELING_GUIDE.md`。key 表不得交给评审者，否则盲审失效。
抽样按 `label × 面数分箱` 分层，并保证每个分层至少抽到 1 个（预算允许时按
`min_per_stratum` 提高下限），因为抽到 0 或 1 个的分层估不出错误率。

## Reports

`probe` 与 `smoke` 各写一份 JSON 报告（路径由 `outputs.probe_report`、`outputs.smoke_report`
决定）。它们描述一次运行，不是数据契约的一部分：字段可以随实现增补，但 `smoke` 报告里的
`verdict`、`checks[].name` 和 `sources.<name>` 下的汇总字段是远端回传与门控依赖的稳定接口。
