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
