# Dataset card (draft)

## Purpose

研究是否能够通过可解释的几何指标，从混合 3D 资产中筛选高质量 Artistic/Low-poly Mesh。

## Composition

- Positive proxy set：约 1600 个 Toys4K 模型；
- Negative proxy set：约 3000 个用户选择的 Objaverse 模型。

实际数量、类别分布、许可证和文件格式将在远端 manifest 生成后补充。

## Negative selection rule

负样本的来历已从生产流水线的原始记录核实（2026-10-15）。流水线本体在数据设备的
`D:\CodexTools\objaverse_bad_3000\`（`pipeline.py`、`blender_worker.py`、`download_worker.py`），
它的 `status.json` 记录了全部参数：

```text
seed 20261014   target 3000   max_vertices 20000   max_faces 20000
processed 13992   accepted 3000   rejected 3478   failed 7514
```

据此，负样本不是人工精选，而是：

- 按种子 `20261014` 的顺序遍历 Objaverse 候选，逐个下载；
- 用 Blender 导入，接受条件为顶点数小于 20000 **且**面数小于 20000 **且**该网格是三角网格；
- 接受后用 Blender 的 OBJ 导出器重写一遍，存成 `models/<uid>.obj`。

每个被接受的模型在上游 `manifest.csv` 里都有记录，字段为
`uid, vertices, faces, triangular, obj_bytes, obj_path, processed_at`。所以"均为三角网格"
这一条是可核对的：以上游 `manifest.csv` 的 `triangular` 字段为准，不需要重新解析。

记录到本项目的方式：`sources.objaverse.selection_rule` 存机器可读的规则名，
`sources.objaverse.selection_limits` 存上下界，`smoke` 会据此检查抽样资产，超出即判失败。

## Two biases this rule introduces

存活偏差。13992 个候选里有 7514 个（54%）下载超时、TLS 断连或 Blender 导入失败，直接出局。
这份负样本集只包含"能下载且能被 Blender 导入"的模型，不代表 Objaverse 的整体分布。

导出器指纹。3000 个负样本全部经过 Blender 的 OBJ 导出器重写，而 1567 个正样本来自 Toys4K
的原始发布。顶点顺序、法线写法、小数精度都会带 Blender 的系统性痕迹。这意味着"模型学会
区分两个来源"这条捷径在这个数据集里格外宽：它甚至不需要看几何，看文件的写法就够了。

实测：尺度通道（Toys4K 最长边恒为 2.0，Objaverse 0.71–12160，零重叠）与 UV 存在率
（70% 对 0%）都是几乎完美的判别器；面数通道则大幅重叠（[12, 15000] 带内各占 92.8% 和
100%），有信息量但不是一票否决。见 `docs/LABELING_GUIDE.md`。

另外，实测发现 3 个重复组（7 个资产）同时带正负标签——同一份几何体经两个来源进入了数据
集。它们已被整体移出并在 `splits/v1/conflicts.csv` 中记录。

## Known limitation

标签与来源完全相关。任何高测试分数都可能来自来源差异，而非 Artistic 拓扑。正式结论必须包含面数匹配评估和人工盲审集。

负样本的面数上限来自上述筛选规则，属于标签定义的一部分。若两个来源的面数分布几乎不
重叠，"面数少"就会被当成"质量好"。跑完 `validate` 后应先比较两个来源的 `face_count`、
`vertex_count` 分布，再解释计数基线的分数。

"均为三角网格"无法在本流水线内复核：trimesh 加载时即三角化，源文件原本是否为三角网格
不可从加载结果还原。

## License and provenance

待从真实数据清单补充。许可证不明或不允许训练的对象必须单独标记，不能因为几何良好就进入训练集。
