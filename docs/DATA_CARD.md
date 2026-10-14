# Dataset card (draft)

## Purpose

研究是否能够通过可解释的几何指标，从混合 3D 资产中筛选高质量 Artistic/Low-poly Mesh。

## Composition

- Positive proxy set：约 1600 个 Toys4K 模型；
- Negative proxy set：约 3000 个用户选择的 Objaverse 模型。

实际数量、类别分布、许可证和文件格式将在远端 manifest 生成后补充。

## Negative selection rule

负样本不是从 Objaverse 随机抽取的，而是提前人工筛过（用户 2026-10-14 说明）：

- 面数小于 20000，且顶点数小于 20000；
- 均为三角网格；
- 不要求质量：允许破碎几何、非必要的面数堆积和局部退化。

记录方式：`sources.objaverse.selection_rule` 存机器可读的规则名，
`sources.objaverse.selection_limits` 存上下界，`smoke` 会据此检查抽样资产，超出即判失败。
细节与后果见 `docs/LABELING_GUIDE.md`。

## Known limitation

标签与来源完全相关。任何高测试分数都可能来自来源差异，而非 Artistic 拓扑。正式结论必须包含面数匹配评估和人工盲审集。

负样本的面数上限来自上述筛选规则，属于标签定义的一部分。若两个来源的面数分布几乎不
重叠，"面数少"就会被当成"质量好"。跑完 `validate` 后应先比较两个来源的 `face_count`、
`vertex_count` 分布，再解释计数基线的分数。

"均为三角网格"无法在本流水线内复核：trimesh 加载时即三角化，源文件原本是否为三角网格
不可从加载结果还原。

## License and provenance

待从真实数据清单补充。许可证不明或不允许训练的对象必须单独标记，不能因为几何良好就进入训练集。
