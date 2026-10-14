# Dataset card (draft)

## Purpose

研究是否能够通过可解释的几何指标，从混合 3D 资产中筛选高质量 Artistic/Low-poly Mesh。

## Composition

- Positive proxy set：约 1600 个 Toys4K 模型；
- Negative proxy set：约 3000 个用户选择的 Objaverse 模型。

实际数量、类别分布、许可证和文件格式将在远端 manifest 生成后补充。

## Known limitation

标签与来源完全相关。任何高测试分数都可能来自来源差异，而非 Artistic 拓扑。正式结论必须包含面数匹配评估和人工盲审集。

## License and provenance

待从真实数据清单补充。许可证不明或不允许训练的对象必须单独标记，不能因为几何良好就进入训练集。
