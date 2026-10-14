# Information needed before production implementation

以下问题不会阻止本机骨架开发，但会影响远端正式实现：

1. 远端操作系统、Python 版本，以及是否允许 Docker/Conda；
2. Toys4K 和 Objaverse 实际主要格式，例如 OBJ、GLB、FBX、PLY；
3. 一个资产包含多个 Mesh、实例、骨骼或动画时，训练希望如何处理；
4. 训练输入是手工几何特征、点云、三角面序列、渲染图像，还是已有组内模型；
5. 是否统一三角化，以及三角化发生在筛选前还是训练前；
6. 面数、顶点数和显存预算上限；
7. Objaverse 负样本的具体选择标准，以及是否保留类别和选择原因（已确认：面数与顶点数
   小于 20000、均为三角网格、不要求质量；见 `docs/LABELING_GUIDE.md`，已写入
   `sources.objaverse.selection_rule` 与 `selection_limits`）；
8. 是否能从 Objaverse 补充少量人工确认的正样本，或从 Toys4K 找到负例，以减轻“来源等于标签”的混杂。

确认这些信息后，才能稳定实现规范化、近重复检测、高级 Artistic 特征、统一渲染和最终训练入口。

其中第 2、3、5 条不再需要口头回答：远端跑 `python -m mesh_filter probe --config
configs/paths.remote.yaml`，报告里的 `unconfigured_extensions`、`aggregate.containers`、
`multi_geometry_share`、`records_with_instancing`、`aggregate.longest_extent` 就是答案，见
`docs/REMOTE_WORKFLOW.md`。第 6 条由 `smoke` 的耗时与内存读数支撑。第 7 条仍需人来定。

## 本机骨架的明确边界

以下是当前代码里没有的东西。列出来是为了避免误以为"没报错就等于支持"：

- `render/` 模块不存在。渲染依赖 Blender、OpenGL 与 GPU，只能在数据设备上做，本机无法
  验证，因此不写占位实现，等远端确定渲染后端后再说；
- `canonicalize.py` 存在但没有接进流水线，manifest 的 `canonical_version` 恒为空字符串。
  在确认真实格式与轴向之前不应启用规范化；
- 近重复检测只覆盖"顶点数、面数、连通分量完全相同"的重导出副本（旋转、镜像、统一缩放、
  平移、顶点重排、换格式都算）。同一形状被重新细分、简化或合并顶点后会改变计数，落进
  不同桶，检测不到；
- 高级 Artistic 特征只有定义，没有实现，配置开关全为 `false`；
- 划分产出 `train/val/test/all`，另外 `build-matched-test` 与 `build-audit-gold` 会生成
  `matched_test.csv`、`audit_gold.csv` 和它的对照表。但盲审还缺渲染图：`audit_gold` 只给出
  样本表和分层，评审者要看的图片依赖尚未实现的渲染环节；
- `runtime.resume` 复用已解析的行，前提是只读源在两次运行之间没有变化。源变更后必须
  设回 `false` 全量重算，否则会保留过期特征。

## 需要用户拍板

负样本挑选标准已确认并落盘（见 `docs/LABELING_GUIDE.md`），代码里由 `smoke` 的
`selection_limits` 检查守住。剩下的是上面第 8 条：能否从 Objaverse 补少量人工确认的正样本，
或从 Toys4K 找出负例，让标签不再与来源完全相关。

这是整个项目最薄弱的一环：`label_origin=dataset_proxy` 只是把"来源与标签完全混杂"这件事
记录下来，并不能消除它。在补上跨来源样本之前，任何模型分数都至少有三种解释——学到的
可能是 Artistic 拓扑，也可能是 Toys4K 与 Objaverse 的格式/尺度/类别差异，还可能只是
"面数少"。区分它们要靠面数匹配测试集和人工盲审集，不能靠单一测试分数。
