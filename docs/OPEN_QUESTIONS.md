# Information needed before production implementation

以下问题不会阻止本机骨架开发，但会影响远端正式实现：

1. 远端操作系统、Python 版本，以及是否允许 Docker/Conda；
2. Toys4K 和 Objaverse 实际主要格式，例如 OBJ、GLB、FBX、PLY；
3. 一个资产包含多个 Mesh、实例、骨骼或动画时，训练希望如何处理；
4. 训练输入是手工几何特征、点云、三角面序列、渲染图像，还是已有组内模型；
5. 是否统一三角化，以及三角化发生在筛选前还是训练前；
6. 面数、顶点数和显存预算上限；
7. Objaverse 负样本的具体选择标准，以及是否保留类别和选择原因；
8. 是否能从 Objaverse 补充少量人工确认的正样本，或从 Toys4K 找到负例，以减轻“来源等于标签”的混杂。

确认这些信息后，才能稳定实现规范化、近重复检测、高级 Artistic 特征、统一渲染和最终训练入口。
