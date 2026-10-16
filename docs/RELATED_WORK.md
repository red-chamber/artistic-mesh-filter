# 相关文献：别人怎么定义"好 mesh"

调查目的：课题组三条判据（面数顶点数不能太大、面的形状不能太规则、顶点分布整齐）在已发表工作里
有没有对应物。按与判据的接近程度排序。每条给可实现/可对齐的阈值。

## 1. 课题组自己的筛选流程（最直接，可逐条对齐）

**LATO**（ICML 2026，arXiv 2603.06357，通讯作者 `weiyangcs@hust.edu.cn`）的数据构建：

> We construct our dataset from Objaverse-XL, G-Objaverse, ShapeNet, Toys4K, and 3D-FUTURE.
> We discard meshes with more than 20k or fewer than 500 faces, excessive discretization
> artifacts (over-aggressive vertex merging), more than 20 connected components, or an
> intersection ratio above 10, yielding roughly 300k meshes.

它的 artist-mesh 评测集从 G-Objaverse 留出 300 个、从 Toys4K 与 ShapeNet 各留出 50 个——
**Toys4K 在这条线里就是 artist mesh 的来源**，与本项目拿它当正样本池一致。

**MeshRipple**（CVPR 2026，arXiv 2512.07514）附录 B.1 "Filtering Training Data" 是完整实现级规则：

> We discard meshes with face counts outside the range [500, 20k], vertex-to-face ratios > 0.8,
> or where narrow faces (minimum angle < 5°) exceed 20% of the total. To ensure topological
> simplicity, we exclude meshes exceeding a maximum BFS displacement of 100, a boundary length
> of 500, or 20 connected components. The component count is determined after pruning small
> clusters (<10 faces) located within a normalized distance of 0.05 from valid components.

外加顶点合并伪影检查：顶点数下降超过 50%、或新引入的自交/重叠面超过 10% 的一律丢弃
（自交用 trimesh 检测，重叠面用共面分组 + 分离轴定理验证）。

两条流程共有的量：**面数区间 [500, 20k]**。MeshRipple 的"窄面（最小内角 < 5°）占比 ≤ 20%"
与本项目的 `small_angle_ratio` 是同一个量。

## 2. artist-created mesh 生成文献的质量语言

MeshAnything（arXiv 2406.10163）对"坏"的描述：抽取式方法
"rely on dense faces and ignore geometric features"、"fail to capture sharp edges and flat
surfaces"；对"好"的度量是 topology efficiency——面数与顶点数都少（"even when remesh methods
achieve a comparable face count, the number of vertices is still several times higher than
ours"）。MeshAnything V2（arXiv 2408.02555）、DeepMesh（arXiv 2503.15265，ICCV 2025）、
EdgeRunner（arXiv 2409.18114）延续同一套语言：compact、topological completeness（不破面）、
忠实于给定形状、面数可控。

**这条线里没有一篇用三角形形状或顶点分布当判据**；评测一律是 Chamfer / Hausdorff / 法向一致性
加面数。

与本项目判据二（冗余的面）直接相关的两条：

- **CGAL 的 `remesh_planar_patches`**（`doc.cgal.org/latest/PMP_Remeshing/`）：把边连通的
  共面面聚成 patch（相邻面法向夹角在一个容差余弦内），把 patch 边界上共线的边串成线段，
  线段端点与多面交汇点是 corner，然后用**只用 corner 顶点**的剖分当作该 patch 的最小
  三角形数。本项目的 `redundant_face_share` 就是这个最小数与实际面数之比；容差（3°）
  同样出自这份文档的口径——"共面"与"共线"共用同一个余弦。
- **Klein, Liebich & Straßer 1996，"Mesh Reduction with Error Control"**：误差受控的顶点
  删除——每个顶点带一个"删掉它会产生多大 Hausdorff 距离"的 potential error，按此排序逐个
  删除；论文明确指出平面区域会被最先削减。这是"冗余 = 可在误差内删除"这一表述的经典来源。

## 3. 经典网格质量文献（方向相反，必须知道）

数值与几何处理界的 "mesh quality" 有标准定义与指标。Sorgente et al.（CGF 2023）的综述把目标
写成 "good geometrical properties and the lowest possible number of elements, able to produce
results in a target range of accuracy"。三角形质量 $q = 4\sqrt3 A / \sum l_i^2$ 是常用指标，
**近等边 = 高质量**，细长面 = 低质量；isotropic remeshing 的目标明确写成 "as equilateral as
possible"（Du, Jin & Tong, JSME 2015），它把"扫描设备获得的、多细长面的网格"直接称为低质量。

所以判据第二条在文献里的正确读法是**逆向标记**：均匀、接近等边的三角形是 isotropic remeshing
与扫描重建的产出特征（这一点有文献支持，而且本项目实测 `equilateral_median` 确实干净地跟踪
三角形形状），但没有人把它当"坏"。另外，曲率自适应加密（采样密度
$\varnothing(x) = \sqrt{|K|} + H^2$，Chen et al. 2012 起）是自适应重网格的标准做法，所以
"平坦处稀疏、转折处密集"也不是手工独有——它是"自适应重网格"的特征。

各向异性重网格（Anisotropic polygonal remeshing, SIGGRAPH 2003 起）追求"细长且沿主曲率方向"
的三角形，与判据第二条的正面方向一致——但它同样是自动方法能产出的，不能当"手工"的证据。

## 4. 数据集策展与质量标注

Objaverse++（ICCV 2025 Workshop，arXiv 2504.07334）：人工标注 1 万个 Objaverse 对象，再训练
网络自动传播到约 79 万个。质量分四级——低：无语义；中：可识别但缺材质贴图；高：身份清晰、
贴图完整；优：专业贴图、审美协调——另附五个二元属性（透明、场景、单色、非单一物体、人物）。

论文里两句与本项目直接相关：

> about one-fifth of the original Objaverse dataset are scanned objects that tend to have high
> polygon counts and low texture quality

> 3D objects with lower polygon counts and vivid color tones resemble those of human artists

注意它的质量分是**外观与语义维度**的，不含拓扑判据。同一方向还有 Gen3DEval（CVPR 2025）、
3D Arena（arXiv 2506.18787）。

## 5. 与三条判据的对照

| 判据 | 文献对应 | 结论 |
| --- | --- | --- |
| 面数/顶点数不能太大 | LATO 与 MeshRipple 的面数区间；AM 文献的 topology efficiency；Objaverse++ 的"扫描件面数高" | **直接支持**，且有实现级阈值可对齐（[500, 20k]） |
| 面的形状 / 冗余的面（学长原意） | 经典质量文献方向相反（近等边 = 好）；MeshRipple 反而设"窄面占比 ≤ 20%"的上限 | 等边本身**没有直接支持**，只能作"自动重网格/扫描"的逆向标记；"冗余的面"这一半由 MeshRipple 的窄面规则与 AM 文献的 "dense faces" 间接支持 |
| 顶点分布整齐 / 平坦处稀疏、转折处密集 | AM 文献的 "capture sharp edges and flat surfaces"；但曲率自适应加密是自适应重网格的标准做法 | **部分支持**，且不是手工独有 |

## 6. 对本项目的待办

- **口径对齐（V/F）**：MeshRipple 的 `vertex-to-face ratio > 0.8` 在我们池子上算什么，完全取决于
  是否合并重合顶点。抽样 300（objaverse）/ 157（toys4k）实测：合并后 0.51–0.52，几乎全过
  （>0.8 的只占 3.3% / 0%）；不合并（按文件原样，含法线/UV 缝的重复顶点）是 0.87–1.13，
  过半不过（56.7% / 75.0%）。要先确认他们的口径，再决定是否采用这条规则。
- **面数下限 500**：我们的池子里面数 < 500 的占 objaverse 13.7% / toys4k 16.6%（同上抽样）。
  本项目目前没有下限。
- **窄面占比 ≤ 20%**：按我们的 `small_angle_ratio`，超过 20% 的占 objaverse 42.3% / toys4k 26.8%。
  对照本项目实测（`sliver_ratio` 是逐对分开两个来源的那个量），这条与"细长面是坏的"方向一致。
- **Objaverse++ 的标注**：值得查我们的 3000 个负样本 uid 是否在其发布范围内——若在，可直接
  得到一批外部质量标注，用于检验本项目的代理标签。
