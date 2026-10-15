# Feature definitions

## Basic counts

- `vertex_count`：顶点数量；
- `face_count`：三角面数量；
- `edge_count`：无向唯一边数量；
- `component_count`：至少被一个面使用的顶点所形成的连通分量数量；
- `boundary_edge_count`：只被一个三角面使用的边数量；
- `nonmanifold_edge_count`：被三个或更多三角面使用的边数量。

## Face validity

- `zero_area_face_count`：面积小于等于 `zero_area_epsilon_ratio` 乘该网格中位面面积的面数
  （默认为中位面面积的 $10^{-6}$）。阈值取相对值而非绝对值，因为两个来源的尺度相差四个
  数量级（0.7 到 12160），一个固定的绝对 epsilon 在不同尺度上含义完全不同；
- `duplicate_face_count`：忽略顶点顺序后重复的额外三角面数量。

这些指标描述几何可用性，不直接定义 Artistic 标签。

## Triangle equilateral score

对边长为 $l_1,l_2,l_3$、面积为 $A$ 的三角形：

$$
q = \frac{4\sqrt{3}A}{l_1^2+l_2^2+l_3^2}
$$

- `equilateral_ratio`：$q>0.9$ 的面比例；
- `sliver_ratio`：$q<0.05$ 的面比例；
- `small_angle_ratio`：最小内角小于 $5^\circ$ 的面比例。

判读方向：**`equilateral_ratio` 高是坏的信号**。整套判据见 `docs/LABELING_GUIDE.md`——
课题组的口径是"面的形状不能太规则"，接近等边三角形正是自动重网格与扫描重建的特征，因为
那类流程追求均匀采样；手工低模的三角形是细长且有方向的。

`sliver_ratio` 单独看没有判别力：扫描网格里同样会出现细长面。它只有和方向类特征一起看才
有意义——细长 + 方向一致才是手工的痕迹。本项目早期文档曾把等边度写成"越接近 1 越好"，
那是错的，已更正。

## Area distribution

`area_coefficient_of_variation` 是非零面积三角面的总体标准差除以均值。它仅描述面积差异，必须结合结构和局部位置解释。

## Artistic features（已实现）

判据见 `docs/LABELING_GUIDE.md`，实现见 `src/mesh_filter/features/artistic.py`。
`configs/features/artistic_v1.yaml` 里的开关名就是它写进特征表的列名。

| 列 | 定义 | 期望方向 |
| --- | --- | --- |
| `vertex_degree_mode_share` | 顶点度数分布的众数占比（度 1–12 分箱） | 高 = 好（度数集中） |
| `vertex_degree_entropy` | 同一分布的归一化香农熵（除以 $\log 12$） | 低 = 好 |
| `one_ring_edge_length_cv_median` | 每个顶点一环边长变异系数的中位数 | 待实测 |
| `local_edge_direction_anisotropy_mean` | 每个顶点入射边方向张量 $\sum d_i d_i^{\top}$ 的 $(\lambda_3-\lambda_2)/\lambda_3$ 之均值 | 待实测 |
| `neighboring_direction_consistency_mean` | 同一顶点入射边两两 $|\cos|$ 的均值，再对所有顶点取均值 | 待实测 |
| `curvature_density_correlation` | 顶点离散曲率与局部面密度的 Spearman 相关 | 正 = 好 |
| `qem_redundancy_median` | 顶点塌缩到一环邻居质心时的二次误差中位数，除以包围盒对角线平方 | 高 = 好 |

标"待实测"的三列是按判据推出来的猜测，还没有数据支持。它们的正负号必须与 200 个人工盲审
标签对一遍才能定下来——表里写方向只是把猜测记下来，不是结论。已经实测否掉的一个例子：
`local_edge_direction_anisotropy_mean` 曾被设想为"局部有主导方向 = 好"，但八面体顶点
（方向成对对称）与平面网格分别给出 0.50 和 0.49——任何入射方向成对对称的顶点都会把这个
量推向 0，与网格是怎么做出来的无关。

## 共同约定

- 只在"在用顶点"上定义：在用顶点指至少属于一个三角面的顶点，孤立顶点不计入任何统计；
- 所有量对平移、旋转、统一缩放都不变。这条是硬要求：两个来源的尺度差四个数量级
  （Toys4K 最长边恒为 2.0，Objaverse 0.7–12160），任何带尺度的量都会被直接学成判别器；
- **测不出来时写空，不写 0**。`curvature_density_correlation` 在曲率或密度只有一个取值
  （平面、球面、规则网格）时无定义——那里的排名由浮点噪声决定，换个旋转就换个答案，
  留空表示"测不出来"，而不是"相关为零"；
- 网格过小或退化导致艺术特征算不出来时，`artistic_status` 记 `failed:<异常名>`、该行
  艺术特征列留空，但 `parse_status` 仍是 `parsed`：一个能解析的网格不该因为附加特征算
  不了就变成解析失败。

## QEM 的一处纠正

标准 QEM 误差在顶点自身求值恒等于零——顶点本来就在它所有邻接平面上。第一版实现就是这么
写的，实测对每种网格都返回 0，等于白算。现在求值点是**该顶点收缩到一环邻居质心后的位置**，
含义才真正是"删掉这个顶点会丢多少形状"。平面网格的邻居质心仍共面，读数接近 0（顶点冗余）；
闭合曲面的邻居质心偏离切平面，读数显著大于 0（顶点承担形状）。

## 与远端确认过的取舍

- $q$（等边度）在自动重网格与扫描重建上都很高，而这两类都不符合目标；它与 QEM 冗余度
  都不能单独当标签用，必须组合；
- 方向类特征依赖模型朝向。若远端不做统一朝向规范化，这类特征只应在同一资产内部比较，
  不能跨资产直接比较。

