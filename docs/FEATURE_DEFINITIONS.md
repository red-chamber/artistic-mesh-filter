# Feature definitions

## Basic counts

- `vertex_count`：顶点数量；
- `face_count`：三角面数量；
- `edge_count`：无向唯一边数量；
- `component_count`：至少被一个面使用的顶点所形成的连通分量数量；
- `boundary_edge_count`：只被一个三角面使用的边数量；
- `nonmanifold_edge_count`：被三个或更多三角面使用的边数量。

## Face validity

- `zero_area_face_count`：面积小于等于 `zero_area_epsilon` 的面数；
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

不能简单认为 $q$ 越低越好：接近 1 可能表示各向同性重网格，接近 0 可能表示退化拓扑。

## Area distribution

`area_coefficient_of_variation` 是非零面积三角面的总体标准差除以均值。它仅描述面积差异，必须结合结构和局部位置解释。

## Draft Artistic features

顶点度数、一环边长变异、局部方向熵、方向连续性、曲率密度相关性和 QEM 冗余度尚未在真实数据上验证。它们已经进入配置草案，但在确认真实拓扑和格式之前不作为稳定接口。
