# SMPLX手部点云采样指南

## 概述

本指南介绍如何根据SMPLX人体模型的手部顶点位置，从扫描点云中采样手部附近的点。这在需要对手部区域进行精细化处理或分析时非常有用。

## 功能特性

- ✅ 支持从点云中采样左手、右手或双手附近的点
- ✅ 提供三种采样方法：半径采样、KNN采样、自适应采样
- ✅ 自动处理torch.Tensor和numpy数组
- ✅ 返回采样点、索引和距离信息
- ✅ 可选的可视化功能

## 安装依赖

```bash
# 基础依赖（项目中已有）
pip install numpy scipy torch trimesh

# 可选：用于可视化
pip install open3d
```

## 快速开始

### 1. 导入模块

```python
from lvd_templ.evaluation.hand_sampling_utils import (
    sample_points_near_hands,
    get_hand_vertices_from_smplx,
    visualize_hand_sampling
)
```

### 2. 基础使用

```python
# 假设你已经有：
# - scan_points: 扫描点云 (N, 3) numpy array
# - smplx_vertices: SMPLX顶点 (10475, 3) 或 (1, 10475, 3)

# 采样双手附近的点
hand_points, indices, distances = sample_points_near_hands(
    scan_points=scan_points,
    smplx_vertices=smplx_vertices,
    radius=0.15,  # 15cm半径
    hand='both',
    method='radius'
)

print(f"采样得到 {len(hand_points)} 个手部点")
```

## 采样方法详解

### 方法1: 半径采样 (推荐用于保留所有手部点)

```python
hand_points, indices, distances = sample_points_near_hands(
    scan_points=scan_points,
    smplx_vertices=smplx_vertices,
    radius=0.15,  # 半径范围（米）
    hand='both',
    method='radius'
)
```

**优点：**
- 保留指定半径内的所有点
- 适合手部完整的扫描数据
- 不会遗漏手部区域的任何点

**缺点：**
- 采样数量不固定
- 可能包含过多或过少的点

**参数调整建议：**
- `radius=0.10`: 紧密贴合手部表面
- `radius=0.15`: 标准范围（推荐）
- `radius=0.20`: 包含手部周围更大区域

### 方法2: KNN采样 (推荐用于固定数量)

```python
hand_points, indices, distances = sample_points_near_hands(
    scan_points=scan_points,
    smplx_vertices=smplx_vertices,
    num_samples=2000,  # 固定采样2000个点
    hand='both',
    method='knn'
)
```

**优点：**
- 采样数量固定，便于后续处理
- 自动选择距离手部最近的点
- 适合需要统一点数的场景

**缺点：**
- 可能包含非手部的点（如果手部点不足）
- 不考虑点的空间分布

**参数调整建议：**
- `num_samples=1000`: 稀疏采样
- `num_samples=2000`: 标准采样（推荐）
- `num_samples=5000`: 密集采样

### 方法3: 自适应采样 (推荐用于均匀分布)

```python
hand_points, indices, distances = sample_points_near_hands(
    scan_points=scan_points,
    smplx_vertices=smplx_vertices,
    num_samples=2000,
    hand='both',
    method='adaptive'
)
```

**优点：**
- 在手部区域均匀分布
- 每个手部顶点都有对应的采样点
- 适合需要完整覆盖手部的场景

**缺点：**
- 实际采样数可能与num_samples略有差异
- 计算稍慢

## 采样单手

```python
# 只采样左手
left_hand_points, _, _ = sample_points_near_hands(
    scan_points=scan_points,
    smplx_vertices=smplx_vertices,
    radius=0.15,
    hand='left',
    method='radius'
)

# 只采样右手
right_hand_points, _, _ = sample_points_near_hands(
    scan_points=scan_points,
    smplx_vertices=smplx_vertices,
    radius=0.15,
    hand='right',
    method='radius'
)
```

## SMPLX手部顶点索引

SMPLX模型有10475个顶点，手部顶点索引：

- **左手顶点**: 索引 5361-5777 (共417个顶点)
- **右手顶点**: 索引 8079-8495 (共417个顶点)

```python
from lvd_templ.evaluation.hand_sampling_utils import get_hand_vertex_indices

# 获取手部顶点索引
hand_indices = get_hand_vertex_indices()
print(f"左手顶点数: {len(hand_indices['left_hand'])}")
print(f"右手顶点数: {len(hand_indices['right_hand'])}")
```

## 在evaluation_benchmark_cape_x.py中集成

在你的评估代码中添加手部采样功能：

```python
# 1. 在文件开头导入
from lvd_templ.evaluation.hand_sampling_utils import sample_points_near_hands

# 2. 在fit_smplx之后添加（约第360行）
out_s, params = fit_smplx(smplx_model, reg_src, gt_idxs)

# 获取SMPLX完整顶点（用于手部采样）
with torch.no_grad():
    smplx_output = smplx_model(
        betas=params['beta'],
        body_pose=params['pose'][:, 3:66],
        global_orient=params['pose'][:, :3],
        transl=params['trans'],
        jaw_pose=params['pose'][:, 66:69],
        leye_pose=params['pose'][:, 69:72],
        reye_pose=params['pose'][:, 72:75],
        left_hand_pose=params['pose'][:, 75:120],
        right_hand_pose=params['pose'][:, 120:165],
        expression=params.get('expression', torch.zeros(1, 10).cuda()),
    )
    smplx_vertices = smplx_output.vertices

# 从扫描点云采样手部点
hand_points, hand_indices, hand_distances = sample_points_near_hands(
    scan_points=mesh_src.vertices,  # 使用变换后的扫描点云
    smplx_vertices=smplx_vertices,
    radius=0.15,
    hand='both',
    method='radius'
)

print(f"手部采样: {len(hand_points)}/{len(mesh_src.vertices)} 点")

# 3. 保存采样结果
np.savez(
    out_dir + "/vis/" + name + "/hand_sampled.npz",
    hand_points=hand_points,
    hand_indices=hand_indices,
    distances=hand_distances
)

# 4. 可选：保存为点云文件用于可视化
import trimesh
hand_pcd = trimesh.PointCloud(hand_points)
hand_pcd.export(out_dir + "/vis/" + name + "/hand_sampled.ply")
```

## 完整示例

```python
import numpy as np
import torch
from smplx import SMPLX
from lvd_templ.evaluation.hand_sampling_utils import sample_points_near_hands

# 1. 加载扫描数据
scan_data = np.load("path/to/scan.npz")
scan_points = scan_data['hitpts']  # 或 'pred_inner_points'

# 2. 加载SMPLX模型
smplx_model = SMPLX(
    model_path="datafolder_new/body_models/smplx/SMPLX_NEUTRAL.pkl",
    ext="pkl",
    use_pca=False,
    num_betas=10,
).cuda()

# 3. 加载SMPLX参数（假设已经fit好）
smplx_params = np.load("path/to/smplx_params.npz")

with torch.no_grad():
    smplx_output = smplx_model(
        betas=torch.from_numpy(smplx_params['betas']).float().cuda().unsqueeze(0),
        body_pose=torch.from_numpy(smplx_params['pose'][:63]).float().cuda().reshape(1, -1),
        global_orient=torch.from_numpy(smplx_params['global_orient']).float().cuda().reshape(1, 3),
        transl=torch.from_numpy(smplx_params['transl']).float().cuda().unsqueeze(0),
    )
    smplx_vertices = smplx_output.vertices

# 4. 采样手部点云
hand_points, indices, distances = sample_points_near_hands(
    scan_points=scan_points,
    smplx_vertices=smplx_vertices,
    radius=0.15,
    hand='both',
    method='radius'
)

# 5. 使用采样结果
print(f"原始点云: {len(scan_points)} 点")
print(f"手部点云: {len(hand_points)} 点")
print(f"平均距离: {distances.mean():.4f} 米")

# 6. 保存结果
np.save("hand_points.npy", hand_points)
```

## 可视化

```python
from lvd_templ.evaluation.hand_sampling_utils import visualize_hand_sampling

# 可视化采样结果
visualize_hand_sampling(
    scan_points=scan_points,
    smplx_vertices=smplx_vertices,
    sampled_points=hand_points,
    hand='both',
    save_path="hand_sampled_visualization.ply"
)
```

颜色说明：
- 灰色：原始扫描点云
- 红色：采样的手部点
- 绿色：SMPLX手部顶点

## 常见问题

### Q1: 采样的点太少怎么办？

**A**: 增加采样半径或使用KNN方法指定点数：
```python
# 方案1：增加半径
hand_points, _, _ = sample_points_near_hands(..., radius=0.20, ...)

# 方案2：使用KNN
hand_points, _, _ = sample_points_near_hands(..., num_samples=3000, method='knn')
```

### Q2: 采样的点太多怎么办？

**A**: 减小半径或使用KNN限制点数：
```python
# 方案1：减小半径
hand_points, _, _ = sample_points_near_hands(..., radius=0.10, ...)

# 方案2：使用KNN
hand_points, _, _ = sample_points_near_hands(..., num_samples=1000, method='knn')
```

### Q3: 如何处理部分遮挡的手部？

**A**: 使用自适应采样，它能更好地处理不完整的数据：
```python
hand_points, _, _ = sample_points_near_hands(
    ..., 
    num_samples=2000,
    method='adaptive'
)
```

### Q4: 返回的索引有什么用？

**A**: 索引可以用于：
- 反向查找原始点云中的对应点
- 提取原始点云的其他属性（如颜色、法向量）
- 创建mask用于后续处理

```python
# 使用索引提取颜色（如果有）
if hasattr(scan_src, 'colors'):
    hand_colors = scan_src.colors[hand_indices]
```

### Q5: 坐标系不匹配怎么办？

**A**: 确保扫描点云和SMPLX顶点在同一坐标系：
```python
# 如果需要应用变换
from trimesh import transformations

# 将SMPLX顶点变换到扫描坐标系
smplx_verts_transformed = transformations.transform_points(
    smplx_vertices.cpu().numpy().squeeze(),
    transformation_matrix
)

hand_points, _, _ = sample_points_near_hands(
    scan_points=scan_points,
    smplx_vertices=smplx_verts_transformed,
    ...
)
```

## 性能优化建议

1. **大规模点云**: 如果扫描点云非常大（>100k点），建议先进行粗采样
```python
# 先粗采样减少点数
step = 2
scan_points_coarse = scan_points[::step]
```

2. **批处理**: 处理多个扫描时，复用SMPLX模型
```python
smplx_model = SMPLX(...).cuda()  # 只加载一次
for scan in scans:
    # 处理每个扫描...
```

3. **GPU加速**: 对于大规模计算，考虑使用GPU加速的KDTree实现

## 参考资料

- SMPLX模型: https://smpl-x.is.tue.mpg.de/
- 项目文档: `/home/lixiaoben/projects/NICP/README.md`
- 示例代码: `/home/lixiaoben/projects/NICP/examples/example_hand_sampling.py`

## 联系方式

如有问题，请查看：
- 代码文件: `src/lvd_templ/evaluation/hand_sampling_utils.py`
- 示例代码: `examples/example_hand_sampling.py`
