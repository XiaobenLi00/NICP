# 手部点云采样工具包 (Hand Sampling Toolkit)

> 根据SMPLX人体模型的手部顶点位置，从扫描点云中采样手部附近的点

## 📁 目录结构

```
hand_sampling_toolkit/
├── README.md                      # 本文件 - 工具包说明
├── hand_sampling_utils.py         # 核心工具模块
├── example_hand_sampling.py       # 使用示例代码
├── test_hand_sampling.py          # 功能测试脚本
└── hand_sampling_guide.md         # 详细使用指南
```

## 🚀 快速开始

### 1. 最简单的使用

```python
from hand_sampling_utils import sample_points_near_hands

# 从扫描点云中采样手部附近的点
hand_points, indices, distances = sample_points_near_hands(
    scan_points=scan_points,        # 扫描点云 (N, 3)
    smplx_vertices=smplx_vertices,  # SMPLX顶点 (10475, 3)
    radius=0.15,                    # 15cm半径
    hand='both',                    # 双手
    method='radius'                 # 半径采样
)

print(f"采样得到 {len(hand_points)} 个手部点")
```

### 2. 运行示例代码

```bash
python example_hand_sampling.py
```

### 3. 运行测试

```bash
python test_hand_sampling.py
```

## 📚 文件说明

### 1️⃣ `hand_sampling_utils.py` - 核心工具模块

**主要函数**:

- `sample_points_near_hands()` - 主要采样函数，支持三种方法
- `get_hand_vertices_from_smplx()` - 从SMPLX顶点提取手部顶点
- `get_hand_vertex_indices()` - 获取手部顶点索引
- `visualize_hand_sampling()` - 可视化采样结果
- `get_hand_region_mask()` - 获取手部区域mask

**导入方式**:
```python
from hand_sampling_utils import (
    sample_points_near_hands,
    get_hand_vertices_from_smplx,
    visualize_hand_sampling
)
```

### 2️⃣ `example_hand_sampling.py` - 使用示例

包含四个完整示例：
- **示例1**: 基础手部点云采样（三种方法对比）
- **示例2**: 分别采样左右手
- **示例3**: 使用实际扫描数据
- **示例4**: 集成到评估流程的代码示例

### 3️⃣ `test_hand_sampling.py` - 功能测试

包含五个测试：
- 测试1: 手部顶点索引
- 测试2: 提取手部顶点
- 测试3: 三种采样方法
- 测试4: 真实SMPLX模型
- 测试5: 不同手部组合

### 4️⃣ `hand_sampling_guide.md` - 详细指南

完整的使用文档，包含：
- 功能特性和安装说明
- 三种采样方法详解和对比
- 完整的API文档
- 常见问题解答
- 性能优化建议

## 🎯 三种采样方法

| 方法 | 使用场景 | 优点 | 推荐参数 |
|------|---------|------|---------|
| **radius** | 手部完整 | 保留所有手部点 | `radius=0.15` |
| **knn** | 固定点数 | 点数可控 | `num_samples=2000` |
| **adaptive** | 均匀覆盖 | 空间分布均匀 | `num_samples=2000` |

## 💡 使用场景

### 场景1: 手部精细化优化

```python
# 1. 采样手部点云
hand_points, indices, _ = sample_points_near_hands(
    scan_points, smplx_vertices, radius=0.12, hand='both', method='radius'
)

# 2. 对手部进行精细优化
# 可以用于手部姿态refinement或者手部细节重建
```

### 场景2: 手部区域分析

```python
# 分别采样左右手
left_hand_points, _, _ = sample_points_near_hands(
    scan_points, smplx_vertices, radius=0.15, hand='left', method='radius'
)
right_hand_points, _, _ = sample_points_near_hands(
    scan_points, smplx_vertices, radius=0.15, hand='right', method='radius'
)

# 分析左右手的点云质量
print(f"左手点数: {len(left_hand_points)}")
print(f"右手点数: {len(right_hand_points)}")
```

### 场景3: 集成到NICP评估流程

```python
# 在evaluation_benchmark_cape_x.py中
from hand_sampling_utils import sample_points_near_hands

# fit_smplx后
out_s, params = fit_smplx(smplx_model, reg_src, gt_idxs)

# 采样手部点云用于后续分析
hand_points, _, _ = sample_points_near_hands(
    scan_points=mesh_src.vertices,
    smplx_vertices=smplx_output.vertices,
    radius=0.15,
    hand='both',
    method='radius'
)

# 保存结果
np.save(out_dir + f"/vis/{name}/hand_points.npy", hand_points)
```

## 🔧 SMPLX手部信息

- **总顶点数**: 10,475
- **左手顶点**: 索引 5361-5777 (417个)
- **右手顶点**: 索引 8079-8495 (417个)
- **双手总计**: 834个顶点

## 📋 依赖项

```bash
# 基础依赖
numpy
scipy
torch
trimesh

# 可选依赖（用于可视化）
open3d
```

## 🛠️ 安装

工具包已经包含在NICP项目中，无需额外安装。

如果需要单独使用，确保安装依赖：

```bash
pip install numpy scipy torch trimesh
pip install open3d  # 可选，用于可视化
```

## 📖 详细文档

查看 `hand_sampling_guide.md` 获取：
- 完整的API文档
- 详细的参数说明
- 使用技巧和最佳实践
- 常见问题解答
- 性能优化建议

## 🧪 测试验证

运行测试脚本验证所有功能：

```bash
python test_hand_sampling.py
```

预期输出：
```
============================================================
手部点云采样功能测试
============================================================

============================================================
测试1: 手部顶点索引
============================================================
左手顶点范围: 5361 - 5777
左手顶点数量: 417
右手顶点范围: 8079 - 8495
右手顶点数量: 417
✓ 手部索引测试通过

... (更多测试输出)

============================================================
✓ 所有测试通过！
============================================================
```

## 🔗 集成示例

### 在项目中使用

**方法1**: 直接使用（推荐）

```python
# 添加工具包路径
import sys
sys.path.append('/home/lixiaoben/projects/NICP/hand_sampling_toolkit')

from hand_sampling_utils import sample_points_near_hands
```

**方法2**: 复制到项目源码

```bash
# 复制到你的evaluation目录
cp hand_sampling_utils.py /path/to/your/project/
```

**方法3**: 使用原位置的模块

```python
# 核心模块也在原位置
from lvd_templ.evaluation.hand_sampling_utils import sample_points_near_hands
```

## ❓ 快速问答

**Q: 采样点太少怎么办？**
```python
# 增加半径
sample_points_near_hands(..., radius=0.20, ...)
# 或使用KNN指定点数
sample_points_near_hands(..., num_samples=3000, method='knn')
```

**Q: 采样点太多怎么办？**
```python
# 减小半径
sample_points_near_hands(..., radius=0.10, ...)
# 或使用KNN限制点数
sample_points_near_hands(..., num_samples=1000, method='knn')
```

**Q: 如何只采样一只手？**
```python
# 左手
sample_points_near_hands(..., hand='left', ...)
# 右手
sample_points_near_hands(..., hand='right', ...)
```

**Q: 返回的索引有什么用？**
```python
# 用于提取原始点云的其他属性
hand_colors = scan_colors[indices]
hand_normals = scan_normals[indices]
```

## 📞 支持

如有问题或建议：
1. 查看 `hand_sampling_guide.md` 详细文档
2. 运行 `test_hand_sampling.py` 验证功能
3. 参考 `example_hand_sampling.py` 中的示例

## 📄 文件清单

- ✅ `hand_sampling_utils.py` - 核心工具 (300+ 行)
- ✅ `example_hand_sampling.py` - 示例代码 (250+ 行)
- ✅ `test_hand_sampling.py` - 测试脚本 (200+ 行)
- ✅ `hand_sampling_guide.md` - 详细指南 (500+ 行)
- ✅ `README.md` - 本说明文件

---

**创建日期**: 2025-10-25  
**版本**: 1.0  
**项目**: NICP - Neural Implicit Correspondences  
**用途**: SMPLX手部点云采样工具
