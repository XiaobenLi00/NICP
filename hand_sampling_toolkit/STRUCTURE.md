# 手部采样工具包 - 文件结构

```
hand_sampling_toolkit/                 # 工具包根目录
│
├── README.md                          # 📖 工具包总览和快速开始指南 (7.4KB)
│   ├── 功能介绍
│   ├── 快速开始示例
│   ├── 文件说明
│   ├── 使用场景
│   └── 常见问题
│
├── __init__.py                        # 📦 Python包初始化文件 (1.1KB)
│   └── 导出主要函数，支持作为包导入
│
├── hand_sampling_utils.py             # 🔧 核心工具模块 (11KB)
│   ├── get_hand_vertex_indices()      # 获取SMPLX手部顶点索引
│   ├── get_hand_vertices_from_smplx() # 从SMPLX提取手部顶点
│   ├── sample_points_near_hands()     # ⭐ 主要采样函数
│   ├── visualize_hand_sampling()      # 可视化采样结果
│   └── get_hand_region_mask()         # 获取手部区域mask
│
├── example_hand_sampling.py           # 💡 使用示例代码 (11KB)
│   ├── example_basic_usage()          # 示例1: 基础用法
│   ├── example_separate_hands()       # 示例2: 分别采样左右手
│   ├── example_with_real_data()       # 示例3: 使用实际数据
│   └── example_integration_in_evaluation() # 示例4: 集成到评估流程
│
├── test_hand_sampling.py              # 🧪 功能测试脚本 (8.5KB)
│   ├── test_hand_indices()            # 测试1: 手部索引
│   ├── test_extract_hand_vertices()   # 测试2: 提取顶点
│   ├── test_sampling_methods()        # 测试3: 三种采样方法
│   ├── test_with_real_smplx_model()   # 测试4: 真实SMPLX模型
│   └── test_different_hands()         # 测试5: 左右手组合
│
├── quick_test.py                      # ⚡ 快速测试脚本 (2.8KB)
│   └── 5个快速测试验证工具是否正常工作
│
└── hand_sampling_guide.md             # 📚 详细使用指南 (9.8KB)
    ├── 功能特性和安装
    ├── 三种采样方法详解
    ├── 完整API文档
    ├── 使用场景和示例
    ├── 常见问题解答
    └── 性能优化建议
```

## 文件用途速查

| 文件 | 大小 | 用途 | 何时使用 |
|------|------|------|---------|
| `README.md` | 7.4KB | 入口说明 | 第一次使用时阅读 |
| `__init__.py` | 1.1KB | 包初始化 | 作为包导入时自动使用 |
| `hand_sampling_utils.py` | 11KB | 核心功能 | **在你的代码中导入使用** ⭐ |
| `example_hand_sampling.py` | 11KB | 示例代码 | 学习如何使用 |
| `test_hand_sampling.py` | 8.5KB | 完整测试 | 验证所有功能 |
| `quick_test.py` | 2.8KB | 快速测试 | **快速验证是否正常** ⭐ |
| `hand_sampling_guide.md` | 9.8KB | 详细文档 | 深入学习和参考 |

## 快速导航

### 🚀 我想马上开始使用
```bash
# 1. 快速测试
cd /home/lixiaoben/projects/NICP/hand_sampling_toolkit
python quick_test.py

# 2. 查看README
cat README.md
```

### 💻 我想看代码示例
```bash
# 查看或运行示例
python example_hand_sampling.py
```

### 🔍 我想了解所有细节
```bash
# 阅读完整文档
cat hand_sampling_guide.md
# 或用编辑器打开
code hand_sampling_guide.md
```

### 🧪 我想测试所有功能
```bash
# 运行完整测试
python test_hand_sampling.py
```

### 📦 我想在我的代码中使用
```python
# 方法1: 直接导入（推荐）
import sys
sys.path.append('/home/lixiaoben/projects/NICP/hand_sampling_toolkit')
from hand_sampling_utils import sample_points_near_hands

# 方法2: 作为包导入
import sys
sys.path.append('/home/lixiaoben/projects/NICP')
from hand_sampling_toolkit import sample_points_near_hands

# 方法3: 使用原位置的模块
from lvd_templ.evaluation.hand_sampling_utils import sample_points_near_hands
```

## 核心功能一览

### 主函数: `sample_points_near_hands()`

```python
def sample_points_near_hands(
    scan_points,        # 扫描点云 (N, 3)
    smplx_vertices,     # SMPLX顶点 (10475, 3)
    radius=0.1,         # 采样半径
    num_samples=None,   # 采样点数
    hand='both',        # 'left', 'right', 'both'
    method='radius'     # 'radius', 'knn', 'adaptive'
)
```

**返回值:**
- `sampled_points`: 采样得到的点 (M, 3)
- `sampled_indices`: 采样点在原始点云中的索引
- `distances`: 采样点到最近手部顶点的距离

### 三种采样方法

| 方法 | 参数 | 适用场景 |
|------|------|---------|
| `radius` | `radius=0.15` | 手部完整，需要所有手部点 |
| `knn` | `num_samples=2000` | 需要固定数量的点 |
| `adaptive` | `num_samples=2000` | 需要均匀分布的点 |

## 总文件大小

```
总计: 51.5 KB (7个文件)
  - 代码文件: 33.4 KB (4个.py文件)
  - 文档文件: 17.2 KB (2个.md文件)
  - 包文件:   1.1 KB  (1个__init__.py)
```

## 版本信息

- **版本**: 1.0.0
- **创建日期**: 2025-10-25
- **项目**: NICP (Neural Implicit Correspondences)
- **位置**: `/home/lixiaoben/projects/NICP/hand_sampling_toolkit/`

---

**提示**: 所有文件也保留在原始位置以保持项目结构完整性
- `src/lvd_templ/evaluation/hand_sampling_utils.py`
- `examples/example_hand_sampling.py`
- `tests/test_hand_sampling.py`
- `docs/hand_sampling_guide.md`
