#!/usr/bin/env python3
"""
快速测试脚本 - 验证手部采样工具是否正常工作

运行方式:
    python quick_test.py
或
    chmod +x quick_test.py
    ./quick_test.py
"""

import sys
import os

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 添加NICP项目路径
nicp_path = os.path.dirname(current_dir)
sys.path.insert(0, nicp_path)
sys.path.insert(0, os.path.join(nicp_path, "src"))

import numpy as np

print("=" * 70)
print("手部采样工具快速测试")
print("=" * 70)

# 测试1: 导入模块
print("\n[1/5] 测试导入模块...")
try:
    from hand_sampling_utils import (
        sample_points_near_hands,
        get_hand_vertices_from_smplx,
        get_hand_vertex_indices,
    )
    print("✓ 模块导入成功")
except ImportError as e:
    print(f"✗ 模块导入失败: {e}")
    sys.exit(1)

# 测试2: 手部索引
print("\n[2/5] 测试手部顶点索引...")
try:
    hand_indices = get_hand_vertex_indices()
    print(f"✓ 左手顶点: {len(hand_indices['left_hand'])} 个")
    print(f"✓ 右手顶点: {len(hand_indices['right_hand'])} 个")
except Exception as e:
    print(f"✗ 测试失败: {e}")
    sys.exit(1)

# 测试3: 提取手部顶点
print("\n[3/5] 测试提取手部顶点...")
try:
    smplx_vertices = np.random.randn(10475, 3)
    hand_verts, hand_idx = get_hand_vertices_from_smplx(smplx_vertices, hand='both')
    print(f"✓ 提取双手顶点: {len(hand_verts)} 个")
except Exception as e:
    print(f"✗ 测试失败: {e}")
    sys.exit(1)

# 测试4: 半径采样
print("\n[4/5] 测试半径采样...")
try:
    scan_points = np.random.randn(5000, 3) * 0.6
    smplx_vertices = np.random.randn(1, 10475, 3) * 0.5
    
    hand_points, indices, distances = sample_points_near_hands(
        scan_points=scan_points,
        smplx_vertices=smplx_vertices,
        radius=0.3,
        hand='both',
        method='radius'
    )
    print(f"✓ 半径采样成功: {len(hand_points)} 个点")
except Exception as e:
    print(f"✗ 测试失败: {e}")
    sys.exit(1)

# 测试5: KNN采样
print("\n[5/5] 测试KNN采样...")
try:
    hand_points, indices, distances = sample_points_near_hands(
        scan_points=scan_points,
        smplx_vertices=smplx_vertices,
        num_samples=1000,
        hand='both',
        method='knn'
    )
    print(f"✓ KNN采样成功: {len(hand_points)} 个点")
except Exception as e:
    print(f"✗ 测试失败: {e}")
    sys.exit(1)

# 全部通过
print("\n" + "=" * 70)
print("✓ 所有测试通过！手部采样工具运行正常")
print("=" * 70)
print("\n下一步:")
print("  1. 运行完整测试: python test_hand_sampling.py")
print("  2. 查看示例代码: python example_hand_sampling.py")
print("  3. 阅读详细文档: cat hand_sampling_guide.md")
print()
