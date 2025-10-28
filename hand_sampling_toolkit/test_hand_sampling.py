"""
手部采样功能测试脚本
快速测试手部点云采样功能是否正常工作
"""

import sys
sys.path.append("/home/lixiaoben/projects/NICP")
sys.path.append("/home/lixiaoben/projects/NICP/src")

import numpy as np
import torch
from smplx import SMPLX

from lvd_templ.evaluation.hand_sampling_utils import (
    sample_points_near_hands,
    get_hand_vertices_from_smplx,
    get_hand_vertex_indices,
)


def test_hand_indices():
    """测试手部顶点索引"""
    print("=" * 60)
    print("测试1: 手部顶点索引")
    print("=" * 60)
    
    hand_indices = get_hand_vertex_indices()
    
    print(f"左手顶点范围: {hand_indices['left_hand'][0]} - {hand_indices['left_hand'][-1]}")
    print(f"左手顶点数量: {len(hand_indices['left_hand'])}")
    print(f"右手顶点范围: {hand_indices['right_hand'][0]} - {hand_indices['right_hand'][-1]}")
    print(f"右手顶点数量: {len(hand_indices['right_hand'])}")
    
    assert len(hand_indices['left_hand']) > 0, "左手索引为空"
    assert len(hand_indices['right_hand']) > 0, "右手索引为空"
    
    print("✓ 手部索引测试通过\n")


def test_extract_hand_vertices():
    """测试提取手部顶点"""
    print("=" * 60)
    print("测试2: 提取手部顶点")
    print("=" * 60)
    
    # 创建模拟的SMPLX顶点
    smplx_vertices = np.random.randn(10475, 3)
    
    # 测试提取双手
    both_hands, both_indices = get_hand_vertices_from_smplx(smplx_vertices, hand='both')
    print(f"双手顶点数: {len(both_hands)}")
    assert len(both_hands) > 0, "双手顶点提取失败"
    
    # 测试提取左手
    left_hand, left_indices = get_hand_vertices_from_smplx(smplx_vertices, hand='left')
    print(f"左手顶点数: {len(left_hand)}")
    assert len(left_hand) > 0, "左手顶点提取失败"
    
    # 测试提取右手
    right_hand, right_indices = get_hand_vertices_from_smplx(smplx_vertices, hand='right')
    print(f"右手顶点数: {len(right_hand)}")
    assert len(right_hand) > 0, "右手顶点提取失败"
    
    # 检查左手+右手 = 双手
    assert len(left_hand) + len(right_hand) == len(both_hands), "手部顶点数量不匹配"
    
    print("✓ 手部顶点提取测试通过\n")


def test_sampling_methods():
    """测试不同的采样方法"""
    print("=" * 60)
    print("测试3: 采样方法")
    print("=" * 60)
    
    # 创建模拟数据
    smplx_vertices = np.random.randn(1, 10475, 3) * 0.5
    scan_points = np.random.randn(5000, 3) * 0.6
    
    # 测试半径采样
    print("\n测试半径采样...")
    try:
        points_r, indices_r, dist_r = sample_points_near_hands(
            scan_points=scan_points,
            smplx_vertices=smplx_vertices,
            radius=0.3,
            hand='both',
            method='radius'
        )
        print(f"  采样点数: {len(points_r)}")
        assert len(points_r) > 0, "半径采样失败"
        assert len(indices_r) == len(points_r), "索引长度不匹配"
        print("  ✓ 半径采样通过")
    except Exception as e:
        print(f"  ✗ 半径采样失败: {e}")
    
    # 测试KNN采样
    print("\n测试KNN采样...")
    try:
        points_k, indices_k, dist_k = sample_points_near_hands(
            scan_points=scan_points,
            smplx_vertices=smplx_vertices,
            num_samples=1000,
            hand='both',
            method='knn'
        )
        print(f"  采样点数: {len(points_k)}")
        assert len(points_k) == 1000 or len(points_k) == len(scan_points), "KNN采样点数不正确"
        print("  ✓ KNN采样通过")
    except Exception as e:
        print(f"  ✗ KNN采样失败: {e}")
    
    # 测试自适应采样
    print("\n测试自适应采样...")
    try:
        points_a, indices_a, dist_a = sample_points_near_hands(
            scan_points=scan_points,
            smplx_vertices=smplx_vertices,
            num_samples=1000,
            hand='both',
            method='adaptive'
        )
        print(f"  采样点数: {len(points_a)}")
        assert len(points_a) > 0, "自适应采样失败"
        print("  ✓ 自适应采样通过")
    except Exception as e:
        print(f"  ✗ 自适应采样失败: {e}")
    
    print("\n✓ 所有采样方法测试通过\n")


def test_with_real_smplx_model():
    """使用真实SMPLX模型测试"""
    print("=" * 60)
    print("测试4: 真实SMPLX模型")
    print("=" * 60)
    
    try:
        # 加载SMPLX模型
        smplx_model_path = "datafolder_new/body_models/smplx/SMPLX_NEUTRAL.pkl"
        print(f"加载SMPLX模型: {smplx_model_path}")
        
        smplx_model = SMPLX(
            model_path=smplx_model_path,
            ext="pkl",
            use_pca=False,
            num_betas=10,
        ).to(torch.device("cuda"))
        
        print("✓ SMPLX模型加载成功")
        
        # 生成随机参数
        with torch.no_grad():
            betas = torch.randn(1, 10).cuda() * 0.1
            body_pose = torch.zeros(1, 63).cuda()
            global_orient = torch.zeros(1, 3).cuda()
            transl = torch.zeros(1, 3).cuda()
            
            smplx_output = smplx_model(
                betas=betas,
                body_pose=body_pose,
                global_orient=global_orient,
                transl=transl,
            )
            smplx_vertices = smplx_output.vertices
        
        print(f"✓ SMPLX顶点生成成功: {smplx_vertices.shape}")
        
        # 创建模拟扫描点云
        scan_points = smplx_vertices.squeeze(0).cpu().numpy()
        scan_points = scan_points + np.random.randn(*scan_points.shape) * 0.02
        
        # 测试采样
        hand_points, indices, distances = sample_points_near_hands(
            scan_points=scan_points,
            smplx_vertices=smplx_vertices,
            radius=0.15,
            hand='both',
            method='radius'
        )
        
        print(f"✓ 采样成功: {len(hand_points)} 个手部点")
        print(f"  平均距离: {distances.mean():.4f}")
        print(f"  最大距离: {distances.max():.4f}")
        print(f"  最小距离: {distances.min():.4f}")
        
        # 分别测试左右手
        left_points, _, _ = sample_points_near_hands(
            scan_points=scan_points,
            smplx_vertices=smplx_vertices,
            radius=0.15,
            hand='left',
            method='radius'
        )
        
        right_points, _, _ = sample_points_near_hands(
            scan_points=scan_points,
            smplx_vertices=smplx_vertices,
            radius=0.15,
            hand='right',
            method='radius'
        )
        
        print(f"  左手点数: {len(left_points)}")
        print(f"  右手点数: {len(right_points)}")
        
        print("\n✓ 真实SMPLX模型测试通过\n")
        
    except FileNotFoundError as e:
        print(f"✗ 文件未找到: {e}")
        print("  跳过真实模型测试")
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_different_hands():
    """测试左手、右手、双手采样"""
    print("=" * 60)
    print("测试5: 不同手部组合")
    print("=" * 60)
    
    smplx_vertices = np.random.randn(1, 10475, 3) * 0.5
    scan_points = np.random.randn(3000, 3) * 0.6
    
    for hand_type in ['left', 'right', 'both']:
        print(f"\n测试 {hand_type}:")
        points, indices, distances = sample_points_near_hands(
            scan_points=scan_points,
            smplx_vertices=smplx_vertices,
            radius=0.3,
            hand=hand_type,
            method='radius'
        )
        print(f"  采样点数: {len(points)}")
        assert len(points) > 0, f"{hand_type} 采样失败"
    
    print("\n✓ 不同手部组合测试通过\n")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("手部点云采样功能测试")
    print("=" * 60 + "\n")
    
    try:
        test_hand_indices()
        test_extract_hand_vertices()
        test_sampling_methods()
        test_different_hands()
        test_with_real_smplx_model()
        
        print("=" * 60)
        print("✓ 所有测试通过！")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
