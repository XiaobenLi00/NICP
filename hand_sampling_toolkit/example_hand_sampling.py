"""
手部点云采样示例
演示如何根据SMPLX手部顶点从扫描点云中采样手部附近的点
"""

import sys
sys.path.append("/home/lixiaoben/projects/NICP")
sys.path.append("/home/lixiaoben/projects/NICP/src")

import numpy as np
import torch
import trimesh
from smplx import SMPLX

from lvd_templ.evaluation.hand_sampling_utils import (
    sample_points_near_hands,
    get_hand_vertices_from_smplx,
    visualize_hand_sampling,
    get_hand_vertex_indices
)


def example_basic_usage():
    """
    基础使用示例：从点云中采样手部附近的点
    """
    print("=" * 60)
    print("示例1: 基础手部点云采样")
    print("=" * 60)
    
    # 1. 加载SMPLX模型
    smplx_model_path = "datafolder_new/body_models/smplx/SMPLX_NEUTRAL.pkl"
    smplx_model = SMPLX(
        model_path=smplx_model_path,
        ext="pkl",
        use_pca=False,
        num_betas=10,
    ).to(torch.device("cuda"))
    
    # 2. 假设你已经有了fit好的SMPLX参数
    # 这里用随机参数作为示例
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
        smplx_vertices = smplx_output.vertices  # (1, 10475, 3)
    
    # 3. 生成模拟的扫描点云（实际使用中替换为真实扫描数据）
    scan_points = smplx_vertices.squeeze(0).cpu().numpy()
    # 添加一些噪声模拟实际扫描
    scan_points = scan_points + np.random.randn(*scan_points.shape) * 0.01
    
    print(f"扫描点云形状: {scan_points.shape}")
    print(f"SMPLX顶点形状: {smplx_vertices.shape}")
    
    # 4. 方法1: 使用半径采样
    print("\n--- 方法1: 半径采样 ---")
    sampled_points_r, indices_r, distances_r = sample_points_near_hands(
        scan_points=scan_points,
        smplx_vertices=smplx_vertices,
        radius=0.15,  # 15cm半径
        hand='both',
        method='radius'
    )
    print(f"半径采样结果: {sampled_points_r.shape}")
    print(f"平均距离: {distances_r.mean():.4f}")
    
    # 5. 方法2: 使用KNN采样
    print("\n--- 方法2: KNN采样 ---")
    sampled_points_k, indices_k, distances_k = sample_points_near_hands(
        scan_points=scan_points,
        smplx_vertices=smplx_vertices,
        num_samples=2000,
        hand='both',
        method='knn'
    )
    print(f"KNN采样结果: {sampled_points_k.shape}")
    print(f"平均距离: {distances_k.mean():.4f}")
    
    # 6. 方法3: 自适应采样
    print("\n--- 方法3: 自适应采样 ---")
    sampled_points_a, indices_a, distances_a = sample_points_near_hands(
        scan_points=scan_points,
        smplx_vertices=smplx_vertices,
        num_samples=2000,
        hand='both',
        method='adaptive'
    )
    print(f"自适应采样结果: {sampled_points_a.shape}")
    print(f"平均距离: {distances_a.mean():.4f}")
    
    return sampled_points_r, smplx_vertices, scan_points


def example_separate_hands():
    """
    示例2: 分别采样左右手
    """
    print("\n" + "=" * 60)
    print("示例2: 分别采样左右手")
    print("=" * 60)
    
    # 加载SMPLX模型
    smplx_model_path = "datafolder_new/body_models/smplx/SMPLX_NEUTRAL.pkl"
    smplx_model = SMPLX(
        model_path=smplx_model_path,
        ext="pkl",
        use_pca=False,
        num_betas=10,
    ).to(torch.device("cuda"))
    
    # 生成SMPLX顶点
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
    
    # 生成模拟扫描点云
    scan_points = smplx_vertices.squeeze(0).cpu().numpy()
    scan_points = scan_points + np.random.randn(*scan_points.shape) * 0.01
    
    # 分别采样左手和右手
    print("\n采样左手附近的点:")
    left_hand_points, _, _ = sample_points_near_hands(
        scan_points=scan_points,
        smplx_vertices=smplx_vertices,
        radius=0.12,
        hand='left',
        method='radius'
    )
    
    print("\n采样右手附近的点:")
    right_hand_points, _, _ = sample_points_near_hands(
        scan_points=scan_points,
        smplx_vertices=smplx_vertices,
        radius=0.12,
        hand='right',
        method='radius'
    )
    
    print(f"\n左手点云: {left_hand_points.shape}")
    print(f"右手点云: {right_hand_points.shape}")
    
    return left_hand_points, right_hand_points


def example_with_real_data(npz_file_path):
    """
    示例3: 使用实际的扫描数据
    
    Args:
        npz_file_path: .npz文件路径，包含扫描点云数据
    """
    print("\n" + "=" * 60)
    print("示例3: 使用实际扫描数据")
    print("=" * 60)
    
    # 1. 加载扫描数据
    data = np.load(npz_file_path)
    
    # 根据你的数据格式，选择合适的键
    # 可能的键名: 'hitpts', 'pred_inner_points', 'points', 'vertices' 等
    if 'hitpts' in data:
        scan_points = data['hitpts']
    elif 'pred_inner_points' in data:
        scan_points = data['pred_inner_points']
    else:
        # 列出所有可用的键
        print(f"可用的数据键: {list(data.keys())}")
        return None
    
    print(f"加载的扫描点云形状: {scan_points.shape}")
    
    # 2. 加载SMPLX参数（如果有保存的话）
    smplx_info_path = npz_file_path.replace('.npz', '_smplx_info.npz')
    try:
        smplx_info = np.load(smplx_info_path)
        print(f"找到SMPLX参数文件: {smplx_info_path}")
    except FileNotFoundError:
        print(f"未找到SMPLX参数文件，使用默认参数")
        smplx_info = None
    
    # 3. 重建SMPLX模型顶点
    smplx_model_path = "datafolder_new/body_models/smplx/SMPLX_NEUTRAL.pkl"
    smplx_model = SMPLX(
        model_path=smplx_model_path,
        ext="pkl",
        use_pca=False,
        num_betas=10,
    ).to(torch.device("cuda"))
    
    with torch.no_grad():
        if smplx_info is not None:
            # 使用保存的参数
            betas = torch.from_numpy(smplx_info['betas']).float().cuda().unsqueeze(0)
            body_pose = torch.from_numpy(smplx_info['pose']).float().cuda().reshape(1, -1)
            global_orient = torch.from_numpy(smplx_info['global_orient']).float().cuda().reshape(1, 3)
            transl = torch.from_numpy(smplx_info['transl']).float().cuda().unsqueeze(0)
        else:
            # 使用默认参数
            betas = torch.zeros(1, 10).cuda()
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
    
    # 4. 采样手部点云
    sampled_points, indices, distances = sample_points_near_hands(
        scan_points=scan_points,
        smplx_vertices=smplx_vertices,
        radius=0.15,  # 可以根据需要调整
        hand='both',
        method='radius'
    )
    
    print(f"\n采样结果:")
    print(f"  原始点云: {scan_points.shape}")
    print(f"  手部点云: {sampled_points.shape}")
    print(f"  采样率: {len(sampled_points) / len(scan_points) * 100:.2f}%")
    
    # 5. 保存采样结果
    output_path = npz_file_path.replace('.npz', '_hand_sampled.npz')
    np.savez(
        output_path,
        hand_points=sampled_points,
        hand_indices=indices,
        distances=distances
    )
    print(f"\n采样结果已保存到: {output_path}")
    
    return sampled_points, indices


def example_integration_in_evaluation():
    """
    示例4: 在evaluation_benchmark_cape_x.py中集成手部采样
    
    这个示例展示如何在你的评估代码中使用手部采样功能
    """
    print("\n" + "=" * 60)
    print("示例4: 集成到评估流程")
    print("=" * 60)
    
    code_example = """
# 在evaluation_benchmark_cape_x.py中的使用示例：

# 1. 在文件开头导入
from lvd_templ.evaluation.hand_sampling_utils import sample_points_near_hands

# 2. 在fit SMPLX之后，获取手部点云
# 在大约第360行，fit_smplx之后添加：

# 获取SMPLX顶点
with torch.no_grad():
    pose = params['pose']
    beta = params['beta']
    trans = params['trans']
    
    smplx_output = smplx_model(
        betas=beta,
        body_pose=pose[:, 3:66],
        global_orient=pose[:, :3],
        transl=trans,
        jaw_pose=pose[:, 66:69],
        leye_pose=pose[:, 69:72],
        reye_pose=pose[:, 72:75],
        left_hand_pose=pose[:, 75:120],
        right_hand_pose=pose[:, 120:165],
        expression=params.get('expression', torch.zeros(1, 10).cuda()),
    )
    smplx_vertices = smplx_output.vertices

# 从扫描点云中采样手部附近的点
hand_points, hand_indices, hand_distances = sample_points_near_hands(
    scan_points=mesh_src.vertices,  # 或 input_points
    smplx_vertices=smplx_vertices,
    radius=0.15,  # 15cm半径，可调整
    hand='both',
    method='radius'
)

print(f"手部采样点数: {len(hand_points)}")

# 3. 保存手部采样结果
np.savez(
    out_dir + "/vis/" + name + "/hand_sampled_points.npz",
    hand_points=hand_points,
    hand_indices=hand_indices,
    distances=hand_distances
)

# 4. 可选：可视化或进一步处理手部点云
# 例如：用于手部精细化fitting，或者手部姿态优化
"""
    
    print(code_example)
    print("\n上述代码展示了如何在你的评估流程中集成手部采样功能")


if __name__ == "__main__":
    # 运行基础示例
    example_basic_usage()
    
    # 运行左右手分别采样示例
    example_separate_hands()
    
    # 展示集成代码示例
    example_integration_in_evaluation()
    
    # 如果你有实际数据，取消下面的注释并提供路径
    # example_with_real_data("path/to/your/scan.npz")
    
    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)
