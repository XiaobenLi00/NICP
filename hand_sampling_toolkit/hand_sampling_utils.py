"""
手部点云采样工具函数
根据SMPLX人体模型的手部顶点位置，从扫描点云中采样手部附近的点
"""

import numpy as np
import torch
from scipy.spatial import cKDTree


# SMPLX手部顶点索引范围
# SMPLX模型总共有10475个顶点
# 左手顶点索引范围: 约5361-5778 (对应smplx.vertex_ids['left_hand'])
# 右手顶点索引范围: 约8079-8496 (对应smplx.vertex_ids['right_hand'])
# 这些索引可能因SMPLX版本而略有不同，建议通过smplx.vertex_ids获取准确索引

def get_hand_vertex_indices():
    """
    获取SMPLX模型的手部顶点索引
    
    Returns:
        dict: 包含'left_hand'和'right_hand'的顶点索引
    """
    # 这里提供一个通用的手部顶点索引范围
    # 建议从SMPLX模型中动态获取: smplx_model.vertex_ids
    hand_indices = {
        'left_hand': np.arange(5361, 5778),   # 左手顶点
        'right_hand': np.arange(8079, 8496),  # 右手顶点
    }
    return hand_indices


def get_hand_vertices_from_smplx(smplx_vertices, hand='both'):
    """
    从SMPLX顶点中提取手部顶点
    
    Args:
        smplx_vertices: SMPLX模型的所有顶点 (N, 3) numpy array或torch tensor
        hand: 'left', 'right' 或 'both'
        
    Returns:
        hand_vertices: 手部顶点坐标 (M, 3)
        hand_indices: 对应的顶点索引
    """
    if isinstance(smplx_vertices, torch.Tensor):
        smplx_vertices = smplx_vertices.detach().cpu().numpy()
    
    # 确保是2D数组
    if smplx_vertices.ndim == 3:
        smplx_vertices = smplx_vertices.squeeze(0)
    
    hand_indices_dict = get_hand_vertex_indices()
    
    if hand == 'left':
        hand_indices = hand_indices_dict['left_hand']
    elif hand == 'right':
        hand_indices = hand_indices_dict['right_hand']
    elif hand == 'both':
        hand_indices = np.concatenate([
            hand_indices_dict['left_hand'],
            hand_indices_dict['right_hand']
        ])
    else:
        raise ValueError(f"hand参数必须是'left', 'right'或'both', 得到: {hand}")
    
    hand_vertices = smplx_vertices[hand_indices]
    
    return hand_vertices, hand_indices


def sample_points_near_hands(scan_points, smplx_vertices, 
                             radius=0.1, 
                             num_samples=None,
                             hands_index=None,
                             method='radius'):
    """
    根据SMPLX手部顶点位置，从扫描点云中采样手部附近的点
    
    Args:
        scan_points: 输入扫描点云 (N, 3) numpy array或torch tensor
        smplx_vertices: SMPLX模型顶点 (10475, 3) 或 (1, 10475, 3)
        radius: 采样半径，单位与点云坐标系一致 (默认0.1)
        num_samples: 期望采样的点数，None表示返回所有在半径内的点
        hand: 'left', 'right' 或 'both' - 指定采样哪只手附近的点
        method: 采样方法
            - 'radius': 采样半径范围内的所有点
            - 'knn': 采样k个最近邻点
            - 'adaptive': 自适应采样，每个手部顶点采样固定数量的最近邻
            
    Returns:
        sampled_points: 采样得到的点 (M, 3)
        sampled_indices: 采样点在原始点云中的索引
        distances: 采样点到最近手部顶点的距离
    """
    # 转换为numpy数组
    if isinstance(scan_points, torch.Tensor):
        scan_points = scan_points.detach().cpu().numpy()
    if isinstance(smplx_vertices, torch.Tensor):
        smplx_vertices = smplx_vertices.detach().cpu().numpy()
    
    # 确保是2D数组
    if scan_points.ndim == 3:
        scan_points = scan_points.squeeze(0)
    if smplx_vertices.ndim == 3:
        smplx_vertices = smplx_vertices.squeeze(0)
    
    # 获取手部顶点
    # hand_vertices, hand_vertex_indices = get_hand_vertices_from_smplx(
    #     smplx_vertices, hand=hand
    # )

    hand_vertices = smplx_vertices[hands_index]
    
    print(f"手部顶点数量: {len(hand_vertices)}")
    print(f"扫描点云数量: {len(scan_points)}")
    
    # 构建KD树用于快速最近邻搜索
    scan_tree = cKDTree(scan_points)
    hand_tree = cKDTree(hand_vertices)
    
    if method == 'radius':
        # 方法1: 半径搜索 - 找到所有在指定半径内的点
        # 对每个扫描点，查询其到手部顶点的最近距离
        distances, _ = hand_tree.query(scan_points, k=1)
        
        # 筛选出距离小于radius的点
        mask = distances < radius
        sampled_indices = np.where(mask)[0]
        sampled_points = scan_points[sampled_indices]
        sampled_distances = distances[sampled_indices]
        
    elif method == 'knn':
        # 方法2: KNN - 找到距离手部最近的k个点
        if num_samples is None:
            num_samples = min(1000, len(scan_points))  # 默认1000个点
        
        # 查询所有扫描点到手部的距离
        distances, _ = hand_tree.query(scan_points, k=1)
        
        # 选择距离最小的num_samples个点
        sampled_indices = np.argpartition(distances, min(num_samples, len(distances)-1))[:num_samples]
        sampled_points = scan_points[sampled_indices]
        sampled_distances = distances[sampled_indices]
        
    elif method == 'adaptive':
        # 方法3: 自适应采样 - 每个手部顶点采样固定数量的最近邻
        points_per_vertex = num_samples // len(hand_vertices) if num_samples else 10
        
        all_indices = []
        all_distances = []
        
        for hand_vert in hand_vertices:
            # 对每个手部顶点，找到最近的points_per_vertex个扫描点
            dists, indices = scan_tree.query(hand_vert, k=points_per_vertex)
            all_indices.extend(indices)
            all_distances.extend(dists)
        
        # 去重（可能多个手部顶点采样到同一个扫描点）
        sampled_indices = np.unique(all_indices)
        sampled_points = scan_points[sampled_indices]
        
        # 重新计算去重后的距离
        sampled_distances, _ = hand_tree.query(sampled_points, k=1)
        
    else:
        raise ValueError(f"不支持的method: {method}")
    
    print(f"采样得到的点数: {len(sampled_points)}")
    
    return sampled_points, sampled_indices, sampled_distances


def visualize_hand_sampling(scan_points, smplx_vertices, sampled_points, 
                            hand='both', save_path=None):
    """
    可视化手部采样结果（需要安装open3d或trimesh）
    
    Args:
        scan_points: 原始扫描点云
        smplx_vertices: SMPLX顶点
        sampled_points: 采样得到的手部点云
        hand: 'left', 'right' 或 'both'
        save_path: 保存路径（可选）
    """
    try:
        import open3d as o3d
        
        # 创建点云对象
        pcd_scan = o3d.geometry.PointCloud()
        pcd_scan.points = o3d.utility.Vector3dVector(scan_points)
        pcd_scan.paint_uniform_color([0.7, 0.7, 0.7])  # 灰色
        
        pcd_sampled = o3d.geometry.PointCloud()
        pcd_sampled.points = o3d.utility.Vector3dVector(sampled_points)
        pcd_sampled.paint_uniform_color([1.0, 0.0, 0.0])  # 红色
        
        # 手部顶点
        hand_vertices, _ = get_hand_vertices_from_smplx(smplx_vertices, hand=hand)
        pcd_hand = o3d.geometry.PointCloud()
        pcd_hand.points = o3d.utility.Vector3dVector(hand_vertices)
        pcd_hand.paint_uniform_color([0.0, 1.0, 0.0])  # 绿色
        
        # 可视化
        o3d.visualization.draw_geometries([pcd_scan, pcd_sampled, pcd_hand],
                                         window_name="Hand Sampling Visualization")
        
        if save_path:
            # 保存采样结果
            o3d.io.write_point_cloud(save_path, pcd_sampled)
            print(f"采样结果已保存到: {save_path}")
            
    except ImportError:
        print("需要安装open3d进行可视化: pip install open3d")


def get_hand_region_mask(smplx_vertices, hand='both'):
    """
    获取手部区域的mask
    
    Args:
        smplx_vertices: SMPLX顶点
        hand: 'left', 'right' 或 'both'
        
    Returns:
        mask: 布尔数组，标记哪些顶点属于手部
    """
    if isinstance(smplx_vertices, torch.Tensor):
        num_vertices = smplx_vertices.shape[-2]
    else:
        num_vertices = smplx_vertices.shape[-2] if smplx_vertices.ndim > 1 else len(smplx_vertices)
    
    mask = np.zeros(num_vertices, dtype=bool)
    
    hand_indices_dict = get_hand_vertex_indices()
    
    if hand in ['left', 'both']:
        mask[hand_indices_dict['left_hand']] = True
    if hand in ['right', 'both']:
        mask[hand_indices_dict['right_hand']] = True
    
    return mask


# 示例用法
if __name__ == "__main__":
    """
    使用示例
    """
    # 假设你已经有了：
    # 1. scan_points: 扫描点云 (N, 3)
    # 2. smplx_model: 已经fit好的SMPLX模型
    # 3. smplx_vertices: SMPLX模型的顶点 (10475, 3)
    
    # 示例代码（需要替换为实际数据）:
    """
    # 从SMPLX模型获取顶点
    with torch.no_grad():
        smplx_output = smplx_model(
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient,
            transl=transl,
            # ... 其他参数
        )
        smplx_vertices = smplx_output.vertices  # (1, 10475, 3)
    
    # 方法1: 半径采样
    sampled_points, indices, distances = sample_points_near_hands(
        scan_points=scan_points,
        smplx_vertices=smplx_vertices,
        radius=0.1,  # 10cm半径
        hand='both',
        method='radius'
    )
    
    # 方法2: KNN采样
    sampled_points, indices, distances = sample_points_near_hands(
        scan_points=scan_points,
        smplx_vertices=smplx_vertices,
        num_samples=2000,  # 采样2000个点
        hand='both',
        method='knn'
    )
    
    # 方法3: 自适应采样
    sampled_points, indices, distances = sample_points_near_hands(
        scan_points=scan_points,
        smplx_vertices=smplx_vertices,
        num_samples=2000,
        hand='left',  # 只采样左手
        method='adaptive'
    )
    
    print(f"采样得到的手部点云形状: {sampled_points.shape}")
    """
    
    print("手部采样工具已加载")
    print("请参考上述示例代码使用")
