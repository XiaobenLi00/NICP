"""
Hand Sampling Toolkit for NICP
根据SMPLX人体模型的手部顶点位置，从扫描点云中采样手部附近的点

主要功能:
- 从SMPLX顶点提取手部区域
- 基于手部顶点采样扫描点云
- 支持三种采样方法：半径、KNN、自适应
- 支持左手、右手、双手采样

快速使用:
    from hand_sampling_toolkit import sample_points_near_hands
    
    hand_points, indices, distances = sample_points_near_hands(
        scan_points=scan_points,
        smplx_vertices=smplx_vertices,
        radius=0.15,
        hand='both',
        method='radius'
    )

版本: 1.0
日期: 2025-10-25
"""

from .hand_sampling_utils import (
    sample_points_near_hands,
    get_hand_vertices_from_smplx,
    get_hand_vertex_indices,
    get_hand_region_mask,
    visualize_hand_sampling,
)

__version__ = "1.0.0"
__author__ = "NICP Project"
__all__ = [
    "sample_points_near_hands",
    "get_hand_vertices_from_smplx",
    "get_hand_vertex_indices",
    "get_hand_region_mask",
    "visualize_hand_sampling",
]
