import os
import glob
import numpy as np
import trimesh
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime
import matplotlib.pyplot as plt
from smplx import SMPL
import json


# CAPE without chamfer refinement
# mean v2v error:  0.017251761702706554 sample num:  1021
# mean mpjpe error:  0.01051824787598793 sample num:  1021
def main():
    body_model_path = (
        "datafolder/body_models/smpl/neutral/SMPL_NEUTRAL_10pc_rmchumpy.pkl"
    )
    SMPL_model = SMPL(
        body_model_path,
        create_body_pose=True,
        create_betas=True,
        create_global_orient=True,
        create_transl=True,
    )
    t_body = SMPL_model.forward()
    # print(t_body.vertices.shape)
    t_body = trimesh.Trimesh(
        vertices=t_body.vertices.detach().cpu().numpy()[0], faces=SMPL_model.faces
    )

    with open("src/lvd_templ/evaluation/smpl_vert_segmentation.json", "r") as f:
        smpl_seg = json.load(f)
    body_parts = list(smpl_seg.keys())  # 24个部位

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = "cape_hitpts"
    log_tensorboard = False
    if log_tensorboard:
        tensorboard_name = f"{timestamp}_{exp_name}"
        writer = SummaryWriter(
            log_dir=os.path.join(
                "output/tensorboard_logs",
                tensorboard_name,
            )
        )
    save_error_mesh = False
    save_error_mesh_mean = True
    save_part_error_mean = True
    save_log_file = False

    flag = 1  # 0: before cham_refine, 1: after cham_refine
    # mode = "before_cham_refine" if flag == 0 else "after_cham_refine"

    gt_smpl_folder = "datafolder/CAPE_reorganized/cape_release/smpl_reorganized"
    gt_scan_folder = "datafolder/CAPE_reorganized/cape_release/model_reorganized"
    pred_folder = f"output/matchAMASS_CAPE/{exp_name}"
    # pred_folder = 'output/matchAMASS_CAPE/cape_hitpts'
    # pred_folder = 'output/matchAMASS_CAPE/cape_eq_hitpts'
    # pred_folder = 'output/matchAMASS_CAPE/cape_pred_inner_points'
    # names = sorted(os.listdir(pred_folder))
    # print(len(names))
    # print(names[0])
    # print(names[-1])
    # exit()

    # gt_smpl_folder = "datafolder/4D-DRESS/data_processed/smplh"
    # pred_folder = "/home/boqian/code/NICP/output/matchAMASS_4D-DRESS/demo"

    all_v2v_error = 0.0
    all_mpjpe_error = 0.0
    all_v2v_error_cham = 0.0
    all_mpjpe_error_cham = 0.0
    sample_num = 0
    v2v_file = os.path.join(pred_folder, "v2v_error.txt")
    mpjpe_file = os.path.join(pred_folder, "mpjpe_error.txt")
    v2v_file_cham = os.path.join(pred_folder, "v2v_error_cham.txt")
    mpjpe_file_cham = os.path.join(pred_folder, "mpjpe_error_cham.txt")
    if os.path.isfile(v2v_file):
        os.remove(v2v_file)
    if os.path.isfile(mpjpe_file):
        os.remove(mpjpe_file)
    if os.path.isfile(v2v_file_cham):
        os.remove(v2v_file_cham)
    if os.path.isfile(mpjpe_file_cham):
        os.remove(mpjpe_file_cham)
    all_v2v_map = np.zeros((6890,))
    all_v2v_map_cham = np.zeros((6890,))
    if save_part_error_mean:
        if sample_num == 0:
            part_v2v_all = {part: [] for part in body_parts}
            part_v2v_all_cham = {part: [] for part in body_parts}
    for name in tqdm(sorted(os.listdir(os.path.join(pred_folder, "vis")))):

        # v2v
        os.rename(
            os.path.join(pred_folder, "vis", name, "outcape_ss.ply"),
            os.path.join(pred_folder, "vis", name, "cape_ss.ply"),
        )
        os.rename(
            os.path.join(pred_folder, "vis", name, "outcape_ss_cham_0.ply"),
            os.path.join(pred_folder, "vis", name, "cape_ss_cham.ply"),
        )
        pred_smpl_path = os.path.join(pred_folder, "vis", name, "cape_ss.ply")
        pred_smpl_path_cham = os.path.join(pred_folder, "vis", name, "cape_ss_cham.ply")
        # assert os.path.isfile(pred_smpl_path)
        if not os.path.isfile(pred_smpl_path):
            continue
        # assert os.path.isfile(pred_smpl_path_cham)
        if not os.path.isfile(pred_smpl_path_cham):
            continue

        gt_smpl_path = os.path.join(gt_smpl_folder, name, f"mesh_smpl_{name}.obj")
        gt_scan_path = os.path.join(gt_scan_folder, name, f"{name}.obj")
        assert os.path.isfile(gt_smpl_path)
        assert os.path.isfile(gt_scan_path)

        gt_smpl_mesh = trimesh.load_mesh(
            gt_smpl_path, maintain_order=True, process=False
        )
        gt_scan_mesh = trimesh.load_mesh(
            gt_scan_path, maintain_order=True, process=False
        )
        pred_smpl_mesh = trimesh.load_mesh(
            pred_smpl_path, maintain_order=True, process=False
        )
        pred_smpl_mesh_cham = trimesh.load_mesh(
            pred_smpl_path_cham, maintain_order=True, process=False
        )

        gt_smpl_vertices = gt_smpl_mesh.vertices
        gt_scan_vertices = gt_scan_mesh.vertices

        gt_scan_min_xyz = np.min(gt_scan_vertices, axis=0)
        gt_scan_max_xyz = np.max(gt_scan_vertices, axis=0)
        gt_scan_center = (gt_scan_min_xyz + gt_scan_max_xyz) / 2.0

        gt_smpl_vertices = gt_smpl_vertices - gt_scan_center
        gt_scan_vertices = gt_scan_vertices - gt_scan_center

        gt_smpl_mesh.vertices = gt_smpl_vertices
        gt_scan_mesh.vertices = gt_scan_vertices

        # gt_smpl_mesh.export(os.path.join(pred_folder, "vis", name, "gt_smpl.ply"))
        # gt_scan_mesh.export(os.path.join(pred_folder, "vis", name, "gt_scan.ply"))

        gt_smpl_verts = np.asarray(gt_smpl_mesh.vertices)
        pred_smpl_verts = np.asarray(pred_smpl_mesh.vertices)
        pred_smpl_verts_cham = np.asarray(pred_smpl_mesh_cham.vertices)
        vertex_errors = np.linalg.norm(gt_smpl_verts - pred_smpl_verts, axis=1)
        vertex_errors_cham = np.linalg.norm(
            gt_smpl_verts - pred_smpl_verts_cham, axis=1
        )
        all_v2v_map += vertex_errors
        all_v2v_map_cham += vertex_errors_cham
        if save_part_error_mean:
            for part in body_parts:
                idxs = smpl_seg[part]
                part_v2v_all[part].append(vertex_errors[idxs])
                part_v2v_all_cham[part].append(vertex_errors_cham[idxs])
        if save_error_mesh:
            # before chamfer refine
            # 1. 计算每个顶点的误差
            # shape: (6890,)

            # 2. 归一化误差
            min_err = vertex_errors.min()
            max_err = vertex_errors.max()
            norm_errors = (vertex_errors - min_err) / (
                max_err - min_err + 1e-8
            )  # 防止除0

            # 3. 选择 colormap 并映射颜色
            cmap = plt.get_cmap("jet")  # 你可以换成 'viridis' 等
            vertex_colors = cmap(norm_errors)[:, :3]  # shape: (6890, 3)，去掉alpha通道

            # 4. 赋值给 mesh 并保存
            colored_mesh = pred_smpl_mesh.copy()
            colored_mesh.visual.vertex_colors = (vertex_colors * 255).astype(np.uint8)

            # 保存带颜色的 mesh
            colored_mesh.export(
                os.path.join(pred_folder, "vis", name, "cape_ss_error.ply")
            )

            # after chamfer refine

            min_err_cham = vertex_errors_cham.min()
            max_err_cham = vertex_errors_cham.max()
            norm_errors_cham = (vertex_errors_cham - min_err_cham) / (
                max_err_cham - min_err_cham + 1e-8
            )
            cmap_cham = plt.get_cmap("jet")
            vertex_colors_cham = cmap_cham(norm_errors_cham)[:, :3]
            colored_mesh_cham = pred_smpl_mesh_cham.copy()
            colored_mesh_cham.visual.vertex_colors = (vertex_colors_cham * 255).astype(
                np.uint8
            )
            colored_mesh_cham.export(
                os.path.join(pred_folder, "vis", name, "cape_ss_error_cham.ply")
            )
        # exit()

        v2v_error = np.linalg.norm(gt_smpl_verts - pred_smpl_verts, axis=1).mean()
        v2v_error_cham = np.linalg.norm(
            gt_smpl_verts - pred_smpl_verts_cham, axis=1
        ).mean()
        # print(v2v_error)
        # print(v2v_error_cham)
        if log_tensorboard:
            # Log to tensorboard
            writer.add_scalar("v2v_error", v2v_error, sample_num)
            writer.add_scalar("v2v_error_cham", v2v_error_cham, sample_num)

        if save_log_file:
            with open(v2v_file, "a") as f:
                f.write(f"{name} {v2v_error}\n")
            with open(v2v_file_cham, "a") as f:
                f.write(f"{name} {v2v_error_cham}\n")
        all_v2v_error += v2v_error
        all_v2v_error_cham += v2v_error_cham

        # mpjpe
        considered_joints_num = 22
        pred_info = np.load(
            os.path.join(
                pred_folder, "vis", name, "pred_smpl_info_before_cham_refine.npz"
            )
        )
        pred_info_cham = np.load(
            os.path.join(
                pred_folder, "vis", name, "pred_smpl_info_after_cham_refine.npz"
            )
        )
        pred_joints = pred_info["joints"]
        pred_joints_cham = pred_info_cham["joints"]

        gt_info = np.load(os.path.join(gt_smpl_folder, name, f"info_{name}.npz"))
        gt_joints = gt_info["joints"] - gt_scan_center

        mpjpe_error = np.linalg.norm(
            pred_joints[:considered_joints_num, :]
            - gt_joints[:considered_joints_num, :],
            axis=1,
        ).mean()
        mpjpe_error_cham = np.linalg.norm(
            pred_joints_cham[:considered_joints_num, :]
            - gt_joints[:considered_joints_num, :],
            axis=1,
        ).mean()
        # print(mpjpe_error)
        # print(mpjpe_error_cham)
        if save_log_file:
            with open(mpjpe_file, "a") as f:
                f.write(f"{name} {mpjpe_error}\n")
            with open(mpjpe_file_cham, "a") as f:
                f.write(f"{name} {mpjpe_error_cham}\n")
        if log_tensorboard:
            writer.add_scalar("mpjpe_error", mpjpe_error, sample_num)
            writer.add_scalar("mpjpe_error_cham", mpjpe_error_cham, sample_num)
        all_mpjpe_error += mpjpe_error
        all_mpjpe_error_cham += mpjpe_error_cham

        sample_num += 1
    if save_part_error_mean:
        part_v2v_all_flat = {
            part: np.concatenate(part_v2v_all[part]) for part in body_parts
        }
        part_v2v_all_flat_cham = {
            part: np.concatenate(part_v2v_all_cham[part]) for part in body_parts
        }

        mean_part_v2v = {part: np.mean(part_v2v_all_flat[part]) for part in body_parts}
        mean_part_v2v_cham = {
            part: np.mean(part_v2v_all_flat_cham[part]) for part in body_parts
        }
        all_v2v = np.concatenate([part_v2v_all_flat[part] for part in body_parts])
        all_v2v_cham = np.concatenate(
            [part_v2v_all_flat_cham[part] for part in body_parts]
        )
        mean_v2v_all = all_v2v.mean()
        mean_v2v_all_cham = all_v2v_cham.mean()

        # 去掉hand
        exclude_hands = ["leftHand", "rightHand"]
        include_parts_no_hand = [p for p in body_parts if p not in exclude_hands]
        v2v_no_hand = np.concatenate(
            [part_v2v_all_flat[p] for p in include_parts_no_hand]
        )
        v2v_no_hand_cham = np.concatenate(
            [part_v2v_all_flat_cham[p] for p in include_parts_no_hand]
        )
        mean_v2v_no_hand = v2v_no_hand.mean()
        mean_v2v_no_hand_cham = v2v_no_hand_cham.mean()

        # 去掉hand和index1
        exclude_hands_index = [
            "leftHand",
            "rightHand",
            "leftHandIndex1",
            "rightHandIndex1",
        ]
        include_parts_no_hand_index = [
            p for p in body_parts if p not in exclude_hands_index
        ]
        v2v_no_hand_index = np.concatenate(
            [part_v2v_all_flat[p] for p in include_parts_no_hand_index]
        )
        v2v_no_hand_index_cham = np.concatenate(
            [part_v2v_all_flat_cham[p] for p in include_parts_no_hand_index]
        )
        mean_v2v_no_hand_index = v2v_no_hand_index.mean()
        mean_v2v_no_hand_index_cham = v2v_no_hand_index_cham.mean()

        # 根据mean_part_v2v为t_body按部位着色
        # 1. 归一化mean_part_v2v到[0,1]
        mean_part_v2v_values = np.array([mean_part_v2v[part] for part in body_parts])
        min_part_v2v = mean_part_v2v_values.min()
        max_part_v2v = mean_part_v2v_values.max()
        norm_mean_part_v2v = (mean_part_v2v_values - min_part_v2v) / (
            max_part_v2v - min_part_v2v + 1e-8
        )

        # 2. 使用jet colormap映射颜色
        cmap = plt.get_cmap("jet")
        part_colors = cmap(norm_mean_part_v2v)[:, :3]  # shape: (24, 3)

        # 3. 为每个顶点分配对应部位的颜色
        vertex_colors_by_part = np.zeros((6890, 3))
        for i, part in enumerate(body_parts):
            idxs = smpl_seg[part]
            vertex_colors_by_part[idxs] = part_colors[i]

        # 4. 保存带颜色的mesh
        colored_mesh_by_part = t_body.copy()
        colored_mesh_by_part.visual.vertex_colors = (
            vertex_colors_by_part * 255
        ).astype(np.uint8)
        colored_mesh_by_part.export(
            os.path.join(pred_folder, f"{exp_name}_mean_error_by_part.ply")
        )

        # 同样处理chamfer refine后的结果
        mean_part_v2v_cham_values = np.array(
            [mean_part_v2v_cham[part] for part in body_parts]
        )
        min_part_v2v_cham = mean_part_v2v_cham_values.min()
        max_part_v2v_cham = mean_part_v2v_cham_values.max()
        norm_mean_part_v2v_cham = (mean_part_v2v_cham_values - min_part_v2v_cham) / (
            max_part_v2v_cham - min_part_v2v_cham + 1e-8
        )

        part_colors_cham = cmap(norm_mean_part_v2v_cham)[:, :3]

        vertex_colors_by_part_cham = np.zeros((6890, 3))
        for i, part in enumerate(body_parts):
            idxs = smpl_seg[part]
            vertex_colors_by_part_cham[idxs] = part_colors_cham[i]

        colored_mesh_by_part_cham = t_body.copy()
        colored_mesh_by_part_cham.visual.vertex_colors = (
            vertex_colors_by_part_cham * 255
        ).astype(np.uint8)
        colored_mesh_by_part_cham.export(
            os.path.join(pred_folder, f"{exp_name}_mean_error_by_part_cham.ply")
        )

        np.savez(
            os.path.join(pred_folder, f"{exp_name}_v2v_parts.npz"),
            part_v2v=part_v2v_all_flat,
            part_v2v_cham=part_v2v_all_flat_cham,
            mean_part_v2v=mean_part_v2v,
            mean_part_v2v_cham=mean_part_v2v_cham,
            mean_v2v_all=mean_v2v_all,
            mean_v2v_all_cham=mean_v2v_all_cham,
            mean_v2v_no_hand=mean_v2v_no_hand,
            mean_v2v_no_hand_cham=mean_v2v_no_hand_cham,
            mean_v2v_no_hand_index=mean_v2v_no_hand_index,
            mean_v2v_no_hand_index_cham=mean_v2v_no_hand_index_cham,
        )
    if save_error_mesh_mean:
        min_err = all_v2v_map.min()
        max_err = all_v2v_map.max()
        norm_errors = (all_v2v_map - min_err) / (max_err - min_err + 1e-8)
        cmap = plt.get_cmap("jet")
        vertex_colors = cmap(norm_errors)[:, :3]
        colored_mesh = t_body.copy()
        colored_mesh.visual.vertex_colors = (vertex_colors * 255).astype(np.uint8)
        colored_mesh.export(os.path.join(pred_folder, f"{exp_name}_error_mean.ply"))

        # 去掉hand后的error map
        exclude_hands = ["leftHand", "rightHand"]
        include_parts_no_hand = [p for p in body_parts if p not in exclude_hands]
        all_v2v_map_no_hand = all_v2v_map.copy()
        for part in exclude_hands:
            idxs = smpl_seg[part]
            all_v2v_map_no_hand[idxs] = 0  # 将hand部位的误差设为0

        min_err_no_hand = all_v2v_map_no_hand[all_v2v_map_no_hand > 0].min()
        max_err_no_hand = all_v2v_map_no_hand.max()
        norm_errors_no_hand = (all_v2v_map_no_hand - min_err_no_hand) / (
            max_err_no_hand - min_err_no_hand + 1e-8
        )
        vertex_colors_no_hand = cmap(norm_errors_no_hand)[:, :3]
        colored_mesh_no_hand = t_body.copy()
        colored_mesh_no_hand.visual.vertex_colors = (
            vertex_colors_no_hand * 255
        ).astype(np.uint8)
        colored_mesh_no_hand.export(
            os.path.join(pred_folder, f"{exp_name}_error_mean_no_hand.ply")
        )

        # 去掉hand和index1后的error map
        exclude_hands_index = [
            "leftHand",
            "rightHand",
            "leftHandIndex1",
            "rightHandIndex1",
        ]
        include_parts_no_hand_index = [
            p for p in body_parts if p not in exclude_hands_index
        ]
        all_v2v_map_no_hand_index = all_v2v_map.copy()
        for part in exclude_hands_index:
            idxs = smpl_seg[part]
            all_v2v_map_no_hand_index[idxs] = 0  # 将hand和index1部位的误差设为0

        min_err_no_hand_index = all_v2v_map_no_hand_index[
            all_v2v_map_no_hand_index > 0
        ].min()
        max_err_no_hand_index = all_v2v_map_no_hand_index.max()
        norm_errors_no_hand_index = (
            all_v2v_map_no_hand_index - min_err_no_hand_index
        ) / (max_err_no_hand_index - min_err_no_hand_index + 1e-8)
        vertex_colors_no_hand_index = cmap(norm_errors_no_hand_index)[:, :3]
        colored_mesh_no_hand_index = t_body.copy()
        colored_mesh_no_hand_index.visual.vertex_colors = (
            vertex_colors_no_hand_index * 255
        ).astype(np.uint8)
        colored_mesh_no_hand_index.export(
            os.path.join(pred_folder, f"{exp_name}_error_mean_no_hand_index.ply")
        )

        # chamfer refine后的结果
        min_err_cham = all_v2v_map_cham.min()
        max_err_cham = all_v2v_map_cham.max()
        norm_errors_cham = (all_v2v_map_cham - min_err_cham) / (
            max_err_cham - min_err_cham + 1e-8
        )
        cmap_cham = plt.get_cmap("jet")
        vertex_colors_cham = cmap_cham(norm_errors_cham)[:, :3]
        colored_mesh_cham = t_body.copy()
        colored_mesh_cham.visual.vertex_colors = (vertex_colors_cham * 255).astype(
            np.uint8
        )
        colored_mesh_cham.export(
            os.path.join(pred_folder, f"{exp_name}_error_mean_cham.ply")
        )

        # chamfer refine后去掉hand的error map
        all_v2v_map_cham_no_hand = all_v2v_map_cham.copy()
        for part in exclude_hands:
            idxs = smpl_seg[part]
            all_v2v_map_cham_no_hand[idxs] = 0

        min_err_cham_no_hand = all_v2v_map_cham_no_hand[
            all_v2v_map_cham_no_hand > 0
        ].min()
        max_err_cham_no_hand = all_v2v_map_cham_no_hand.max()
        norm_errors_cham_no_hand = (all_v2v_map_cham_no_hand - min_err_cham_no_hand) / (
            max_err_cham_no_hand - min_err_cham_no_hand + 1e-8
        )
        vertex_colors_cham_no_hand = cmap_cham(norm_errors_cham_no_hand)[:, :3]
        colored_mesh_cham_no_hand = t_body.copy()
        colored_mesh_cham_no_hand.visual.vertex_colors = (
            vertex_colors_cham_no_hand * 255
        ).astype(np.uint8)
        colored_mesh_cham_no_hand.export(
            os.path.join(pred_folder, f"{exp_name}_error_mean_cham_no_hand.ply")
        )

        # chamfer refine后去掉hand和index1的error map
        all_v2v_map_cham_no_hand_index = all_v2v_map_cham.copy()
        for part in exclude_hands_index:
            idxs = smpl_seg[part]
            all_v2v_map_cham_no_hand_index[idxs] = 0

        min_err_cham_no_hand_index = all_v2v_map_cham_no_hand_index[
            all_v2v_map_cham_no_hand_index > 0
        ].min()
        max_err_cham_no_hand_index = all_v2v_map_cham_no_hand_index.max()
        norm_errors_cham_no_hand_index = (
            all_v2v_map_cham_no_hand_index - min_err_cham_no_hand_index
        ) / (max_err_cham_no_hand_index - min_err_cham_no_hand_index + 1e-8)
        vertex_colors_cham_no_hand_index = cmap_cham(norm_errors_cham_no_hand_index)[
            :, :3
        ]
        colored_mesh_cham_no_hand_index = t_body.copy()
        colored_mesh_cham_no_hand_index.visual.vertex_colors = (
            vertex_colors_cham_no_hand_index * 255
        ).astype(np.uint8)
        colored_mesh_cham_no_hand_index.export(
            os.path.join(pred_folder, f"{exp_name}_error_mean_cham_no_hand_index.ply")
        )
        # # 在循环结束后添加验证
        # print("=== 验证两种计算方式 ===")
        # print(f"方式1 - mean_v2v_all: {mean_v2v_all}")
        # print(f"方式2 - all_v2v_error/sample_num: {all_v2v_error / sample_num}")
        # print(f"差异: {abs(mean_v2v_all - all_v2v_error / sample_num)}")

        # # 验证顶点数量
        # total_vertices_way1 = sum(len(part_v2v_all_flat[part]) for part in body_parts)
        # total_vertices_way2 = sample_num * 6890
        # print(f"方式1顶点总数: {total_vertices_way1}")
        # print(f"方式2顶点总数: {total_vertices_way2}")

    # print(sample_num)
    print("mean v2v error: ", all_v2v_error / sample_num, "sample num: ", sample_num)
    print(
        "mean mpjpe error: ", all_mpjpe_error / sample_num, "sample num: ", sample_num
    )
    print(
        "mean v2v error with chamfer refine: ",
        all_v2v_error_cham / sample_num,
        "sample num: ",
        sample_num,
    )
    print(
        "mean mpjpe error with chamfer refine: ",
        all_mpjpe_error_cham / sample_num,
        "sample num: ",
        sample_num,
    )
    if save_log_file:
        with open(v2v_file, "a") as f:
            f.write(
                f"mean v2v error: {all_v2v_error / sample_num} sample num: {sample_num}\n"
            )
        with open(mpjpe_file, "a") as f:
            f.write(
                f"mean mpjpe error: {all_mpjpe_error / sample_num} sample num: {sample_num}\n"
            )
        with open(v2v_file_cham, "a") as f:
            f.write(
                f"mean v2v error with chamfer refine: {all_v2v_error_cham / sample_num} sample num: {sample_num}\n"
            )
        with open(mpjpe_file_cham, "a") as f:
            f.write(
                f"mean mpjpe error with chamfer refine: {all_mpjpe_error_cham / sample_num} sample num: {sample_num}\n"
            )
        if not os.path.isfile(
            os.path.join(pred_folder, f"v2v_error_{all_v2v_error / sample_num}.txt")
        ):
            os.rename(
                v2v_file,
                os.path.join(
                    pred_folder, f"v2v_error_{all_v2v_error / sample_num}.txt"
                ),
            )
        else:
            os.remove(v2v_file)
        if not os.path.isfile(
            os.path.join(pred_folder, f"mpjpe_error_{all_mpjpe_error / sample_num}.txt")
        ):
            os.rename(
                mpjpe_file,
                os.path.join(
                    pred_folder, f"mpjpe_error_{all_mpjpe_error / sample_num}.txt"
                ),
            )
        else:
            os.remove(mpjpe_file)
        if not os.path.isfile(
            os.path.join(
                pred_folder, f"v2v_error_cham_{all_v2v_error_cham / sample_num}.txt"
            )
        ):
            os.rename(
                v2v_file_cham,
                os.path.join(
                    pred_folder, f"v2v_error_cham_{all_v2v_error_cham / sample_num}.txt"
                ),
            )
        else:
            os.remove(v2v_file_cham)
        if not os.path.isfile(
            os.path.join(
                pred_folder, f"mpjpe_error_cham_{all_mpjpe_error_cham / sample_num}.txt"
            )
        ):
            os.rename(
                mpjpe_file_cham,
                os.path.join(
                    pred_folder,
                    f"mpjpe_error_cham_{all_mpjpe_error_cham / sample_num}.txt",
                ),
            )
        else:
            os.remove(mpjpe_file_cham)
    if log_tensorboard:
        writer.add_scalar("mean_v2v_error", all_v2v_error / sample_num, 0)
        writer.add_scalar("mean_mpjpe_error", all_mpjpe_error / sample_num, 0)
        writer.add_scalar("mean_v2v_error_cham", all_v2v_error_cham / sample_num, 0)
        writer.add_scalar("mean_mpjpe_error_cham", all_mpjpe_error_cham / sample_num, 0)
        writer.close()


if __name__ == "__main__":
    main()

# CAPE with chamfer refine:
# mean v2v error:  0.01245252614256733 sample num:  1021
# mean mpjpe error:  0.010510880577612403 sample num:  1021

# CAPE without chamfer refine:
# mean v2v error:  0.01725598894871101 sample num:  1021
# mean mpjpe error:  0.013426505782651504 sample num:  1021


# 4D-dress without chamfer refine:
# mean v2v error:  0.047543300797230875 sample num:  1943
# mean mpjpe error:  0.03653855350854022 sample num:  1943


# 4D-dress with chamfer refine:
# mean v2v error:  0.047382420365031784 sample num:  1943
# mean mpjpe error:  0.03728696454423746 sample num:  1943
