import os
import glob
import numpy as np
import trimesh
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime
import matplotlib.pyplot as plt
from smplx import SMPL


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
    # exit()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = "4d-dress_gen_eq_hitpts"
    log_tensorboard = True
    if log_tensorboard:
        tensorboard_name = f"{timestamp}_{exp_name}"
        writer = SummaryWriter(
            log_dir=os.path.join(
                "output/tensorboard_logs",
                tensorboard_name,
            )
        )
    save_error_mesh = True

    flag = 1  # 0: before cham_refine, 1: after cham_refine
    # mode = "before_cham_refine" if flag == 0 else "after_cham_refine"

    gt_smpl_folder = "datafolder/4D-DRESS/data_processed/smplh"
    gt_scan_folder = "datafolder/4D-DRESS/data_processed/model"

    pred_folder = f"output/matchAMASS_4D-DRESS/{exp_name}"

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
    for name in tqdm(sorted(os.listdir(pred_folder))):

        # v2v
        pred_smpl_path = os.path.join(pred_folder, name, "outtag_4d-dress_ss.ply")
        pred_smpl_path_cham = os.path.join(
            pred_folder, name, "outtag_4d-dress_ss_cham_0.ply"
        )
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

        gt_smpl_verts = np.asarray(gt_smpl_mesh.vertices)
        pred_smpl_verts = np.asarray(pred_smpl_mesh.vertices)
        pred_smpl_verts_cham = np.asarray(pred_smpl_mesh_cham.vertices)

        if save_error_mesh:
            # before chamfer refine
            # 1. 计算每个顶点的误差
            vertex_errors = np.linalg.norm(
                gt_smpl_verts - pred_smpl_verts, axis=1
            )  # shape: (6890,)
            all_v2v_map += vertex_errors
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
                os.path.join(pred_folder, name, "4d-dress_ss_error.ply")
            )

            # after chamfer refine
            vertex_errors_cham = np.linalg.norm(
                gt_smpl_verts - pred_smpl_verts_cham, axis=1
            )
            all_v2v_map_cham += vertex_errors_cham
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
                os.path.join(pred_folder, name, "4d-dress_ss_error_cham.ply")
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

        with open(v2v_file, "a") as f:
            f.write(f"{name} {v2v_error}\n")
        with open(v2v_file_cham, "a") as f:
            f.write(f"{name} {v2v_error_cham}\n")
        all_v2v_error += v2v_error
        all_v2v_error_cham += v2v_error_cham

        # mpjpe
        considered_joints_num = 22
        pred_info = np.load(
            os.path.join(pred_folder, name, "pred_smpl_info_before_cham_refine.npz")
        )
        pred_info_cham = np.load(
            os.path.join(pred_folder, name, "pred_smpl_info_after_cham_refine.npz")
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
        # break

    if save_error_mesh:
        min_err = all_v2v_map.min()
        max_err = all_v2v_map.max()
        norm_errors = (all_v2v_map - min_err) / (max_err - min_err + 1e-8)
        cmap = plt.get_cmap("jet")
        vertex_colors = cmap(norm_errors)[:, :3]
        colored_mesh = t_body.copy()
        colored_mesh.visual.vertex_colors = (vertex_colors * 255).astype(np.uint8)
        colored_mesh.export(os.path.join(pred_folder, f"{exp_name}_error_mean.ply"))
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
            os.path.join(pred_folder, f"v2v_error_{all_v2v_error / sample_num}.txt"),
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
                pred_folder, f"mpjpe_error_cham_{all_mpjpe_error_cham / sample_num}.txt"
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
