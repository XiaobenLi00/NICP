import os
import glob
import json
import numpy as np
import sys

# sys.modules["numpy._core"] = np.core

import trimesh
from tqdm import tqdm
import pickle


# CAPE without chamfer refinement
# mean v2v error:  0.017251761702706554 sample num:  1021
# mean mpjpe error:  0.01051824787598793 sample num:  1021
def main():

    flag = 1  # 0: before cham_refine, 1: after cham_refine
    # mode = "before_cham_refine" if flag == 0 else "after_cham_refine"

    gt_smpl_folder = (
        "datafolder_new/CAPE_reorganized/cape_release/smplx_ratio5_from_PTF"
    )
    gt_scan_folder = "datafolder_new/CAPE_reorganized/cape_release/model_reorganized"
    # pred_folder = "output/matchAMASS_CAPE/cape_hitpts_77_x_new_fit"
    # pred_folder = "output/matchAMASS_CAPE/cape_pred_inner_points_cut_partial_68"
    # pred_folder = "output/matchAMASS_CAPE/cape_pred_inner_points_single_view_76"
    # pred_folder = "output/matchAMASS_CAPE/cape_pred_inner_points_50_500_cape"
    # pred_folder = "output/matchAMASS_CAPE/cape_pred_inner_points_500_50_cape"
    # pred_folder = "output/matchAMASS_CAPE/cape_pred_inner_points_500_500_cape"
    # pred_folder = "output/matchAMASS_CAPE/cape_pred_inner_points_gen_cape"
    # pred_folder = "output/matchAMASS_CAPE/cape_pred_inner_points_gen+_cape"
    pred_folder = "output/matchAMASS/cape_pred_inner_points_x_lovd_amass_epoch_39"

    # gt_smpl_folder = "datafolder/4D-DRESS/data_processed/smplh"
    # pred_folder = "/home/boqian/code/NICP/output/matchAMASS_4D-DRESS/demo"

    with open("src/lvd_templ/evaluation/smplx_vert_segmentation.json", "r") as f:
        smplx_seg = json.load(f)
    body_parts = list(smplx_seg.keys())  # 27个部位
    print(body_parts)
    print(len(body_parts))
    # 合并rightHand和rightHandIndex1的顶点索引，去重
    right_hand = list(set(smplx_seg["rightHand"] + smplx_seg["rightHandIndex1"]))
    # 合并leftHand和leftHandIndex1的顶点索引，去重
    left_hand = list(set(smplx_seg["leftHand"] + smplx_seg["leftHandIndex1"]))
    # 合并左右手所有顶点，去重
    two_hands = list(set(right_hand + left_hand))
    head = list(set(smplx_seg["head"]))
    # 除去手部相关key，合并剩余所有body part的顶点索引，去重
    hand_face_keys = {"rightHand", "rightHandIndex1", "leftHand", "leftHandIndex1", "head"}
    body_others = []
    for part in body_parts:
        if part not in hand_face_keys:
            body_others += smplx_seg[part]
    body_others = list(set(body_others))

    full_idxs = np.arange(0, 55)
    # 不能直接用np.arange减去list，应该用np.setdiff1d来排除指定的索引
    body_idxs = np.setdiff1d(np.arange(0, 22), [12, 15])
    lhand_idxs = np.arange(25, 40)
    rhand_idxs = np.arange(40, 55)
    head_idxs = [12, 15, 22, 23, 24]
    two_hands_idxs = np.arange(25, 55)

    # exit()

    sum_v2v_error_all = 0.0
    sum_v2v_error_hands = 0.0
    sum_v2v_error_lhand = 0.0
    sum_v2v_error_rhand = 0.0
    sum_v2v_error_head = 0.0
    sum_v2v_error_other = 0.0

    sum_mpjpe_error_all = 0.0
    sum_mpjpe_error_hands = 0.0
    sum_mpjpe_error_lhand = 0.0
    sum_mpjpe_error_rhand = 0.0
    sum_mpjpe_error_head = 0.0
    sum_mpjpe_error_other = 0.0

    sum_v2v_error_cham_all = 0.0
    sum_v2v_error_cham_hands = 0.0
    sum_v2v_error_cham_lhand = 0.0
    sum_v2v_error_cham_rhand = 0.0
    sum_v2v_error_cham_head = 0.0
    sum_v2v_error_cham_other = 0.0

    sum_mpjpe_error_cham_all = 0.0
    sum_mpjpe_error_cham_hands = 0.0
    sum_mpjpe_error_cham_lhand = 0.0
    sum_mpjpe_error_cham_rhand = 0.0
    sum_mpjpe_error_cham_head = 0.0
    sum_mpjpe_error_cham_other = 0.0

    sample_num = 0
    v2v_file_all = os.path.join(pred_folder, "v2v_error_all.txt")
    mpjpe_file_all = os.path.join(pred_folder, "mpjpe_error_all.txt")
    v2v_file_cham_all = os.path.join(pred_folder, "v2v_error_cham_all.txt")
    mpjpe_file_cham_all = os.path.join(pred_folder, "mpjpe_error_cham_all.txt")

    v2v_file_hands = os.path.join(pred_folder, "v2v_error_hands.txt")
    mpjpe_file_hands = os.path.join(pred_folder, "mpjpe_error_hands.txt")
    v2v_file_cham_hands = os.path.join(pred_folder, "v2v_error_cham_hands.txt")
    mpjpe_file_cham_hands = os.path.join(pred_folder, "mpjpe_error_cham_hands.txt")

    v2v_file_lhand = os.path.join(pred_folder, "v2v_error_lhand.txt")
    mpjpe_file_lhand = os.path.join(pred_folder, "mpjpe_error_lhand.txt")
    v2v_file_cham_lhand = os.path.join(pred_folder, "v2v_error_cham_lhand.txt")
    mpjpe_file_cham_lhand = os.path.join(pred_folder, "mpjpe_error_cham_lhand.txt")

    v2v_file_rhand = os.path.join(pred_folder, "v2v_error_rhand.txt")
    mpjpe_file_rhand = os.path.join(pred_folder, "mpjpe_error_rhand.txt")
    v2v_file_cham_rhand = os.path.join(pred_folder, "v2v_error_cham_rhand.txt")
    mpjpe_file_cham_rhand = os.path.join(pred_folder, "mpjpe_error_cham_rhand.txt")

    v2v_file_head = os.path.join(pred_folder, "v2v_error_head.txt")
    mpjpe_file_head = os.path.join(pred_folder, "mpjpe_error_head.txt")
    v2v_file_cham_head = os.path.join(pred_folder, "v2v_error_cham_head.txt")
    mpjpe_file_cham_head = os.path.join(pred_folder, "mpjpe_error_cham_head.txt")

    v2v_file_other = os.path.join(pred_folder, "v2v_error_other.txt")
    mpjpe_file_other = os.path.join(pred_folder, "mpjpe_error_other.txt")
    v2v_file_cham_other = os.path.join(pred_folder, "v2v_error_cham_other.txt")
    mpjpe_file_cham_other = os.path.join(pred_folder, "mpjpe_error_cham_other.txt")

    # if os.path.isfile(v2v_file):
    #     os.remove(v2v_file)
    # if os.path.isfile(mpjpe_file):
    #     os.remove(mpjpe_file)
    # if os.path.isfile(v2v_file_cham):
    #     os.remove(v2v_file_cham)
    # if os.path.isfile(mpjpe_file_cham):
    #     os.remove(mpjpe_file_cham)
    for name in tqdm(os.listdir(os.path.join(pred_folder, "vis"))):

        # v2v
        pred_smpl_path = os.path.join(pred_folder, "vis", name, "cape_ss.ply")
        pred_smpl_cham_path = os.path.join(pred_folder, "vis", name, "cape_ss_cham.ply")
        # assert os.path.isfile(pred_smpl_path)
        if not os.path.isfile(pred_smpl_path):
            continue
        # assert os.path.isfile(pred_smpl_cham_path)
        if not os.path.isfile(pred_smpl_cham_path):
            continue

        gt_smpl_path = os.path.join(gt_smpl_folder, name, f"mesh_smplx_{name}.obj")
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
        pred_smpl_cham_mesh = trimesh.load_mesh(
            pred_smpl_cham_path, maintain_order=True, process=False
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

        # gt_smpl_mesh.export(os.path.join(pred_folder, "vis", name, f"gt_smpl_mesh.obj"))
        # gt_scan_mesh.export(os.path.join(pred_folder, "vis", name, f"gt_scan_mesh.obj"))
        # pred_smpl_mesh.export(os.path.join(pred_folder, "vis", name, f"pred_smpl_mesh.obj"))
        # pred_smpl_cham_mesh.export(os.path.join(pred_folder, "vis", name, f"pred_smpl_cham_mesh.obj"))

        gt_smpl_verts = np.asarray(gt_smpl_mesh.vertices)
        pred_smpl_verts = np.asarray(pred_smpl_mesh.vertices)
        pred_smpl_cham_verts = np.asarray(pred_smpl_cham_mesh.vertices)

        v2v_error_all = np.linalg.norm(gt_smpl_verts - pred_smpl_verts, axis=1).mean()
        v2v_error_cham_all = np.linalg.norm(
            gt_smpl_verts - pred_smpl_cham_verts, axis=1
        ).mean()

        v2v_error_hands = np.linalg.norm(
            gt_smpl_verts[two_hands] - pred_smpl_verts[two_hands], axis=1
        ).mean()
        v2v_error_cham_hands = np.linalg.norm(
            gt_smpl_verts[two_hands] - pred_smpl_cham_verts[two_hands], axis=1
        ).mean()

        v2v_error_lhand = np.linalg.norm(
            gt_smpl_verts[left_hand] - pred_smpl_verts[left_hand], axis=1
        ).mean()
        v2v_error_cham_lhand = np.linalg.norm(
            gt_smpl_verts[left_hand] - pred_smpl_cham_verts[left_hand], axis=1
        ).mean()

        v2v_error_rhand = np.linalg.norm(
            gt_smpl_verts[right_hand] - pred_smpl_verts[right_hand], axis=1
        ).mean()
        v2v_error_cham_rhand = np.linalg.norm(
            gt_smpl_verts[right_hand] - pred_smpl_cham_verts[right_hand], axis=1
        ).mean()

        v2v_error_head = np.linalg.norm(
            gt_smpl_verts[head] - pred_smpl_verts[head], axis=1
        ).mean()
        v2v_error_cham_head = np.linalg.norm(
            gt_smpl_verts[head] - pred_smpl_cham_verts[head], axis=1
        ).mean()

        v2v_error_other = np.linalg.norm(
            gt_smpl_verts[body_others] - pred_smpl_verts[body_others], axis=1
        ).mean()
        v2v_error_cham_other = np.linalg.norm(
            gt_smpl_verts[body_others] - pred_smpl_cham_verts[body_others], axis=1
        ).mean()

        sum_v2v_error_hands += v2v_error_hands
        sum_v2v_error_cham_hands += v2v_error_cham_hands
        sum_v2v_error_lhand += v2v_error_lhand
        sum_v2v_error_cham_lhand += v2v_error_cham_lhand
        sum_v2v_error_rhand += v2v_error_rhand
        sum_v2v_error_cham_rhand += v2v_error_cham_rhand
        sum_v2v_error_head += v2v_error_head
        sum_v2v_error_cham_head += v2v_error_cham_head
        sum_v2v_error_other += v2v_error_other
        sum_v2v_error_cham_other += v2v_error_cham_other

        with open(v2v_file_hands, "a") as f:
            f.write(f"{name} {v2v_error_hands}\n")
        with open(v2v_file_cham_hands, "a") as f:
            f.write(f"{name} {v2v_error_cham_hands}\n")
        with open(v2v_file_lhand, "a") as f:
            f.write(f"{name} {v2v_error_lhand}\n")
        with open(v2v_file_cham_lhand, "a") as f:
            f.write(f"{name} {v2v_error_cham_lhand}\n")
        with open(v2v_file_rhand, "a") as f:
            f.write(f"{name} {v2v_error_rhand}\n")
        with open(v2v_file_cham_rhand, "a") as f:
            f.write(f"{name} {v2v_error_cham_rhand}\n")
        with open(v2v_file_head, "a") as f:
            f.write(f"{name} {v2v_error_head}\n")
        with open(v2v_file_cham_head, "a") as f:
            f.write(f"{name} {v2v_error_cham_head}\n")
        with open(v2v_file_other, "a") as f:
            f.write(f"{name} {v2v_error_other}\n")
        with open(v2v_file_cham_other, "a") as f:
            f.write(f"{name} {v2v_error_cham_other}\n")

        # print(v2v_error)
        # print(v2v_error_cham)
        with open(v2v_file_all, "a") as f:
            f.write(f"{name} {v2v_error_all}\n")
        with open(v2v_file_cham_all, "a") as f:
            f.write(f"{name} {v2v_error_cham_all}\n")
        sum_v2v_error_all += v2v_error_all
        sum_v2v_error_cham_all += v2v_error_cham_all

        # mpjpe
        # considered_joints_num = 22
        pred_info = np.load(
            os.path.join(
                pred_folder, "vis", name, "pred_smplx_info_before_cham_refine.npz"
            )
        )
        pred_info_cham = np.load(
            os.path.join(
                pred_folder, "vis", name, "pred_smplx_info_after_cham_refine.npz"
            )
        )
        pred_joints = pred_info["joints"]
        pred_joints_cham = pred_info_cham["joints"]
        # print(pred_joints.shape)

        # exit()

        # gt_info = np.load(os.path.join(gt_smpl_folder, name, f"info_{name}.npz"))
        # gt_info = pickle.load(
        #     open(os.path.join(gt_smpl_folder, name, f"smplx_params_{name}.pkl"), "rb")
        # )
        gt_joints = np.load(
            os.path.join(gt_smpl_folder, name, f"smplx_joints_{name}.npy")
        )
        gt_joints = gt_joints - gt_scan_center
        # gt_joints = gt_info['joints']

        mpjpe_error_all = np.linalg.norm(
            pred_joints[full_idxs] - gt_joints[full_idxs],
            axis=1,
        ).mean()
        mpjpe_error_cham_all = np.linalg.norm(
            pred_joints_cham[full_idxs] - gt_joints[full_idxs],
            axis=1,
        ).mean()

        mpjpe_error_hands = np.linalg.norm(
            pred_joints[two_hands_idxs] - gt_joints[two_hands_idxs],
            axis=1,
        ).mean()
        mpjpe_error_cham_hands = np.linalg.norm(
            pred_joints_cham[two_hands_idxs] - gt_joints[two_hands_idxs],
            axis=1,
        ).mean()

        mpjpe_error_lhand = np.linalg.norm(
            pred_joints[lhand_idxs] - gt_joints[lhand_idxs],
            axis=1,
        ).mean()
        mpjpe_error_cham_lhand = np.linalg.norm(
            pred_joints_cham[lhand_idxs] - gt_joints[lhand_idxs],
            axis=1,
        ).mean()

        mpjpe_error_rhand = np.linalg.norm(
            pred_joints[rhand_idxs] - gt_joints[rhand_idxs],
            axis=1,
        ).mean()
        mpjpe_error_cham_rhand = np.linalg.norm(
            pred_joints_cham[rhand_idxs] - gt_joints[rhand_idxs],
            axis=1,
        ).mean()
        mpjpe_error_head = np.linalg.norm(
            pred_joints[head_idxs] - gt_joints[head_idxs],
            axis=1,
        ).mean()
        mpjpe_error_cham_head = np.linalg.norm(
            pred_joints_cham[head_idxs] - gt_joints[head_idxs],
            axis=1,
        ).mean()

        mpjpe_error_other = np.linalg.norm(
            pred_joints[body_idxs] - gt_joints[body_idxs],
            axis=1,
        ).mean()
        mpjpe_error_cham_other = np.linalg.norm(
            pred_joints_cham[body_idxs] - gt_joints[body_idxs],
            axis=1,
        ).mean()

        # print(mpjpe_error_all)
        # print(mpjpe_error_cham_all)
        with open(mpjpe_file_all, "a") as f:
            f.write(f"{name} {mpjpe_error_all}\n")
        with open(mpjpe_file_cham_all, "a") as f:
            f.write(f"{name} {mpjpe_error_cham_all}\n")
        with open(mpjpe_file_hands, "a") as f:
            f.write(f"{name} {mpjpe_error_hands}\n")
        with open(mpjpe_file_cham_hands, "a") as f:
            f.write(f"{name} {mpjpe_error_cham_hands}\n")
        with open(mpjpe_file_lhand, "a") as f:
            f.write(f"{name} {mpjpe_error_lhand}\n")
        with open(mpjpe_file_cham_lhand, "a") as f:
            f.write(f"{name} {mpjpe_error_cham_lhand}\n")
        with open(mpjpe_file_rhand, "a") as f:
            f.write(f"{name} {mpjpe_error_rhand}\n")
        with open(mpjpe_file_cham_rhand, "a") as f:
            f.write(f"{name} {mpjpe_error_cham_rhand}\n")
        with open(mpjpe_file_head, "a") as f:
            f.write(f"{name} {mpjpe_error_head}\n")
        with open(mpjpe_file_cham_head, "a") as f:
            f.write(f"{name} {mpjpe_error_cham_head}\n")
        with open(mpjpe_file_other, "a") as f:
            f.write(f"{name} {mpjpe_error_other}\n")
        with open(mpjpe_file_cham_other, "a") as f:
            f.write(f"{name} {mpjpe_error_cham_other}\n")

        sum_mpjpe_error_all += mpjpe_error_all
        sum_mpjpe_error_cham_all += mpjpe_error_cham_all

        sum_mpjpe_error_hands += mpjpe_error_hands
        sum_mpjpe_error_cham_hands += mpjpe_error_cham_hands
        sum_mpjpe_error_lhand += mpjpe_error_lhand
        sum_mpjpe_error_cham_lhand += mpjpe_error_cham_lhand
        sum_mpjpe_error_rhand += mpjpe_error_rhand
        sum_mpjpe_error_cham_rhand += mpjpe_error_cham_rhand
        sum_mpjpe_error_head += mpjpe_error_head
        sum_mpjpe_error_cham_head += mpjpe_error_cham_head
        sum_mpjpe_error_other += mpjpe_error_other
        sum_mpjpe_error_cham_other += mpjpe_error_cham_other

        sample_num += 1

    print(
        "mean v2v error all: ",
        sum_v2v_error_all / sample_num,
        "sample num: ",
        sample_num,
    )
    print(
        "mean v2v error all with chamfer refine: ",
        sum_v2v_error_cham_all / sample_num,
        "sample num: ",
        sample_num,
    )
    print(
        "mean v2v error hands: ",
        sum_v2v_error_hands / sample_num,
        "sample num: ",
        sample_num,
    )
    print(
        "mean v2v error hands with chamfer refine: ",
        sum_v2v_error_cham_hands / sample_num,
        "sample num: ",
        sample_num,
    )
    print(
        "mean v2v error lhand: ",
        sum_v2v_error_lhand / sample_num,
        "sample num: ",
        sample_num,
    )
    print(
        "mean v2v error lhand with chamfer refine: ",
        sum_v2v_error_cham_lhand / sample_num,
        "sample num: ",
        sample_num,
    )
    print(
        "mean v2v error rhand: ",
        sum_v2v_error_rhand / sample_num,
        "sample num: ",
        sample_num,
    )
    print(
        "mean v2v error rhand with chamfer refine: ",
        sum_v2v_error_cham_rhand / sample_num,
        "sample num: ",
        sample_num,
    )
    print(
        "mean v2v error head: ",
        sum_v2v_error_head / sample_num,
        "sample num: ",
        sample_num,
    )
    print(
        "mean v2v error head with chamfer refine: ",
        sum_v2v_error_cham_head / sample_num,
        "sample num: ",
        sample_num,
    )
    print(
        "mean v2v error other: ",
        sum_v2v_error_other / sample_num,
        "sample num: ",
        sample_num,
    )
    print(
        "mean v2v error other with chamfer refine: ",
        sum_v2v_error_cham_other / sample_num,
        "sample num: ",
        sample_num,
    )
    print(
        "mean mpjpe error all: ",
        sum_mpjpe_error_all / sample_num,
        "sample num: ",
        sample_num,
    )
    print(
        "mean mpjpe error all with chamfer refine: ",
        sum_mpjpe_error_cham_all / sample_num,
        "sample num: ",
        sample_num,
    )
    print(
        "mean mpjpe error hands: ",
        sum_mpjpe_error_hands / sample_num,
        "sample num: ",
        sample_num,
    )
    print(
        "mean mpjpe error hands with chamfer refine: ",
        sum_mpjpe_error_cham_hands / sample_num,
        "sample num: ",
        sample_num,
    )
    print(
        "mean mpjpe error lhand: ",
        sum_mpjpe_error_lhand / sample_num,
        "sample num: ",
        sample_num,
    )
    print(
        "mean mpjpe error lhand with chamfer refine: ",
        sum_mpjpe_error_cham_lhand / sample_num,
        "sample num: ",
        sample_num,
    )
    print(
        "mean mpjpe error rhand: ",
        sum_mpjpe_error_rhand / sample_num,
        "sample num: ",
        sample_num,
    )
    print(
        "mean mpjpe error rhand with chamfer refine: ",
        sum_mpjpe_error_cham_rhand / sample_num,
        "sample num: ",
        sample_num,
    )
    print(
        "mean mpjpe error head: ",
        sum_mpjpe_error_head / sample_num,
        "sample num: ",
        sample_num,
    )
    print(
        "mean mpjpe error head with chamfer refine: ",
        sum_mpjpe_error_cham_head / sample_num,
        "sample num: ",
        sample_num,
    )
    print(
        "mean mpjpe error other: ",
        sum_mpjpe_error_other / sample_num,
        "sample num: ",
        sample_num,
    )
    print(
        "mean mpjpe error other with chamfer refine: ",
        sum_mpjpe_error_cham_other / sample_num,
        "sample num: ",
        sample_num,
    )
    with open(v2v_file_all, "a") as f:
        f.write(
            f"mean v2v error all: {sum_v2v_error_all / sample_num} sample num: {sample_num}\n"
        )
    with open(v2v_file_cham_all, "a") as f:
        f.write(
            f"mean v2v error all with chamfer refine: {sum_v2v_error_cham_all / sample_num} sample num: {sample_num}\n"
        )
    with open(v2v_file_hands, "a") as f:
        f.write(
            f"mean v2v error hands: {sum_v2v_error_hands / sample_num} sample num: {sample_num}\n"
        )
    with open(v2v_file_cham_hands, "a") as f:
        f.write(
            f"mean v2v error hands with chamfer refine: {sum_v2v_error_cham_hands / sample_num} sample num: {sample_num}\n"
        )
    with open(v2v_file_lhand, "a") as f:
        f.write(
            f"mean v2v error lhand: {sum_v2v_error_lhand / sample_num} sample num: {sample_num}\n"
        )
    with open(v2v_file_cham_lhand, "a") as f:
        f.write(
            f"mean v2v error lhand with chamfer refine: {sum_v2v_error_cham_lhand / sample_num} sample num: {sample_num}\n"
        )
    with open(v2v_file_rhand, "a") as f:
        f.write(
            f"mean v2v error rhand: {sum_v2v_error_rhand / sample_num} sample num: {sample_num}\n"
        )
    with open(v2v_file_cham_rhand, "a") as f:
        f.write(
            f"mean v2v error rhand with chamfer refine: {sum_v2v_error_cham_rhand / sample_num} sample num: {sample_num}\n"
        )
    with open(v2v_file_head, "a") as f:
        f.write(
            f"mean v2v error head: {sum_v2v_error_head / sample_num} sample num: {sample_num}\n"
        )
    with open(v2v_file_cham_head, "a") as f:
        f.write(
            f"mean v2v error head with chamfer refine: {sum_v2v_error_cham_head / sample_num} sample num: {sample_num}\n"
        )
    with open(v2v_file_other, "a") as f:
        f.write(
            f"mean v2v error other: {sum_v2v_error_other / sample_num} sample num: {sample_num}\n"
        )
    with open(v2v_file_cham_other, "a") as f:
        f.write(
            f"mean v2v error other with chamfer refine: {sum_v2v_error_cham_other / sample_num} sample num: {sample_num}\n"
        )
    with open(mpjpe_file_all, "a") as f:
        f.write(
            f"mean mpjpe error all: {sum_mpjpe_error_all / sample_num} sample num: {sample_num}\n"
        )
    with open(mpjpe_file_cham_all, "a") as f:
        f.write(
            f"mean mpjpe error all with chamfer refine: {sum_mpjpe_error_cham_all / sample_num} sample num: {sample_num}\n"
        )
    with open(mpjpe_file_hands, "a") as f:
        f.write(
            f"mean mpjpe error hands: {sum_mpjpe_error_hands / sample_num} sample num: {sample_num}\n"
        )
    with open(mpjpe_file_cham_hands, "a") as f:
        f.write(
            f"mean mpjpe error hands with chamfer refine: {sum_mpjpe_error_cham_hands / sample_num} sample num: {sample_num}\n"
        )
    with open(mpjpe_file_lhand, "a") as f:
        f.write(
            f"mean mpjpe error lhand: {sum_mpjpe_error_lhand / sample_num} sample num: {sample_num}\n"
        )
    with open(mpjpe_file_cham_lhand, "a") as f:
        f.write(
            f"mean mpjpe error lhand with chamfer refine: {sum_mpjpe_error_cham_lhand / sample_num} sample num: {sample_num}\n"
        )
    with open(mpjpe_file_rhand, "a") as f:
        f.write(
            f"mean mpjpe error rhand: {sum_mpjpe_error_rhand / sample_num} sample num: {sample_num}\n"
        )
    with open(mpjpe_file_cham_rhand, "a") as f:
        f.write(
            f"mean mpjpe error rhand with chamfer refine: {sum_mpjpe_error_cham_rhand / sample_num} sample num: {sample_num}\n"
        )
    with open(mpjpe_file_head, "a") as f:
        f.write(
            f"mean mpjpe error head: {sum_mpjpe_error_head / sample_num} sample num: {sample_num}\n"
        )
    with open(mpjpe_file_cham_head, "a") as f:
        f.write(
            f"mean mpjpe error head with chamfer refine: {sum_mpjpe_error_cham_head / sample_num} sample num: {sample_num}\n"
        )
    with open(mpjpe_file_other, "a") as f:
        f.write(
            f"mean mpjpe error other: {sum_mpjpe_error_other / sample_num} sample num: {sample_num}\n"
        )
    with open(mpjpe_file_cham_other, "a") as f:
        f.write(
            f"mean mpjpe error other with chamfer refine: {sum_mpjpe_error_cham_other / sample_num} sample num: {sample_num}\n"
        )
    os.rename(
        v2v_file_all,
        os.path.join(
            pred_folder, f"v2v_error_{sum_v2v_error_all / sample_num}_new.txt"
        ),
    )
    os.rename(
        v2v_file_cham_all,
        os.path.join(
            pred_folder, f"v2v_error_cham_{sum_v2v_error_cham_all / sample_num}_new.txt"
        ),
    )
    os.rename(
        v2v_file_hands,
        os.path.join(
            pred_folder, f"v2v_error_hands_{sum_v2v_error_hands / sample_num}_new.txt"
        ),
    )
    os.rename(
        v2v_file_cham_hands,
        os.path.join(
            pred_folder,
            f"v2v_error_cham_hands_{sum_v2v_error_cham_hands / sample_num}_new.txt",
        ),
    )
    os.rename(
        v2v_file_lhand,
        os.path.join(
            pred_folder, f"v2v_error_lhand_{sum_v2v_error_lhand / sample_num}_new.txt"
        ),
    )
    os.rename(
        v2v_file_cham_lhand,
        os.path.join(
            pred_folder,
            f"v2v_error_cham_lhand_{sum_v2v_error_cham_lhand / sample_num}_new.txt",
        ),
    )
    os.rename(
        v2v_file_rhand,
        os.path.join(
            pred_folder, f"v2v_error_rhand_{sum_v2v_error_rhand / sample_num}_new.txt"
        ),
    )
    os.rename(
        v2v_file_cham_rhand,
        os.path.join(
            pred_folder,
            f"v2v_error_cham_rhand_{sum_v2v_error_cham_rhand / sample_num}_new.txt",
        ),
    )
    os.rename(
        v2v_file_head,
        os.path.join(
            pred_folder, f"v2v_error_head_{sum_v2v_error_head / sample_num}_new.txt"
        ),
    )
    os.rename(
        v2v_file_cham_head,
        os.path.join(
            pred_folder, f"v2v_error_cham_head_{sum_v2v_error_cham_head / sample_num}_new.txt"
        ),
    )
    os.rename(
        v2v_file_other,
        os.path.join(
            pred_folder, f"v2v_error_other_{sum_v2v_error_other / sample_num}_new.txt"
        ),
    )
    os.rename(
        v2v_file_cham_other,
        os.path.join(
            pred_folder,
            f"v2v_error_cham_other_{sum_v2v_error_cham_other / sample_num}_new.txt",
        ),
    )
    os.rename(
        mpjpe_file_all,
        os.path.join(
            pred_folder, f"mpjpe_error_{sum_mpjpe_error_all / sample_num}_new.txt"
        ),
    )
    os.rename(
        mpjpe_file_cham_all,
        os.path.join(
            pred_folder,
            f"mpjpe_error_cham_{sum_mpjpe_error_cham_all / sample_num}_new.txt",
        ),
    )
    os.rename(
        mpjpe_file_hands,
        os.path.join(
            pred_folder,
            f"mpjpe_error_hands_{sum_mpjpe_error_hands / sample_num}_new.txt",
        ),
    )
    os.rename(
        mpjpe_file_cham_hands,
        os.path.join(
            pred_folder,
            f"mpjpe_error_cham_hands_{sum_mpjpe_error_cham_hands / sample_num}_new.txt",
        ),
    )
    os.rename(
        mpjpe_file_lhand,
        os.path.join(
            pred_folder,
            f"mpjpe_error_lhand_{sum_mpjpe_error_lhand / sample_num}_new.txt",
        ),
    )
    os.rename(
        mpjpe_file_cham_lhand,
        os.path.join(
            pred_folder,
            f"mpjpe_error_cham_lhand_{sum_mpjpe_error_cham_lhand / sample_num}_new.txt",
        ),
    )
    os.rename(
        mpjpe_file_rhand,
        os.path.join(
            pred_folder,
            f"mpjpe_error_rhand_{sum_mpjpe_error_rhand / sample_num}_new.txt",
        ),
    )
    os.rename(
        mpjpe_file_cham_rhand,
        os.path.join(
            pred_folder,
            f"mpjpe_error_cham_rhand_{sum_mpjpe_error_cham_rhand / sample_num}_new.txt",
        ),
    )
    os.rename(
        mpjpe_file_head,
        os.path.join(
            pred_folder,
            f"mpjpe_error_head_{sum_mpjpe_error_head / sample_num}_new.txt",
        ),
    )
    os.rename(
        mpjpe_file_cham_head,
        os.path.join(
            pred_folder,
            f"mpjpe_error_cham_head_{sum_mpjpe_error_cham_head / sample_num}_new.txt",
        ),
    )
    os.rename(
        mpjpe_file_other,
        os.path.join(
            pred_folder,
            f"mpjpe_error_other_{sum_mpjpe_error_other / sample_num}_new.txt",
        ),
    )
    os.rename(
        mpjpe_file_cham_other,
        os.path.join(
            pred_folder,
            f"mpjpe_error_cham_other_{sum_mpjpe_error_cham_other / sample_num}_new.txt",
        ),
    )


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
