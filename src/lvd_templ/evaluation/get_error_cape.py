import os
import glob
import numpy as np
import trimesh
from tqdm import tqdm 

# CAPE without chamfer refinement
# mean v2v error:  0.017251761702706554 sample num:  1021
# mean mpjpe error:  0.01051824787598793 sample num:  1021
def main():

    flag = 1 # 0: before cham_refine, 1: after cham_refine
    # mode = "before_cham_refine" if flag == 0 else "after_cham_refine"


    gt_smpl_folder = 'datafolder/CAPE_reorganized/cape_release/smpl_reorganized'
    gt_scan_folder = 'datafolder/CAPE_reorganized/cape_release/model_reorganized'
    pred_folder = 'output/matchAMASS_CAPE/cape_eq_pred_inner_points'

    # gt_smpl_folder = "datafolder/4D-DRESS/data_processed/smplh"
    # pred_folder = "/home/boqian/code/NICP/output/matchAMASS_4D-DRESS/demo"

    all_v2v_error = 0.0
    all_mpjpe_error = 0.0
    all_v2v_error_cham = 0.0
    all_mpjpe_error_cham = 0.0
    sample_num = 0
    v2v_file = os.path.join(pred_folder, 'v2v_error.txt')
    mpjpe_file = os.path.join(pred_folder, 'mpjpe_error.txt')
    v2v_file_cham = os.path.join(pred_folder, 'v2v_error_cham.txt')
    mpjpe_file_cham = os.path.join(pred_folder, 'mpjpe_error_cham.txt')
    if os.path.isfile(v2v_file):
        os.remove(v2v_file)
    if os.path.isfile(mpjpe_file):
        os.remove(mpjpe_file)
    if os.path.isfile(v2v_file_cham):
        os.remove(v2v_file_cham)
    if os.path.isfile(mpjpe_file_cham):
        os.remove(mpjpe_file_cham)
    for name in tqdm(os.listdir(pred_folder)):

        # v2v
        pred_smpl_path = os.path.join(pred_folder, name, "outcape_ss.ply")
        pred_smpl_cham_path = os.path.join(pred_folder, name, "outcape_ss_cham_0.ply")
        # assert os.path.isfile(pred_smpl_path)
        if not os.path.isfile(pred_smpl_path):
            continue
        # assert os.path.isfile(pred_smpl_cham_path)
        if not os.path.isfile(pred_smpl_cham_path):
            continue

        gt_smpl_path = os.path.join(gt_smpl_folder, name, f"mesh_smpl_{name}.obj")
        gt_scan_path = os.path.join(gt_scan_folder, name, f"{name}.obj")
        assert os.path.isfile(gt_smpl_path)
        assert os.path.isfile(gt_scan_path)

        gt_smpl_mesh = trimesh.load_mesh(gt_smpl_path, maintain_order=True, process=False)
        gt_scan_mesh = trimesh.load_mesh(gt_scan_path, maintain_order=True, process=False)
        pred_smpl_mesh = trimesh.load_mesh(pred_smpl_path, maintain_order=True, process=False)
        pred_smpl_cham_mesh = trimesh.load_mesh(pred_smpl_cham_path, maintain_order=True, process=False)

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
        pred_smpl_cham_verts = np.asarray(pred_smpl_cham_mesh.vertices)

        v2v_error = np.linalg.norm(gt_smpl_verts - pred_smpl_verts, axis=1).mean()
        v2v_error_cham = np.linalg.norm(gt_smpl_verts - pred_smpl_cham_verts, axis=1).mean()
        print(v2v_error)
        print(v2v_error_cham)
        with open(v2v_file, 'a') as f:
            f.write(f"{name} {v2v_error}\n")
        with open(v2v_file_cham, 'a') as f:
            f.write(f"{name} {v2v_error_cham}\n")
        all_v2v_error += v2v_error
        all_v2v_error_cham += v2v_error_cham

        # mpjpe 
        considered_joints_num = 22
        pred_info = np.load(os.path.join(pred_folder, name, "pred_smpl_info_before_cham_refine.npz"))
        pred_info_cham = np.load(os.path.join(pred_folder, name, "pred_smpl_info_after_cham_refine.npz"))
        pred_joints = pred_info['joints']
        pred_joints_cham = pred_info_cham['joints']

        gt_info = np.load(os.path.join(gt_smpl_folder, name, f"info_{name}.npz"))
        gt_joints = gt_info['joints'] - gt_scan_center

        mpjpe_error = np.linalg.norm(pred_joints[:considered_joints_num, :] - gt_joints[:considered_joints_num, :], axis=1).mean()
        mpjpe_error_cham = np.linalg.norm(pred_joints_cham[:considered_joints_num, :] - gt_joints[:considered_joints_num, :], axis=1).mean()
        print(mpjpe_error)
        print(mpjpe_error_cham)
        with open(mpjpe_file, 'a') as f:
            f.write(f"{name} {mpjpe_error}\n")
        with open(mpjpe_file_cham, 'a') as f:
            f.write(f"{name} {mpjpe_error_cham}\n")
        all_mpjpe_error += mpjpe_error
        all_mpjpe_error_cham += mpjpe_error_cham

    
        sample_num += 1

    print("mean v2v error: ", all_v2v_error / sample_num, "sample num: ", sample_num)
    print("mean mpjpe error: ", all_mpjpe_error / sample_num, "sample num: ", sample_num)
    print("mean v2v error with chamfer refine: ", all_v2v_error_cham / sample_num, "sample num: ", sample_num)
    print("mean mpjpe error with chamfer refine: ", all_mpjpe_error_cham / sample_num, "sample num: ", sample_num)
    with open(v2v_file, 'a') as f:
        f.write(f"mean v2v error: {all_v2v_error / sample_num} sample num: {sample_num}\n")
    with open(mpjpe_file, 'a') as f:
        f.write(f"mean mpjpe error: {all_mpjpe_error / sample_num} sample num: {sample_num}\n")
    with open(v2v_file_cham, 'a') as f:
        f.write(f"mean v2v error with chamfer refine: {all_v2v_error_cham / sample_num} sample num: {sample_num}\n")
    with open(mpjpe_file_cham, 'a') as f:
        f.write(f"mean mpjpe error with chamfer refine: {all_mpjpe_error_cham / sample_num} sample num: {sample_num}\n")
    os.rename(v2v_file, os.path.join(pred_folder, f'v2v_error_{all_v2v_error / sample_num}.txt'))
    os.rename(mpjpe_file, os.path.join(pred_folder, f'mpjpe_error_{all_mpjpe_error / sample_num}.txt'))
    os.rename(v2v_file_cham, os.path.join(pred_folder, f'v2v_error_cham_{all_v2v_error_cham / sample_num}.txt'))
    os.rename(mpjpe_file_cham, os.path.join(pred_folder, f'mpjpe_error_cham_{all_mpjpe_error_cham / sample_num}.txt'))
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