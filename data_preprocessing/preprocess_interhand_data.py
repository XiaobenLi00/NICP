# Automatic building of cython extensions
import numpy as np
import pyximport

# print(np.get_include())
# pyximport.install(
#     setup_args={
#         "include_dirs": [np.get_include(), "./utils/libvoxelize", "/home/boqian/code/NICP/training/data_preprocessing"],
#         "script_args": ["--cython-cplus"]
#     }, reload_support=True, language_level=3
# )

import argparse
import os

# import traceback
from pathlib import Path
import multiprocessing as mp
from multiprocessing import Pool

import torch
import trimesh
from tqdm import tqdm

# from utils.voxels import VoxelGrid
# from utils.parallel_map import parallel_map

## SET RESOLUTION OF THE VOXELIZATION
res = 64

# faces = np.load("./assets/faces.npy")


### VOXELIZATION UTILS  ####


def create_grid(
    resX,
    resY,
    resZ,
    b_min=np.array([0, 0, 0]),
    b_max=np.array([1, 1, 1]),
    transform=None,
):
    """
    Create a dense grid of given resolution and bounding box
    :param resX: resolution along X axis
    :param resY: resolution along Y axis
    :param resZ: resolution along Z axis
    :param b_min: vec3 (x_min, y_min, z_min) bounding box corner
    :param b_max: vec3 (x_max, y_max, z_max) bounding box corner
    :return: [3, resX, resY, resZ] coordinates of the grid, and transform matrix from mesh index
    """
    coords = np.mgrid[:resX, :resY, :resZ]
    coords = coords.reshape(3, -1)
    coords_matrix = np.eye(4)
    length = b_max - b_min
    coords_matrix[0, 0] = length[0] / resX
    coords_matrix[1, 1] = length[1] / resY
    coords_matrix[2, 2] = length[2] / resZ
    coords_matrix[0:3, 3] = b_min
    coords = np.matmul(coords_matrix[:3, :3], coords) + coords_matrix[:3, 3:4]
    if transform is not None:
        coords = np.matmul(transform[:3, :3], coords) + transform[:3, 3:4]
        coords_matrix = np.matmul(transform, coords_matrix)
    coords = coords.reshape(3, resX, resY, resZ)
    return coords, coords_matrix


def voxelize_distance(mesh, res, OUT_PATH_OCC, OUT_PATH_SCAL, id_):

    resolution = res  # Voxel resolution
    b_min = np.array([-0.08, -0.08, -0.08])
    b_max = np.array([0.08, 0.08, 0.08])
    step = 5000

    total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
    centers = (mesh.bounds[1] + mesh.bounds[0]) / 2
    mesh.apply_translation(-centers)
    mesh.apply_scale(1 / total_size)

    vertices = mesh.vertices
    factor = max(
        1, int(len(vertices) / 20000)
    )  # We will subsample vertices when there's too many in a scan !

    with torch.no_grad():
        v = torch.FloatTensor(vertices).cuda()
        coords, mat = create_grid(resolution, resolution, resolution, b_min, b_max)
        points = torch.FloatTensor(coords.reshape(3, -1)).transpose(1, 0).cuda()
        # points_npy = coords.reshape(3, -1).T
        iters = len(points) // step + 1

        all_distances = []
        for it in range(iters):
            it_v = points[it * step : (it + 1) * step]
            distance = ((it_v.unsqueeze(0) - v[::factor].unsqueeze(1)) ** 2).sum(-1)
            distance = distance.min(0)[0].cpu().data.numpy()
            all_distances.append(distance)
        signed_distance = np.concatenate(all_distances)
    del v
    del coords

    voxels = signed_distance.reshape(resolution, resolution, resolution)

    ## Save voxels and aligned vertices
    torch.save(voxels, OUT_PATH_OCC / str(f"{id_}.pt"))
    torch.save(vertices, OUT_PATH_SCAL / str(f"{id_}.pt"))


##############


# 定义处理单个任务的函数
def process_single_task(task):
    input_path, file, out_occ, out_scal = task
    id_ = file.replace(".ply", "")
    try:
        # smpl_mesh = trimesh.load_mesh(
        #     input_path / id_ / f"mesh_smpl_{id_}.obj",
        #     process=False,
        #     maintain_order=True,
        # )
        mano_mesh = trimesh.load_mesh(
            input_path / file, process=False, maintain_order=True
        )
        voxelize_distance(mano_mesh, res, out_occ, out_scal, id_)
    except Exception as e:
        print(f"Error processing {id_}: {str(e)}")


def main(cfg):

    # needed to support cuda in parallel_map
    torch.multiprocessing.set_start_method("spawn", force=True)

    # tqdm_bar.set_description(f"Processing {DATASET}")
    # INPUT_PATH = args.input_path / cfg.exp / 'stage_III' / DATASET / 'data_v.pt'
    # OUT_PATH_OCC = args.input_path / cfg.exp / 'stage_III' / DATASET / 'ifnet_indi' / "occ_dist"
    # OUT_PATH_SCAL = args.input_path / cfg.exp /'stage_III' / DATASET / 'ifnet_indi' / "verts_occ_dist"
    OUT_PATH_OCC = (
        args.output_path / "stage_III" / "default" / "ifnet_indi" / "occ_dist"
    )
    OUT_PATH_SCAL = (
        args.output_path / "stage_III" / "default" / "ifnet_indi" / "verts_occ_dist"
    )

    OUT_PATH_OCC.mkdir(parents=True, exist_ok=True)
    OUT_PATH_SCAL.mkdir(parents=True, exist_ok=True)

    #########
    # 准备任务列表
    tasks = []
    for file in os.listdir(args.input_path):
        # if os.path.isdir(args.input_path / file):
        tasks.append((args.input_path, file, OUT_PATH_OCC, OUT_PATH_SCAL))

    # 使用进程池并行处理
    num_processes = 32
    with Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap(process_single_task, tasks), total=len(tasks)))


#########
# for file in tqdm(os.listdir(args.input_path)):
#     if os.path.isdir(args.input_path / file):
#         id_ = file
#         smpl_mesh = trimesh.load_mesh(args.input_path / id_ / f'mesh_smpl_{id_}.obj', process=False, maintain_order=True)
#         voxelize_distance(smpl_mesh, res, OUT_PATH_OCC, OUT_PATH_SCAL, id_)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Data voxelization")

    # parser.add_argument("--exp", "-e", type=str, default='4D-DRESS', help="Experiment name")
    # parser.add_argument("--datasets", "-d", type=str, default='test', nargs="+", choices=["train", "vald", "test"], help="Dataset name")
    parser.add_argument(
        "--input_path",
        "-i",
        type=Path,
        default=Path(
            "/home/lixiaoben/projects/NICP/datafolder_new/interhand/selected_meshes/",
        ),
        help="Path to input folder",
    )  # /home/boqian/code/NICP/datafolder/4D-DRESS/data_processed/smplh/
    parser.add_argument(
        "--output_path",
        "-o",
        type=Path,
        default=Path("/home/lixiaoben/projects/NICP/datafolder_new/interhand/nicp_hand/"),
        help="Path to output folder",
    )  # /home/boqian/code/NICP/training/processed_data/4D-DRESS/
    # parser.add_argument("--jobs", "-j", type=int, default=4, help="Number of parallel jobs")

    args = parser.parse_args()

    main(args)
