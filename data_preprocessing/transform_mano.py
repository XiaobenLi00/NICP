import numpy as np
from multiprocessing import Pool
import torch
import trimesh
import os
from tqdm import tqdm
import json
import smplx
mano_root_path = "/home/lixiaoben/projects/NICP/datafolder_new/body_models"
mano_layer = {'right': smplx.create(mano_root_path, 'mano', is_rhand=True, use_pca=False, flat_hand_mean=False).cuda(), 'left': smplx.create(mano_root_path, 'mano', is_rhand=False, use_pca=False, flat_hand_mean=False).cuda()}
if torch.sum(torch.abs(mano_layer['left'].shapedirs[:,0,:] - mano_layer['right'].shapedirs[:,0,:])) < 1:
    print('Fix shapedirs bug of MANO')
    mano_layer['left'].shapedirs[:,0,:] *= -1


def process_single_task(task):
    id, idx, tgt_dir = task
    with open(os.path.join("/home/lixiaoben/projects/NICP/datafolder_new/interhand", id), 'r') as f:
        mano_param = json.load(f)
    pose = torch.FloatTensor(mano_param['pose']).view(1,-1).cuda()
    shape = torch.FloatTensor(mano_param['shape']).view(1,-1).cuda()
    trans = torch.FloatTensor(mano_param['trans']).view(1,-1).cuda()
    if id.endswith('_left.json'):
        pose = pose.reshape(1, -1, 3)
        pose[:,:,1:] *= -1
        pose = pose.reshape(1, -1)
        trans[:,0] *= -1
    #     with torch.no_grad():
    #         output = mano_layer['left'](hand_pose=pose[:,3:], betas=shape)
    # else:
    with torch.no_grad():
        output = mano_layer['right'](hand_pose=pose[:,3:], betas=shape)
    vertices = output.vertices[0].cpu().numpy()
    faces = mano_layer['right'].faces
    mesh = trimesh.Trimesh(vertices, faces)
    mesh.export(os.path.join(tgt_dir, f"{str(idx).zfill(5)}.ply"))

def main():
    torch.multiprocessing.set_start_method("spawn", force=True)
    src_dir = "/home/lixiaoben/projects/NICP/datafolder_new/interhand"
    tgt_dir = "/home/lixiaoben/projects/NICP/datafolder_new/interhand/selected_meshes"
    ids = np.load(os.path.join(src_dir, "intercap_ids.npy"))
    # print(ids[0:10])
    tasks = []
    for idx, id in enumerate(sorted(ids)):
        tasks.append((id, idx, tgt_dir))
    # print(tasks[0:10])
    # exit()
    num_processes = 32
    with Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap(process_single_task, tasks), total=len(tasks)))
if __name__ == "__main__":
    main()