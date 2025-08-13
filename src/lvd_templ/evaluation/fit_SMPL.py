import trimesh
from smplx import SMPL
import torch
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import DBSCAN
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
import os
import theseus as th


def monitor_memory():
    """监控显存使用"""
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")


def fit_smpl(
    smpl_model,
    in_points,
    gt_idxs,
    steps_stage0=30,
    steps_stage1=50,
    lr_stage0=5e-1,
    lr_stage1=2e-1,
):
    print("Start fitting SMPL")
    # in_points = in_points[::10]  # (69, 3)
    # gt_idxs = gt_idxs[::10]  # (69,)
    # print(in_points.shape)
    # print(gt_idxs)
    # print(len(gt_idxs))

    B = 1
    pred_markers_position = (
        torch.from_numpy(in_points)
        .to(torch.device("cuda"), dtype=torch.float32)
        .unsqueeze(0)
    )  # (1, 69, 3)
    # print(pred_markers_position.shape)
    loss_weights = {
        "marker_loss": 1.0,
        # "mean_shape_loss": 1e-2 * 10 ** 0,
        # "point_mesh_distance": 1.0 * 10 ** 2,
        # "part_pmdistance": 1.0,
        # "pose_prior_loss": 1e-7 * 10 ** 0,
    }

    def marker_error_fn_0(optim_vars, aux_vars):
        pose, shape_optimized, global_orient, translation = optim_vars
        pred_markers_position = aux_vars[0]

        batch_size = shape_optimized.tensor.shape[0]
        shape_frozen = torch.zeros((batch_size, smpl_model.num_betas - 2)).to(
            torch.device("cuda")
        )

        # forward
        smpl_output = smpl_model(
            global_orient=global_orient.tensor,
            body_pose=pose.tensor,
            betas=torch.cat([shape_optimized.tensor, shape_frozen], dim=1),
            transl=translation.tensor,
            return_verts=True,
        )
        smpl_vertices = smpl_output.vertices  # shape(B, V, 3)

        marker_vindices = (
            torch.tensor(list(gt_idxs), device=smpl_vertices.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        forwarded_markers_position = torch.gather(
            smpl_vertices, 1, marker_vindices.unsqueeze(-1).expand(-1, -1, 3)
        )  # shape(B, num_markers, 3)

        err = pred_markers_position.tensor - forwarded_markers_position
        err = err.reshape(batch_size, -1)
        # print(err.shape)

        return err

    def marker_error_fn_1(optim_vars, aux_vars):
        pose, shape, global_orient, translation = optim_vars
        pred_markers_position = aux_vars[0]

        batch_size = shape.tensor.shape[0]

        # forward
        smpl_output = smpl_model(
            global_orient=global_orient.tensor,
            body_pose=pose.tensor,
            betas=shape.tensor,
            transl=translation.tensor,
            return_verts=True,
        )
        smpl_vertices = smpl_output.vertices  # shape(B, V, 3)

        marker_vindices = (
            torch.tensor(list(gt_idxs), device=smpl_vertices.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        forwarded_markers_position = torch.gather(
            smpl_vertices, 1, marker_vindices.unsqueeze(-1).expand(-1, -1, 3)
        )  # shape(B, num_markers, 3)

        err = pred_markers_position.tensor - forwarded_markers_position
        err = err.reshape(batch_size, -1)

        return err

    # STAGE 0: ONLY OPTIMIZE POSE AND TOP BETAS
    print("Optimization stage 0:")

    # Initialize optimization variables
    pose = torch.zeros((B, smpl_model.NUM_BODY_JOINTS * 3)).to(torch.device("cuda"))
    shape_optimized = torch.zeros((B, 2)).to(torch.device("cuda"))
    global_orient = torch.zeros((B, 3)).to(torch.device("cuda"))
    translation = torch.zeros((B, 3)).to(torch.device("cuda"))

    pose = th.Vector(tensor=pose, name="pose")
    shape_optimized = th.Vector(tensor=shape_optimized, name="shape_optimized")
    global_orient = th.Vector(tensor=global_orient, name="global_orient")
    translation = th.Vector(tensor=translation, name="translation")

    pred_markers_position = th.Variable(
        tensor=pred_markers_position, name="pred_markers_position"
    )

    optim_vars = [pose, shape_optimized, global_orient, translation]
    aux_vars = [pred_markers_position]

    w_marker = th.ScaleCostWeight(loss_weights["marker_loss"])
    # monitor_memory()
    marker_cost_function = th.AutoDiffCostFunction(
        optim_vars,
        marker_error_fn_0,
        len(gt_idxs) * 3,
        cost_weight=w_marker,
        aux_vars=aux_vars,
        name="marker_cost_function",
    )
    # monitor_memory()
    objective = th.Objective().to(torch.device("cuda"))
    objective.add(marker_cost_function)

    optimizer = th.LevenbergMarquardt(
        objective, max_iterations=steps_stage0, step_size=lr_stage0
    )
    # monitor_memory()
    # optimizer = th.GaussNewton(objective, max_iterations=steps_stage0, step_size=lr_stage0)
    theseus_layer = th.TheseusLayer(optimizer).to(torch.device("cuda"))

    theseus_inputs = {
        "pose": pose,
        "shape_optimized": shape_optimized,
        "global_orient": global_orient,
        "translation": translation,
        "pred_markers_position": pred_markers_position,
    }
    # monitor_memory()

    updated_inputs, _ = theseus_layer.forward(
        theseus_inputs, optimizer_kwargs={"verbose": False, "damping": 0.01}
    )  # TODO: damping = ??

    pose = updated_inputs["pose"]
    shape_optimized = updated_inputs["shape_optimized"]
    global_orient = updated_inputs["global_orient"]
    translation = updated_inputs["translation"]

    # STAGE 1: OPTIMIZE POSE AND ALL BETAS
    print("Optimization stage 1:")

    pose = pose.detach()
    shape_frozen = torch.zeros((B, smpl_model.num_betas - 2)).to(torch.device("cuda"))
    shape = torch.cat([shape_optimized, shape_frozen], dim=1).detach()
    global_orient = global_orient.detach()
    translation = translation.detach()

    pose = th.Vector(tensor=pose, name="pose")
    shape = th.Vector(tensor=shape, name="shape")
    global_orient = th.Vector(tensor=global_orient, name="global_orient")
    translation = th.Vector(tensor=translation, name="translation")

    optim_vars = [pose, shape, global_orient, translation]
    aux_vars = [pred_markers_position]

    w_marker = th.ScaleCostWeight(loss_weights["marker_loss"])
    marker_cost_function = th.AutoDiffCostFunction(
        optim_vars,
        marker_error_fn_1,
        len(gt_idxs) * 3,
        cost_weight=w_marker,
        aux_vars=aux_vars,
        name="marker_cost_function",
    )

    objective = th.Objective().to(torch.device("cuda"))
    objective.add(marker_cost_function)
    optimizer = th.LevenbergMarquardt(
        objective, max_iterations=steps_stage1, step_size=lr_stage1
    )
    theseus_layer = th.TheseusLayer(optimizer).to(torch.device("cuda"))

    theseus_inputs = {
        "pose": pose,
        "shape": shape,
        "global_orient": global_orient,
        "translation": translation,
        "pred_markers_position": pred_markers_position,
    }

    updated_inputs, _ = theseus_layer.forward(
        theseus_inputs, optimizer_kwargs={"verbose": False}
    )  # TODO: damping = ??

    pose = updated_inputs["pose"]
    shape = updated_inputs["shape"]
    global_orient = updated_inputs["global_orient"]
    translation = updated_inputs["translation"]

    # get final smpl meshes
    smpl_output = smpl_model(
        global_orient=global_orient,
        body_pose=pose,
        betas=shape,
        transl=translation,
        return_verts=True,
    )
    joints = smpl_output.joints  # shape(B, J, 3)

    final_mesh_list = []
    for b in range(B):
        final_smpl_mesh = trimesh.Trimesh(
            smpl_output.vertices[b].detach().cpu().numpy(),
            smpl_model.faces,
            process=False,
            maintain_order=True,
        )
        final_mesh_list.append(final_smpl_mesh)

    # output_smpl_info = [
    #     pose.detach().cpu().numpy().reshape(B, 23, 3),
    #     shape.detach().cpu().numpy(),
    #     global_orient.detach().cpu().numpy(),
    #     translation.detach().cpu().numpy(),
    #     joints.detach().cpu().numpy(),
    # ]

    output_smpl_info = {}
    output_smpl_info["pose"] = torch.nn.Parameter(
        torch.cat([global_orient, pose], dim=1)
    )
    output_smpl_info["beta"] = torch.nn.Parameter(shape)
    # output_smpl_info["global_orient"] = global_orient.detach().cpu().numpy()
    output_smpl_info["trans"] = torch.nn.Parameter(translation)
    output_smpl_info["joints"] = joints
    # shape(B, 23, 3), shape(B, 10), shape(B, 3), shape(B, 3), shape(B, 45, 3)

    return (
        final_mesh_list[0].vertices,
        # pred_markers_position.tensor,
        output_smpl_info,
    )


if __name__ == "__main__":
    pass
