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


def fit_smplx(
    smplx_model,
    in_points,
    gt_idxs,
    steps_stage0=30,
    steps_stage1=50,
    lr_stage0=5e-1,
    lr_stage1=2e-1,
):
    print("Start fitting SMPLX")
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
        (
            body_pose,
            lhand_pose,
            rhand_pose,
            jaw_pose,
            leye_pose,
            reye_pose,
            expression_optimized,
            shape_optimized,
            global_orient,
            translation,
        ) = optim_vars
        pred_markers_position = aux_vars[0]

        batch_size = shape_optimized.tensor.shape[0]
        shape_frozen = torch.zeros((batch_size, 10 - 2)).to(torch.device("cuda"))
        expression_frozen = torch.zeros((batch_size, 10 - 2)).to(torch.device("cuda"))

        # forward
        smplx_output = smplx_model(
            global_orient=global_orient.tensor,
            body_pose=body_pose.tensor,
            betas=torch.cat([shape_optimized.tensor, shape_frozen], dim=1),
            expression=torch.cat(
                [expression_optimized.tensor, expression_frozen], dim=1
            ),
            jaw_pose=jaw_pose.tensor,
            leye_pose=leye_pose.tensor,
            reye_pose=reye_pose.tensor,
            left_hand_pose=lhand_pose.tensor,
            right_hand_pose=rhand_pose.tensor,
            transl=translation.tensor,
            return_verts=True,
        )
        smplx_vertices = smplx_output.vertices  # shape(B, V, 3)

        marker_vindices = (
            torch.tensor(list(gt_idxs), device=smplx_vertices.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        forwarded_markers_position = torch.gather(
            smplx_vertices, 1, marker_vindices.unsqueeze(-1).expand(-1, -1, 3)
        )  # shape(B, num_markers, 3)

        err = pred_markers_position.tensor - forwarded_markers_position
        err = err.reshape(batch_size, -1)
        # print(err.shape)

        return err

    def marker_error_fn_1(optim_vars, aux_vars):
        (
            body_pose,
            lhand_pose,
            rhand_pose,
            jaw_pose,
            leye_pose,
            reye_pose,
            expression,
            shape,
            global_orient,
            translation,
        ) = optim_vars
        pred_markers_position = aux_vars[0]

        batch_size = shape.tensor.shape[0]

        # forward
        smplx_output = smplx_model(
            global_orient=global_orient.tensor,
            body_pose=body_pose.tensor,
            betas=shape.tensor,
            expression=expression.tensor,
            jaw_pose=jaw_pose.tensor,
            leye_pose=leye_pose.tensor,
            reye_pose=reye_pose.tensor,
            left_hand_pose=lhand_pose.tensor,
            right_hand_pose=rhand_pose.tensor,
            transl=translation.tensor,
            return_verts=True,
        )
        smplx_vertices = smplx_output.vertices  # shape(B, V, 3)

        marker_vindices = (
            torch.tensor(list(gt_idxs), device=smplx_vertices.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        forwarded_markers_position = torch.gather(
            smplx_vertices, 1, marker_vindices.unsqueeze(-1).expand(-1, -1, 3)
        )  # shape(B, num_markers, 3)

        err = pred_markers_position.tensor - forwarded_markers_position
        err = err.reshape(batch_size, -1)

        return err

    # STAGE 0: ONLY OPTIMIZE POSE AND TOP BETAS
    print("Optimization stage 0:")

    # Initialize optimization variables
    body_pose = torch.zeros((B, smplx_model.NUM_BODY_JOINTS * 3)).to(
        torch.device("cuda")
    )
    lhand_pose = torch.zeros((B, smplx_model.NUM_HAND_JOINTS * 3)).to(
        torch.device("cuda")
    )
    rhand_pose = torch.zeros((B, smplx_model.NUM_HAND_JOINTS * 3)).to(
        torch.device("cuda")
    )
    jaw_pose = torch.zeros((B, 1 * 3)).to(torch.device("cuda"))
    leye_pose = torch.zeros((B, 1 * 3)).to(torch.device("cuda"))
    reye_pose = torch.zeros((B, 1 * 3)).to(torch.device("cuda"))
    expression_optimized = torch.zeros((B, 2)).to(torch.device("cuda"))
    shape_optimized = torch.zeros((B, 2)).to(torch.device("cuda"))
    global_orient = torch.zeros((B, 3)).to(torch.device("cuda"))
    translation = torch.zeros((B, 3)).to(torch.device("cuda"))

    body_pose = th.Vector(tensor=body_pose, name="body_pose")
    lhand_pose = th.Vector(tensor=lhand_pose, name="lhand_pose")
    rhand_pose = th.Vector(tensor=rhand_pose, name="rhand_pose")
    jaw_pose = th.Vector(tensor=jaw_pose, name="jaw_pose")
    leye_pose = th.Vector(tensor=leye_pose, name="leye_pose")
    reye_pose = th.Vector(tensor=reye_pose, name="reye_pose")
    expression_optimized = th.Vector(
        tensor=expression_optimized, name="expression_optimized"
    )
    shape_optimized = th.Vector(tensor=shape_optimized, name="shape_optimized")
    global_orient = th.Vector(tensor=global_orient, name="global_orient")
    translation = th.Vector(tensor=translation, name="translation")

    pred_markers_position = th.Variable(
        tensor=pred_markers_position, name="pred_markers_position"
    )

    optim_vars = [
        body_pose,
        lhand_pose,
        rhand_pose,
        jaw_pose,
        leye_pose,
        reye_pose,
        expression_optimized,
        shape_optimized,
        global_orient,
        translation,
    ]
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
        "body_pose": body_pose,
        "lhand_pose": lhand_pose,
        "rhand_pose": rhand_pose,
        "jaw_pose": jaw_pose,
        "leye_pose": leye_pose,
        "reye_pose": reye_pose,
        "expression_optimized": expression_optimized,
        "shape_optimized": shape_optimized,
        "global_orient": global_orient,
        "translation": translation,
        "pred_markers_position": pred_markers_position,
    }
    # monitor_memory()

    updated_inputs, _ = theseus_layer.forward(
        theseus_inputs, optimizer_kwargs={"verbose": False, "damping": 0.01}
    )  # TODO: damping = ??

    body_pose = updated_inputs["body_pose"]
    lhand_pose = updated_inputs["lhand_pose"]
    rhand_pose = updated_inputs["rhand_pose"]
    jaw_pose = updated_inputs["jaw_pose"]
    leye_pose = updated_inputs["leye_pose"]
    reye_pose = updated_inputs["reye_pose"]
    expression_optimized = updated_inputs["expression_optimized"]
    shape_optimized = updated_inputs["shape_optimized"]
    global_orient = updated_inputs["global_orient"]
    translation = updated_inputs["translation"]

    # STAGE 1: OPTIMIZE POSE AND ALL BETAS
    print("Optimization stage 1:")

    body_pose = body_pose.detach()
    lhand_pose = lhand_pose.detach()
    rhand_pose = rhand_pose.detach()
    jaw_pose = jaw_pose.detach()
    leye_pose = leye_pose.detach()
    reye_pose = reye_pose.detach()
    shape_frozen = torch.zeros((B, 10 - 2)).to(torch.device("cuda"))
    expression_frozen = torch.zeros((B, 10 - 2)).to(torch.device("cuda"))
    shape = torch.cat([shape_optimized, shape_frozen], dim=1).detach()
    expression = torch.cat([expression_optimized, expression_frozen], dim=1).detach()
    global_orient = global_orient.detach()
    translation = translation.detach()

    body_pose = th.Vector(tensor=body_pose, name="body_pose")
    lhand_pose = th.Vector(tensor=lhand_pose, name="lhand_pose")
    rhand_pose = th.Vector(tensor=rhand_pose, name="rhand_pose")
    jaw_pose = th.Vector(tensor=jaw_pose, name="jaw_pose")
    leye_pose = th.Vector(tensor=leye_pose, name="leye_pose")
    reye_pose = th.Vector(tensor=reye_pose, name="reye_pose")
    expression = th.Vector(tensor=expression, name="expression")
    shape = th.Vector(tensor=shape, name="shape")
    global_orient = th.Vector(tensor=global_orient, name="global_orient")
    translation = th.Vector(tensor=translation, name="translation")

    optim_vars = [
        body_pose,
        lhand_pose,
        rhand_pose,
        jaw_pose,
        leye_pose,
        reye_pose,
        expression,
        shape,
        global_orient,
        translation,
    ]
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
        "body_pose": body_pose,
        "lhand_pose": lhand_pose,
        "rhand_pose": rhand_pose,
        "jaw_pose": jaw_pose,
        "leye_pose": leye_pose,
        "reye_pose": reye_pose,
        "expression": expression,
        "shape": shape,
        "global_orient": global_orient,
        "translation": translation,
        "pred_markers_position": pred_markers_position,
    }

    updated_inputs, _ = theseus_layer.forward(
        theseus_inputs, optimizer_kwargs={"verbose": False}
    )  # TODO: damping = ??

    body_pose = updated_inputs["body_pose"]
    lhand_pose = updated_inputs["lhand_pose"]
    rhand_pose = updated_inputs["rhand_pose"]
    jaw_pose = updated_inputs["jaw_pose"]
    leye_pose = updated_inputs["leye_pose"]
    reye_pose = updated_inputs["reye_pose"]
    expression = updated_inputs["expression"]
    shape = updated_inputs["shape"]
    global_orient = updated_inputs["global_orient"]
    translation = updated_inputs["translation"]

    # get final smpl meshes
    smplx_output = smplx_model(
        global_orient=global_orient,
        body_pose=body_pose,
        betas=shape,
        expression=expression,
        jaw_pose=jaw_pose,
        leye_pose=leye_pose,
        reye_pose=reye_pose,
        left_hand_pose=lhand_pose,
        right_hand_pose=rhand_pose,
        transl=translation,
        return_verts=True,
    )
    joints = smplx_output.joints  # shape(B, J, 3)

    final_mesh_list = []
    for b in range(B):
        final_smpl_mesh = trimesh.Trimesh(
            smplx_output.vertices[b].detach().cpu().numpy(),
            smplx_model.faces,
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

    output_smplx_info = {}
    output_smplx_info["pose"] = torch.nn.Parameter(
        torch.cat(
            [
                global_orient,
                body_pose,
                jaw_pose,
                leye_pose,
                reye_pose,
                lhand_pose,
                rhand_pose,
            ],
            dim=1,
        )
    )
    output_smplx_info["beta"] = torch.nn.Parameter(shape)
    output_smplx_info["expression"] = torch.nn.Parameter(expression)
    # output_smpl_info["global_orient"] = global_orient.detach().cpu().numpy()
    output_smplx_info["trans"] = torch.nn.Parameter(translation)
    output_smplx_info["joints"] = joints
    # shape(B, 23, 3), shape(B, 10), shape(B, 3), shape(B, 3), shape(B, 45, 3)

    return (
        final_mesh_list[0].vertices,
        # pred_markers_position.tensor,
        output_smplx_info,
    )


if __name__ == "__main__":
    pass
