import os
import sys
from collections import OrderedDict
from functools import partial
from types import MappingProxyType

import numpy as np
import torch
from tqdm import tqdm

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from util import blender_utils, dataset_mixamo
from util.blender_utils import bpy as bpy
from util.dataset_mixamo import PoseData, get_connected_idx_pairs, get_kinematic_tree

ADDITIONAL_JOINTS = (
    "mixamorig:LRabbitEar2",
    "mixamorig:RRabbitEar2",
    "mixamorig:FoxTail1",
    "mixamorig:FoxTail2",
    "mixamorig:FoxTail3",
    "mixamorig:FoxTail4",
    "mixamorig:FoxTail5",
)
MIXAMO_JOINTS = dataset_mixamo.MIXAMO_JOINTS + ADDITIONAL_JOINTS
JOINTS_NUM = len(MIXAMO_JOINTS)
BONES_IDX_DICT = OrderedDict({name: i for i, name in enumerate(MIXAMO_JOINTS)})
BONES_IDX_DICT = MappingProxyType(BONES_IDX_DICT)

CONNECTED_BONES = list(dataset_mixamo.CONNECTED_BONES)
CONNECTED_BONES.append(
    (
        "FoxTail1",
        "FoxTail2",
        "FoxTail3",
        "FoxTail4",
        "FoxTail5",
    )
)
CONNECTED_BONES = tuple(CONNECTED_BONES)
CONNECTED_IDX_PAIRS = get_connected_idx_pairs(CONNECTED_BONES, BONES_IDX_DICT)

TEMPLATE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/Mixamo/bones_vroid.fbx"))
KINEMATIC_TREE = get_kinematic_tree(TEMPLATE_PATH, BONES_IDX_DICT)
assert len(KINEMATIC_TREE) == len(MIXAMO_JOINTS)


class MixamoDataset(dataset_mixamo.MixamoDataset):

    def __init__(
        self,
        data_dir: str,
        character_subdir="character_vroid_refined",
        retarget=True,
        bones_idx_dict: dict[str, int] = BONES_IDX_DICT,
        **xargs,
    ):
        super().__init__(
            data_dir=data_dir,
            character_subdir=character_subdir,
            retarget=retarget,
            bones_idx_dict=bones_idx_dict,
            **xargs,
        )
        self.animated_dir = os.path.join(data_dir, "animated_vroid")
        if os.path.isdir(self.animated_dir):
            print(f"Using pre-animated files in {self.animated_dir}.")
            self.load_fn = self.load_from_animated

    def load_from_animated(self, char_file: str, anim_file: str, *args, **xargs):
        get_base_name = lambda s: os.path.splitext(os.path.basename(s))[0]
        animated_filename = f"{get_base_name(char_file)}-{get_base_name(anim_file)}.fbx"
        animated_filepath = os.path.abspath(os.path.join(self.animated_dir, animated_filename))
        assert os.path.isfile(animated_filepath), f"Animation file not found: {animated_filepath}"
        return blender_utils.load_file(animated_filepath)


connect_loss_fn = partial(dataset_mixamo.connect_loss_fn, connected_idx_pairs=CONNECTED_IDX_PAIRS)


if __name__ == "__main__":
    import json

    from pytorch3d.transforms import Rotate, Transform3d, Translate, quaternion_to_matrix, random_rotations
    from torch.utils.data import DataLoader

    from util.dataset_mixamo import collate, seed_worker
    from util.utils import fix_random, get_normalize_transform

    fix_random()
    dataset = MixamoDataset(
        "../data/Mixamo",
        sample_frames=1,
        sample_points=32768,
        hands_resample_ratio=0.5,
        sample_vertices=-1,
        include_rest=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collate,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )
    print(len(dataset))

    # Test transformation matrix
    assert dataset.sample_vertices == -1 and dataloader.batch_size == 1
    for i, data in enumerate(tqdm(dataloader, dynamic_ncols=True)):
        data: PoseData
        meta = data.meta

        rest_joints, posed_joints = data.rest_joints[-1], data.joints[-1]
        rest_verts, posed_verts = data.rest_verts[-1], data.verts[-1]
        faces = meta.faces[-1]
        weights = data.weights[-1]
        pose = data.joints_transform[-1]
        if not data.weights_mask.all():
            continue

        # pose1 = compose_transform(decompose_transform(pose, return_quat=True, return_concat=True))
        # assert torch.allclose(pose, pose1, atol=1e-5)
        # pose2 = compose_transform(decompose_transform(pose.numpy(), return_quat=True, return_concat=True))
        # assert np.allclose(pose.numpy(), pose2, atol=1e-5)
        # pose_decompose1 = decompose_transform(pose, return_quat=True, return_concat=True)
        # pose_decompose2 = torch.from_numpy(decompose_transform(pose.numpy(), return_quat=True, return_concat=True))
        # # import bpy, mathutils
        # # pose_decompose3 = torch.from_numpy(np.concatenate(mathutils.Matrix(pose[0].numpy()).decompose(), -1))

        pose_inv_decomposed = data.joints_transform_inv_decomposed[-1]
        transl = pose_inv_decomposed[..., :3]
        rotation = pose_inv_decomposed[..., 3:-3]
        scaling = pose_inv_decomposed[..., -3:]
        assert torch.allclose(scaling, torch.ones_like(scaling), atol=1e-5)
        pose_transform_inv = Rotate(quaternion_to_matrix(rotation).transpose(-1, -2)).compose(Translate(transl))
        assert torch.allclose(pose_transform_inv.get_matrix().transpose(-1, -2), pose.inverse(), atol=1e-5)

        # rotate = Rotate(R=random_rotations(len(data)))
        # norm = get_normalize_transform(rotate.transform_points(data.pts), keep_ratio=True)
        # transform = rotate.compose(norm)
        # transform_decomposed = decompose_transform(
        #     transform.get_matrix().transpose(-1, -2), return_quat=True, return_concat=True
        # )
        # assert torch.allclose(transform_decomposed[:, -3], transform_decomposed[:, -2]) and torch.allclose(
        #     transform_decomposed[:, -2], transform_decomposed[:, -1]
        # )  # keep_ratio=True

        # posed_joints = transform[-1].transform_points(posed_joints)
        # posed_verts = transform[-1].transform_points(posed_verts)
        # pose = transform[-1].get_matrix().transpose(-1, -2) @ pose

        # LBS manually
        # ! PyTorch3D wants a right-multiplied transformation matrix (applied on row vectors) <https://pytorch3d.readthedocs.io/en/latest/modules/transforms.html#pytorch3d.transforms.Transform3d>
        pose_transform = Transform3d(matrix=pose.transpose(-1, -2))
        posed_joints_ = pose_transform.transform_points(rest_joints.unsqueeze(1)).squeeze(1)
        tqdm.write(f"{(posed_joints - posed_joints_).abs().mean()}")
        # trimesh.Trimesh(posed_joints, process=False, maintain_order=True).export("posed_joints_gt.ply")
        # trimesh.Trimesh(posed_joints_, process=False, maintain_order=True).export("posed_joints.ply")
        rest_joints_ = pose_transform.inverse().transform_points(posed_joints.unsqueeze(1)).squeeze(1)
        tqdm.write(f"{(rest_joints - rest_joints_).abs().mean()}")
        # trimesh.Trimesh(rest_joints, process=False, maintain_order=True).export("rest_joints_gt.ply")
        # trimesh.Trimesh(rest_joints_, process=False, maintain_order=True).export("rest_joints.ply")

        verts_transform = Transform3d(matrix=torch.einsum("kij,nk->nij", pose, weights).transpose(-1, -2))
        posed_verts_ = verts_transform.transform_points(rest_verts.unsqueeze(1)).squeeze(1)
        tqdm.write(f"{(posed_verts - posed_verts_).abs().mean()}")
        # trimesh.Trimesh(posed_verts, faces, process=False, maintain_order=True).export("posed_verts_gt.ply")
        # trimesh.Trimesh(posed_verts_, faces, process=False, maintain_order=True).export("posed_verts.ply")
        rest_verts_ = verts_transform.inverse().transform_points(posed_verts.unsqueeze(1)).squeeze(1)
        tqdm.write(f"{(rest_verts - rest_verts_).abs().mean()}")
        # trimesh.Trimesh(rest_verts, faces, process=False, maintain_order=True).export("rest_verts_gt.ply")
        # trimesh.Trimesh(rest_verts_, faces, process=False, maintain_order=True).export("rest_verts.ply")

    # Count vertices of all characters
    verts_num = {}
    for i, data in enumerate(tqdm(dataloader, dynamic_ncols=True)):
        meta, verts = data.meta, data.verts
        verts_num[meta.char_id[0]] = verts.shape[1]
        if i == len(dataset.character_list) - 1:
            break
    print(np.min(list(verts_num.values())))
    print(np.max(list(verts_num.values())))
    print(np.mean(list(verts_num.values())))
    with open("./character_verts.json", "w") as f:
        json.dump(verts_num, f, indent=4, ensure_ascii=False)

    # Count frames of all animations
    dataset.same_animation_first = False
    frame_num = {}
    for i, data in enumerate(tqdm(dataloader, dynamic_ncols=True)):
        meta = data.meta
        frame_num[meta.anim_id[0]] = len(meta.keyframes[0])
        if i == len(dataset.animation_list) - 1:
            break
    print(np.min(list(frame_num.values())))
    print(np.max(list(frame_num.values())))
    print(np.mean(list(frame_num.values())))
    with open("./animation_frames.json", "w") as f:
        json.dump(frame_num, f, indent=4, ensure_ascii=False)
