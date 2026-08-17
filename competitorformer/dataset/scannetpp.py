import os.path as osp

import numpy as np
import torch
from torch_scatter import scatter_mean

from ..utils import Instances3D
from .scannetpp_constants import CLASS_LABELS_PP, INST_LABELS_PP
from .scannetv2 import ScanNetDataset, as_1d_label, load_npy_file


def _sem_to_inst_lut():
    lut = np.full(len(CLASS_LABELS_PP), -100, dtype=np.int64)
    for inst_id, name in enumerate(INST_LABELS_PP):
        lut[CLASS_LABELS_PP.index(name)] = inst_id
    return lut


SEM_TO_INST = _sem_to_inst_lut()


class ScanNetPPDataset(ScanNetDataset):
    CLASSES = INST_LABELS_PP
    # ScanNet++ uses 100-class segment ids mapped to 84 things; do not apply ScanNet v2 label-2.
    inst_stuff_remap = False

    def load(self, scene_dir):
        xyz = np.load(osp.join(scene_dir, 'coord.npy'))
        rgb = np.load(osp.join(scene_dir, 'color.npy'))
        superpoint = load_npy_file(scene_dir, ('superpoint.npy',))
        if superpoint is None:
            raise FileNotFoundError(f'superpoint.npy not found in {scene_dir}')
        superpoint = np.asarray(superpoint).reshape(-1)

        rgb = rgb.astype(np.float32)
        if rgb.max() > 1.5:
            rgb = rgb / 127.5 - 1.0

        if self.with_normals:
            normal = load_npy_file(scene_dir, ('normal.npy',))
            if normal is None:
                print(f'Warning: Normal file not found for {scene_dir}')
            else:
                normal = normal.astype(np.float32)
        else:
            normal = None

        if self.with_label:
            sem_label = as_1d_label(load_npy_file(scene_dir, ('semantic_label.npy', 'segment.npy')))
            inst_label = as_1d_label(load_npy_file(scene_dir, ('instance_label.npy', 'instance.npy')))
            if sem_label is None or inst_label is None:
                print(f'Warning: Label files not found for {scene_dir}, using dummy labels')
                sem_label = np.full(xyz.shape[0], -100, dtype=np.int64)
                inst_label = np.full(xyz.shape[0], -1, dtype=np.int64)
            else:
                mapped = np.full(sem_label.shape[0], -100, dtype=np.int64)
                valid = (sem_label >= 0) & (sem_label < len(SEM_TO_INST))
                mapped[valid] = SEM_TO_INST[sem_label[valid]]
                sem_label = mapped
                inst_label = inst_label.copy()
                inst_label[sem_label < 0] = -1
        else:
            sem_label = np.full(xyz.shape[0], -100, dtype=np.int64)
            inst_label = np.full(xyz.shape[0], -1, dtype=np.int64)

        return xyz, rgb, superpoint, sem_label, inst_label, normal

    def get_instance3D(self, instance_label, semantic_label, superpoint, coord_float, scan_id):
        num_insts = int(instance_label.max().item()) + 1 if instance_label.numel() else 0
        num_points = len(instance_label)
        gt_masks, gt_labels, gt_bboxes = [], [], []

        if self.use_normalized:
            scene_min = coord_float.min(0)[0]
            scene_max = coord_float.max(0)[0]

        gt_inst = torch.zeros(num_points, dtype=torch.int64)
        for i in range(max(num_insts, 0)):
            idx = torch.where(instance_label == i)
            if idx[0].numel() == 0:
                continue
            sem_ids = torch.unique(semantic_label[idx])
            sem_ids = sem_ids[sem_ids != -100]
            if sem_ids.numel() == 0:
                continue
            sem_id = sem_ids[0]
            gt_mask = torch.zeros(num_points)
            gt_mask[idx] = 1
            gt_masks.append(gt_mask)
            gt_labels.append(sem_id)
            gt_inst[idx] = (sem_id + 1) * 1000 + i + 1

            xyz_i = coord_float[idx]
            mean_xyz_i = xyz_i.mean(0)
            min_xyz_i = xyz_i.min(0)[0]
            max_xyz_i = xyz_i.max(0)[0]
            center_xyz_i = (min_xyz_i + max_xyz_i) / 2
            hwz_i = (max_xyz_i - min_xyz_i)
            gt_bbox = torch.cat([mean_xyz_i, center_xyz_i, hwz_i], dim=0)

            if self.use_normalized:
                mean_xyz_i_norm = (mean_xyz_i - scene_min) / (scene_max - scene_min)
                center_xyz_i_norm = (center_xyz_i - scene_min) / (scene_max - scene_min)
                hwz_i_norm = hwz_i / (scene_max - scene_min)
                gt_bbox = torch.cat([gt_bbox, mean_xyz_i_norm, center_xyz_i_norm, hwz_i_norm], dim=0)

            gt_bboxes.append(gt_bbox)

        if gt_masks:
            gt_masks = torch.stack(gt_masks, dim=0)
            gt_spmasks = scatter_mean(gt_masks.float(), superpoint, dim=-1)
            gt_spmasks = (gt_spmasks > 0.5).float()
        else:
            gt_spmasks = torch.tensor([])
            gt_masks = torch.tensor([])
        gt_labels = torch.tensor(gt_labels)
        if len(gt_bboxes) > 0:
            gt_bboxes = torch.stack(gt_bboxes, dim=0)
        else:
            gt_bboxes = torch.tensor(gt_bboxes)
        assert gt_labels.shape[0] == gt_bboxes.shape[0]

        inst = Instances3D(num_points, gt_instances=gt_inst.numpy())
        inst.gt_labels = gt_labels.long()
        inst.gt_spmasks = gt_spmasks
        inst.gt_masks = gt_masks
        inst.gt_bboxes = gt_bboxes
        return inst
