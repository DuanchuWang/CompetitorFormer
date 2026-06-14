import numpy as np
import pandas as pd
import torch

import argparse
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

# Load external constants
from scannet200_constants import VALID_CLASS_IDS_200
from utils import point_indices_from_group, read_plymesh


warnings.filterwarnings("ignore", category=DeprecationWarning)

###################################

def face_normal(vertex, face):
    v01 = vertex[face[:, 1]] - vertex[face[:, 0]]
    v02 = vertex[face[:, 2]] - vertex[face[:, 0]]
    vec = np.cross(v01, v02)
    length = np.sqrt(np.sum(vec ** 2, axis=1, keepdims=True)) + 1.0e-8
    nf = vec / length
    area = length * 0.5
    return nf, area


def vertex_normal(vertex, face):
    nf, area = face_normal(vertex, face)
    nf = nf * area

    nv = np.zeros_like(vertex)
    for i in range(face.shape[0]):
        nv[face[i]] += nf[i]

    length = np.sqrt(np.sum(nv ** 2, axis=1, keepdims=True)) + 1.0e-8
    nv = nv / length
    return nv

###################################

CLOUD_FILE_PFIX = "_vh_clean_2"
SEGMENTS_FILE_PFIX = ".0.010000.segs.json"
AGGREGATIONS_FILE_PFIX = ".aggregation.json"
CLASS_IDs = VALID_CLASS_IDS_200

NORMALIZED_CLASS_IDS_200 = [-100 for _ in range(1192)]
REVERSE_NORMALIZED_CLASS_IDS_200 = [-100 for _ in range(200)]

count_id = 2
for i, cls_id in enumerate(VALID_CLASS_IDS_200):
    if cls_id == 1:  # wall
        NORMALIZED_CLASS_IDS_200[cls_id] = 0
        REVERSE_NORMALIZED_CLASS_IDS_200[0] = cls_id        # [wall]
    elif cls_id == 3:  # floor
        NORMALIZED_CLASS_IDS_200[cls_id] = 1
        REVERSE_NORMALIZED_CLASS_IDS_200[1] = cls_id        # [wall, floor]
    else:
        NORMALIZED_CLASS_IDS_200[cls_id] = count_id
        REVERSE_NORMALIZED_CLASS_IDS_200[count_id] = cls_id # [wall, floor, inst1, inst2, ..., inst198], [1, 3, 2, 4, .....]
        count_id += 1

REVERSE_NORMALIZED_CLASS_IDS_200_np = np.array(REVERSE_NORMALIZED_CLASS_IDS_200)

# SAVED_DIR =

def handle_process(scene_path,
                   output_path,
                   labels_pd,
                   train_scenes,
                   val_scenes,
                   ):

    scene_id = os.path.basename(scene_path.rstrip("/\\"))
    mesh_path = os.path.join(scene_path, f"{scene_id}{CLOUD_FILE_PFIX}.ply")
    segments_file = os.path.join(scene_path, f"{scene_id}{CLOUD_FILE_PFIX}{SEGMENTS_FILE_PFIX}")
    aggregations_file = os.path.join(scene_path, f"{scene_id}{AGGREGATIONS_FILE_PFIX}")
    info_file = os.path.join(scene_path, f"{scene_id}.txt")

    if scene_id in train_scenes:
        split_name = "train"
    elif scene_id in val_scenes:
        split_name = "val"
    else:
        print("Skip (not in train/val):", scene_id)
        return

    output_file = os.path.join(output_path, split_name, f"{scene_id}_inst_nostuff.pth")

    print("Processing: ", scene_id, "in ", split_name)

    # Rotating the mesh to axis aligned
    info_dict = {}
    with open(info_file) as f:
        for line in f:
            (key, val) = line.split(" = ")
            info_dict[key] = np.fromstring(val, sep=" ")

    if "axisAlignment" not in info_dict:
        rot_matrix = np.identity(4)
    else:
        rot_matrix = info_dict["axisAlignment"].reshape(4, 4)

    pointcloud, faces_array = read_plymesh(mesh_path)       # xyzrgb alpha
    # points = pointcloud[:, :3]
    # colors = pointcloud[:, 3:6]
    # alphas = pointcloud[:, -1]

    # Rotate PC to axis aligned
    r_points = pointcloud[:, :3].transpose()
    r_points = np.append(r_points, np.ones((1, r_points.shape[1])), axis=0)
    r_points = np.dot(rot_matrix, r_points)
    pointcloud = np.append(r_points.transpose()[:, :3], pointcloud[:, 3:], axis=1)

    has_annotation = os.path.exists(segments_file) and os.path.exists(aggregations_file)

    if has_annotation:
        with open(segments_file) as f:
            seg_indices = np.array(json.load(f)["segIndices"])

        with open(aggregations_file) as f:
            seg_groups = np.array(json.load(f)["segGroups"])

        labelled_pc = np.ones((pointcloud.shape[0], 1)) * -100
        instance_ids = np.ones((pointcloud.shape[0], 1)) * -100
        for group in seg_groups:
            segment_points, p_inds, label_id = point_indices_from_group(
                pointcloud, seg_indices, group, labels_pd, CLASS_IDs
            )
            labelled_pc[p_inds] = NORMALIZED_CLASS_IDS_200[label_id]
            instance_ids[p_inds] = group["id"]

        labelled_pc = labelled_pc.astype(int)
        instance_ids = instance_ids.astype(int)
    else:
        labelled_pc = np.ones(pointcloud.shape[0], dtype=int) * -100
        instance_ids = np.ones(pointcloud.shape[0], dtype=int) * -100

    vertices_rotated = pointcloud[:, :3].astype(np.float64)
    normals = vertex_normal(vertices_rotated, faces_array)

    torch.save(
        (pointcloud[:, :3],
         pointcloud[:, 3:6] / 127.5 - 1.0,
         normals.astype(np.float32),
         labelled_pc.reshape(-1),
         instance_ids.reshape(-1)),
        output_file,
    )
    # print("Saving to", output_file)


def collect_scene_paths(dataset_root, train_scenes, val_scenes):
    scene_paths = []
    for scene_id in sorted(set(train_scenes) | set(val_scenes)):
        scene_dir = os.path.join(dataset_root, scene_id)
        if os.path.isdir(scene_dir):
            scene_paths.append(scene_dir)
        else:
            print("Warning: scene not found:", scene_dir)
    return scene_paths


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default='', help="Path to ScanNet scans folder (train/val scenes)")
    parser.add_argument("--output_root", default='', help="Output path where train/val folders will be located")
    parser.add_argument("--label_map_file", default='../scannetv2/scannetv2-labels.combined.tsv', help="path to scannetv2-labels.combined.tsv")
    parser.add_argument("--num_workers", default=16, type=int, help="The number of parallel workers")
    parser.add_argument(
        "--train_val_splits_path",
        default="../scannetv2",
        help="Where the txt files with the train/val splits live",
    )
    config = parser.parse_args()

    labels_pd = pd.read_csv(config.label_map_file, sep="\t", header=0)

    with open(os.path.join(config.train_val_splits_path, "scannetv2_train.txt")) as train_file:
        train_scenes = train_file.read().splitlines()
    with open(os.path.join(config.train_val_splits_path, "scannetv2_val.txt")) as val_file:
        val_scenes = val_file.read().splitlines()

    for subdir in ("train", "val"):
        os.makedirs(os.path.join(config.output_root, subdir), exist_ok=True)

    # np.save(
    #     os.path.join(config.output_root, "reverse_norm_ids.npy"),
    #     REVERSE_NORMALIZED_CLASS_IDS_200_np,
    # )

    scene_paths = collect_scene_paths(config.dataset_root, train_scenes, val_scenes)

    print("Processing train/val scenes...")
    with ProcessPoolExecutor(max_workers=config.num_workers) as pool:
        _ = list(
            pool.map(
                handle_process,
                scene_paths,
                repeat(config.output_root),
                repeat(labels_pd),
                repeat(train_scenes),
                repeat(val_scenes),
            )
        )
