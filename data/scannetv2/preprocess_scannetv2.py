"""ScanNetV2 preprocessing.

Read scenes directly from scans/ and scans_test/, write per-split outputs under
train/val/test without split_data.py. Processing logic follows prepare_data_inst.py.
Normals follow prepare_data_inst_with_normal.py (computed on original mesh).
"""

import argparse
import glob
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np
import open3d as o3d
import plyfile
import torch

warnings.filterwarnings("ignore", category=DeprecationWarning)

G_LABEL_NAMES = [
    "unannotated", "wall", "floor", "chair", "table", "desk", "bed", "bookshelf", "sofa", "sink", "bathtub", "toilet",
    "curtain", "counter", "door", "window", "shower curtain", "refridgerator", "picture", "cabinet", "otherfurniture",
]


def get_raw2scannetv2_label_map(label_map_file):
    lines = [line.rstrip() for line in open(label_map_file)]
    lines = lines[1:]
    raw2scannet = {}
    label_classes_set = set(G_LABEL_NAMES)
    for line in lines:
        elements = line.split("\t")
        raw_name = elements[1]
        nyu40_name = elements[7]
        if nyu40_name not in label_classes_set:
            raw2scannet[raw_name] = "unannotated"
        else:
            raw2scannet[raw_name] = nyu40_name
    return raw2scannet

# Map relevant classes to {0,1,...,19}, and ignored classes to -100
remapper = np.ones(150) * (-100)
for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
    remapper[x] = i

# NYU40 ids for ScanNet instance evaluation txt (prepare_data_inst_gttxt.py)
semantic_label_idxs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]


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


def save_eval_gt_txt(sem_labels, instance_labels, output_path, scene_id):
    """Generate per-point instance GT txt for ScanNet evaluation."""
    instance_label_new = np.zeros(instance_labels.shape, dtype=np.int32)

    instance_num = int(instance_labels.max()) + 1
    for inst_id in range(instance_num):
        instance_mask = np.where(instance_labels == inst_id)[0]
        sem_id = int(sem_labels[instance_mask[0]])
        if sem_id == -100:
            sem_id = 0
        semantic_label = semantic_label_idxs[sem_id]
        instance_label_new[instance_mask] = semantic_label * 1000 + inst_id + 1

    np.savetxt(os.path.join(output_path, f"{scene_id}.txt"), instance_label_new, fmt="%d")


def process_scene(scene_path, output_path, train_scenes, val_scenes, g_raw2scannetv2):
    scene_id = os.path.basename(scene_path.rstrip("/\\"))
    mesh_path = os.path.join(scene_path, f"{scene_id}_vh_clean_2.ply")
    labels_path = os.path.join(scene_path, f"{scene_id}_vh_clean_2.labels.ply")
    segments_file = os.path.join(scene_path, f"{scene_id}_vh_clean_2.0.010000.segs.json")
    aggregations_file = os.path.join(scene_path, f"{scene_id}.aggregation.json")

    if not os.path.isfile(mesh_path):
        print("Skip (no mesh):", scene_id)
        return

    if scene_id in train_scenes:
        split_name = "train"
    elif scene_id in val_scenes:
        split_name = "val"
    else:
        split_name = "test"

    output_file = os.path.join(output_path, split_name, f"{scene_id}_inst_nostuff.pth")
    print("Processing:", scene_id, "in", split_name)

    f = plyfile.PlyData.read(mesh_path)
    points = np.array([list(x) for x in f.elements[0]])
    coords = np.ascontiguousarray(points[:, :3] - points[:, :3].mean(0))
    colors = np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    vertices = torch.from_numpy(np.array(mesh.vertices).astype(np.float32))
    faces = torch.from_numpy(np.array(mesh.triangles).astype(np.int64))
    normals = vertex_normal(vertices.numpy()[:, :3], faces.numpy()).astype(np.float32)

    has_annotation = (
        os.path.exists(labels_path)
        and os.path.exists(segments_file)
        and os.path.exists(aggregations_file)
    )

    if has_annotation:
        f2 = plyfile.PlyData.read(labels_path)
        sem_labels = remapper[np.array(f2.elements[0]["label"])]

        with open(segments_file) as jsondata:
            seg = json.load(jsondata)["segIndices"]

        segid_to_pointid = {}
        for i in range(len(seg)):
            if seg[i] not in segid_to_pointid:
                segid_to_pointid[seg[i]] = []
            segid_to_pointid[seg[i]].append(i)

        instance_segids = []
        with open(aggregations_file) as jsondata:
            d = json.load(jsondata)
            for x in d["segGroups"]:
                label = x["label"]
                if label not in g_raw2scannetv2:
                    continue
                if g_raw2scannetv2[label] != "wall" and g_raw2scannetv2[label] != "floor":
                    instance_segids.append(x["segments"])

        if (
            scene_id == "scene0217_00"
            and split_name == "val"
            and len(instance_segids) > 0
            and instance_segids[0] == instance_segids[int(len(instance_segids) / 2)]
        ):
            instance_segids = instance_segids[: int(len(instance_segids) / 2)]

        check = []
        for i in range(len(instance_segids)):
            check += instance_segids[i]
        assert len(np.unique(check)) == len(check)

        instance_labels = np.ones(sem_labels.shape[0]) * -100
        for i in range(len(instance_segids)):
            segids = instance_segids[i]
            pointids = []
            for segid in segids:
                pointids += segid_to_pointid[segid]
            instance_labels[pointids] = i
            assert len(np.unique(sem_labels[pointids])) == 1

        torch.save((coords, colors, normals, sem_labels, instance_labels), output_file)
    else:
        torch.save((coords, colors, normals), output_file)

    # print("Saving to", output_file)

    if has_annotation and split_name == "val":
        gt_dir = os.path.join(output_path, "val_gt")
        os.makedirs(gt_dir, exist_ok=True)
        save_eval_gt_txt(sem_labels, instance_labels, gt_dir, scene_id)
        # print("Saving eval gt to", os.path.join(gt_dir, f"{scene_id}.txt"))


def collect_scene_paths(dataset_root, test_scenes, dataset_root_test):
    scene_paths = []
    for scene_dir in sorted(glob.glob(os.path.join(dataset_root, "*"))):
        if os.path.isdir(scene_dir):
            scene_paths.append(scene_dir)

    if dataset_root_test:
        for scene_id in test_scenes:
            scene_dir = os.path.join(dataset_root_test, scene_id)
            if os.path.isdir(scene_dir):
                scene_paths.append(scene_dir)
            else:
                print("Warning: test scene not found:", scene_dir)

    return scene_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="", help="Path to ScanNet scans folder")
    parser.add_argument("--dataset_root_test", default="", help="Path to ScanNet scans_test folder")
    parser.add_argument("--output_root", default="", help="Output root with train/val/test subfolders")
    parser.add_argument("--train_val_splits_path", default=".", help="Where scannetv2_*.txt live")
    parser.add_argument(
        "--label_map_file",
        default="./scannetv2-labels.combined.tsv",
        help="path to scannetv2-labels.combined.tsv",
    )
    parser.add_argument("--num_workers", default=16, type=int)
    config = parser.parse_args()

    g_raw2scannetv2 = get_raw2scannetv2_label_map(config.label_map_file)

    with open(os.path.join(config.train_val_splits_path, "scannetv2_train.txt")) as f:
        train_scenes = f.read().splitlines()
    with open(os.path.join(config.train_val_splits_path, "scannetv2_val.txt")) as f:
        val_scenes = f.read().splitlines()
    with open(os.path.join(config.train_val_splits_path, "scannetv2_test.txt")) as f:
        test_scenes = f.read().splitlines()

    for subdir in ("train", "val", "test"):
        os.makedirs(os.path.join(config.output_root, subdir), exist_ok=True)

    scene_paths = collect_scene_paths(config.dataset_root, test_scenes, config.dataset_root_test)

    print("Processing scenes...")
    with ProcessPoolExecutor(max_workers=config.num_workers) as pool:
        _ = list(
            pool.map(
                process_scene,
                scene_paths,
                repeat(config.output_root),
                repeat(train_scenes),
                repeat(val_scenes),
                repeat(g_raw2scannetv2),
            )
        )
