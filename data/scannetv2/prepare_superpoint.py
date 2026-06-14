import argparse
import glob
import os

import numpy as np
import open3d as o3d
import segmentator
import torch
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat


def get_superpoint(mesh_file):
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    vertices = torch.from_numpy(np.array(mesh.vertices).astype(np.float32))
    faces = torch.from_numpy(np.array(mesh.triangles).astype(np.int64))
    return segmentator.segment_mesh(vertices, faces).numpy()


def handle_process(scene_path, output_root):
    scene_id = os.path.basename(scene_path.rstrip("/\\"))
    mesh_path = os.path.join(scene_path, f"{scene_id}_vh_clean_2.ply")
    output_file = os.path.join(output_root, f"{scene_id}.pth")

    if not os.path.isfile(mesh_path):
        print("Skip (no mesh):", scene_id)
        return

    if os.path.exists(output_file):
        print("Skip (exists):", output_file)
        return

    superpoint = get_superpoint(mesh_path)
    torch.save(superpoint, output_file)
    print("Saving to", output_file)


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
    parser.add_argument("--dataset_root", default="/data/yangjunjie/scannetv2/scans", help="Path to ScanNet scans folder")
    parser.add_argument("--dataset_root_test", default="/data/yangjunjie/scannetv2/scans_test", help="Path to ScanNet scans_test folder")
    parser.add_argument("--output_root", default="/data/yangjunjie/scannetv2/superpoints", help="Shared superpoints output folder")
    parser.add_argument("--train_val_splits_path", default=".", help="Where scannetv2_test.txt lives")
    parser.add_argument("--num_workers", default=16, type=int)
    config = parser.parse_args()

    with open(os.path.join(config.train_val_splits_path, "scannetv2_test.txt")) as test_file:
        test_scenes = test_file.read().splitlines()

    os.makedirs(config.output_root, exist_ok=True)
    scene_paths = collect_scene_paths(config.dataset_root, test_scenes, config.dataset_root_test)

    print("Processing superpoints...")
    with ProcessPoolExecutor(max_workers=config.num_workers) as pool:
        _ = list(
            pool.map(
                handle_process,
                scene_paths,
                repeat(config.output_root),
            )
        )
