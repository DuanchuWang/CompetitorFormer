#!/bin/bash
set -e

SCNNETV2_OUTPUT=/data/yangjunjie/scannetv2_spformer

echo "Preprocess scannet200 train/val"
python3 preprocess_scannet200.py --dataset_root ../scannetv2/scans --output_root ./ --label_map_file ../scannetv2/scannetv2-labels.combined.tsv --train_val_splits_path ./

echo "Link shared test and superpoints from scannetv2"
ln -sfn ${SCNNETV2_OUTPUT}/test ./test
ln -sfn ${SCNNETV2_OUTPUT}/superpoints ./superpoints
