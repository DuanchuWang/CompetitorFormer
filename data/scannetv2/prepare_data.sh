#!/bin/bash
echo Prepare shared superpoints
python3 prepare_superpoint.py \
  --dataset_root /data/yangjunjie/scannetv2/scans \
  --dataset_root_test /data/yangjunjie/scannetv2/scans_test \
  --output_root /data/yangjunjie/scannetv2_spformer/superpoints \
  --train_val_splits_path ./

echo Preprocess scannetv2 data
python3 preprocess_scannetv2.py \
  --dataset_root /data/yangjunjie/scannetv2/scans \
  --dataset_root_test /data/yangjunjie/scannetv2/scans_test \
  --output_root /data/yangjunjie/scannetv2_spformer \
  --train_val_splits_path ./
