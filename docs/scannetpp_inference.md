# ScanNet++ inference and evaluation

This document is **ScanNet++ (PP) only**. It covers running **inference / eval** on this repo as it exists today. It does not cover training, and it does not cover ScanNet v2.

Published val numbers for the official checkpoint:

- **AP / AP50 / AP25 = 0.339 / 0.485 / 0.581** on the official **50-scene** val split
- Checkpoint: `competitorformer_scannetpp.pth` (`num_query=500`)

All commands below assume the **repository root** (the directory that contains `tools/`, `configs/`, and `competitorformer/`).

---

## 1. Environment

Use the same conda env as [Installation](../README.md#-installation):

```bash
conda activate competitorformer
cd /path/to/CompetitorFormer
export PYTHONPATH=$PWD:$PYTHONPATH
```

- **Python / CUDA:** Python 3.8, PyTorch 1.13.1, CUDA 11.7 (see the README). Eval runs on GPU (`model.cuda()`). Val batch size is already `1`.
- **PYTHONPATH:** the eval command below also prefixes `PYTHONPATH=.`. After `python3 setup.py develop` the `competitorformer` package is importable, but setting `PYTHONPATH` to the repo root is the reliable way to run `tools/train.py`.
- **seaborn:** not required by this package for PP eval (`requirements.txt` does not list it). If you hit `ModuleNotFoundError: seaborn`, install it with `pip install seaborn`.

---

## 2. Checkpoint

Download the official ScanNet++ weight from Hugging Face [`WangDuanchu/CompetitorFormer`](https://huggingface.co/WangDuanchu/CompetitorFormer) and save it as `checkpoints/competitorformer_scannetpp.pth`.

This checkpoint is trained with **`model.decoder.num_query: 500`** (the value already in `configs/scannetpp/competitorformer_scannetpp.yaml`). Do not reuse a ScanNet v2 config (`num_query: 400`).

```bash
mkdir -p checkpoints
wget https://huggingface.co/WangDuanchu/CompetitorFormer/resolve/main/competitorformer_scannetpp.pth \
  -O checkpoints/competitorformer_scannetpp.pth
```

or:

```bash
huggingface-cli download WangDuanchu/CompetitorFormer competitorformer_scannetpp.pth --local-dir checkpoints
```

`--resume` below points at this file. With `--eval_only`, `tools/train.py` loads the checkpoint with `gorilla.load_checkpoint(..., strict=False)` and does **not** load `train.pretrain`.

---

## 3. Data layout

### Official split

Use the official ScanNet++ NVS/semantic **val** list `splits/nvs_sem_val.txt` (**50 scenes**). A copy ships with this repo at `data/scannetpp/nvs_sem_val.txt`.

The `scannetpp` loader does **not** read that txt file. It globs every **directory** under `data_root/prefix` (default: `data/scannetpp/val_vtx`). To match the published AP, put **exactly those 50 scenes** under `val_vtx/` (no extra rooms).

### Processed npy (what eval actually loads)

Eval expects already-processed Pointcept / LaSSM-style per-scene folders. This repo does **not** ship a ScanNet++ preprocess script (unlike `data/scannetv2/preprocess_scannetv2.py`). Superpoints use the same `segmentator` library as in Installation.

Required files in each scene folder:

| File | Role |
| --- | --- |
| `coord.npy` | XYZ |
| `color.npy` | RGB. Typically **uint8** 0–255; the loader converts with `rgb/127.5 - 1` when `rgb.max() > 1.5` |
| `normal.npy` | Vertex normals (`input_channel: 9` / `with_normals: True`) |
| `superpoint.npy` | Superpoint ids (required) |
| `segment.npy` | Semantic ids (100-class ScanNet++ labels). Alias: `semantic_label.npy` |
| `instance.npy` | Instance ids. Alias: `instance_label.npy` |

Default layout matching the yaml (`data_root: data/scannetpp`, `prefix: val_vtx`, `suffix: .npy`):

```
CompetitorFormer
├── data
│   ├── scannetpp
│   │   ├── nvs_sem_val.txt                          # official 50-scene val split
│   │   ├── val_vtx
│   │   │   ├── 09c1414f1b
│   │   │   │   ├── coord.npy
│   │   │   │   ├── color.npy
│   │   │   │   ├── normal.npy
│   │   │   │   ├── superpoint.npy
│   │   │   │   ├── segment.npy
│   │   │   │   └── instance.npy
│   │   │   ├── 0d2ee665be
│   │   │   └── ...                                  # 50 scenes total
```

`--eval_only` builds **only** `cfg.data.val`. The train prefix `train_grid1mm_chunk6x6_stride3x3` is unused; a missing train split is fine.

### Classes (84 things; no ScanNet v2 remap)

- Loader type: `scannetpp` (`competitorformer/dataset/scannetpp.py`).
- `model.num_class: 84` — instance classes in `INST_LABELS_PP` (`competitorformer/dataset/scannetpp_constants.py`).
- Semantic ids are the official **100** ScanNet++ classes. Thing names are mapped to 0–83; **stuff** (wall, floor, ceiling, …) is set to **`-100`** and those instance ids are dropped (`-1`).
- **`inst_stuff_remap = False`**: do **not** apply the ScanNet v2 `label-2` remap.

### Pointing `data_root` at data elsewhere

Paths in the yaml are relative to your **cwd** (repo root). Eval-only reads `data.val` only.

**Option A — symlink** (keep the default yaml):

```bash
mkdir -p data/scannetpp
ln -s /path/to/your/processed_scannetpp/val_vtx data/scannetpp/val_vtx
```

**Option B — edit the yaml:** set `data.val.data_root` in `configs/scannetpp/competitorformer_scannetpp.yaml` to the parent of `val_vtx`. Example: if scenes live at `/data/scannetpp_processed/val_vtx/<scene>/`, use:

```yaml
data:
  val:
    type: scannetpp
    data_root: /data/scannetpp_processed
    prefix: val_vtx
```

(`data.train.data_root` / `data.test.data_root` are ignored under `--eval_only`.)

---

## 4. Inference / eval command

```bash
conda activate competitorformer
export PYTHONPATH=$PWD:$PYTHONPATH

PYTHONPATH=. python tools/train.py configs/scannetpp/competitorformer_scannetpp.yaml \
  --work_dir exps/scannetpp_eval --eval_only \
  --resume checkpoints/competitorformer_scannetpp.pth
```

What this does:

- **`--eval_only`:** skip the train set. Only `build_dataset(cfg.data.val)` + one `eval()` pass, then exit. You do not need `train_grid1mm_chunk6x6_stride3x3`.
- **`--resume`:** the checkpoint to load (`checkpoints/competitorformer_scannetpp.pth`). Not a training resume of optimizer/epoch when `--eval_only` is set.
- **`--work_dir exps/scannetpp_eval`:** logs and a copied yaml go here.
- **`model.test_cfg.topk_insts: 1300`** comes from `configs/scannetpp/competitorformer_scannetpp.yaml` (with `score_thr: 0.0`, `npoint_thr: 100`). Do not reuse ScanNet v2 `topk_insts` (200).

Other PP-specific yaml knobs that must match the checkpoint:

- `model.decoder.num_query: 500`
- `model.num_class: 84`
- `data.val.prefix: val_vtx`

---

## 5. Expected output

On the official 50-scene val set with this checkpoint, you should see:

**AP / AP50 / AP25 = 0.339 / 0.485 / 0.581**

`tools/train.py` prints that line after `Evaluate instance segmentation`:

```text
AP: 0.339. AP_50: 0.485. AP_25: 0.581
```

Look for `AP:` in:

- stdout (the training logger)
- `exps/scannetpp_eval/<YYYYMMDD_HHMMSS>.log`

TensorBoard scalars `val/AP`, `val/AP_50`, `val/AP_25` are written under the same work dir.

---

## 6. Pitfalls

- **Missing train split is OK** with `--eval_only`. Only `data/scannetpp/val_vtx/<scene>/` is required.
- **OOM on huge scenes** is already patched in `competitorformer/model/competitorformer.py` `predict_by_feat`: score / `npoint_thr` filters run on **superpoints first**, then masks are expanded to points. You should not need to lower `topk_insts` for that OOM.
- **Yaml already in `work_dir`:** `tools/train.py` copies the config into `work_dir` unless source and destination are the same file (`os.path.realpath`). If you pass `exps/scannetpp_eval/competitorformer_scannetpp.yaml` as the config, the copy is skipped.
- **uint8 RGB:** keep `color.npy` as 0–255 uint8 (or float in that range). The loader normalizes only when `max > 1.5`. Do not pre-scale to `[-1, 1]` unless you also keep `max ≤ 1.5`.
- **Stuff → `-100`:** 100-class semantic ids that are not in the 84 thing list become ignore. Do not apply ScanNet v2 `label-2`.
- **Extra scenes under `val_vtx/`** change the metric. Keep the folder to the 50 ids in `data/scannetpp/nvs_sem_val.txt`.
- **Wrong config:** ScanNet v2 yaml (`num_query: 400`, `num_class: 18`, `topk_insts: 200`, `.pth` scenes) will not reproduce these numbers.

---

## 7. Preprocess

There is **no** ScanNet++ preprocess script in this repository. Eval expects the npy folders in §3 to already exist (Pointcept / LaSSM-style `coord` / `color` / `normal` / `superpoint` / `segment` / `instance`).

High-level recipe if you are preparing data yourself from the [official ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/) release (`mesh_aligned_0.05.ply` plus `segments.json` / `segments_anno.json`):

1. Restrict to the 50 ids in `data/scannetpp/nvs_sem_val.txt`.
2. Dump vertex `coord` and uint8 `color`; compute vertex `normal` from the mesh.
3. Compute `superpoint` with `segmentator` (same build as Installation).
4. Write `segment.npy` (100-class semantic ids) and `instance.npy`.

In-repo preprocess scripts (`data/scannetv2/preprocess_scannetv2.py`, `data/scannet200/preprocess_scannet200.py`) are **not** for ScanNet++.
