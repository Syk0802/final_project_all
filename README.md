# Multi-View Consistency for Occlusion- and Scale-Robust Person Re-identification

This repository is the official code for the paper *"A Multi-View Consistency Method for Occlusion- and Scale-Robust Person Re-identification"*, built on top of [Bag of Tricks and a Strong Baseline for Deep Person Re-identification](https://github.com/michuanhaohao/reid-strong-baseline) (`reid-strong-baseline`).

Centered on the idea of *multi-view consistency*, the paper progressively introduces Random Oval Erasing (ROE), the Scale View Consistency Network (SVC-Net), the Occlusion View Consistency Network (OVC-Net), the Occlusion-Scale View Consistency Network (OSVC-Net), as well as the View Consistency Loss (VCLoss) and the Bipartite Part Alignment mechanism (BiPA).

---

## Table of Contents

- [Project Structure](#project-structure)
- [Method–Directory Mapping](#methoddirectory-mapping)
- [Dependencies](#dependencies)
- [Dataset Preparation](#dataset-preparation)
- [Pretrained Weights](#pretrained-weights)
- [Training](#training)
- [Testing & Evaluation](#testing--evaluation)
- [Key Configurations](#key-configurations)
- [Reproduction](#reproduction)
- [Citation](#citation)

---

## Project Structure

The repository contains four independent, self-contained subprojects, each corresponding to a group of experiments in the paper:

```
.
├── main.tex          # LaTeX source of the paper
├── README.md
├── baseline/         # Single-view strong baseline + erasing-augmentation ablation (platform for §4.3 ROE ablation)
├── svc_net/          # SVC-Net (original + scaled views)
├── ovc_net/          # OVC-Net (original + occluded views)
└── osvc_net/         # OSVC-Net (original + occluded + scaled views)
```

All subprojects share the same internal layout (inherited from `reid-strong-baseline`):

```
<subproject>/
├── config/            # yacs default configuration (defaults.py)
├── configs/           # experiment-specific yml config files
├── data/              # data loading, sampling, transforms (incl. ROE implementation)
├── engine/            # training/inference engines (trainer_global.py / trainer_bipa.py)
├── layers/            # loss functions (incl. VCLoss: layers/vcloss.py)
├── modeling/          # models (baseline.py / bipa.py, i.e. the BiPA bipartite-part model)
├── solver/            # optimizer and learning-rate schedulers
├── tools/             # entry points (train.py / train_global.py / train_bipa.py / test.py)
├── utils/             # utility functions
├── train.sh           # baseline training script
├── train_global.sh    # Global-configuration training script
├── train_bipa.sh      # BiPA-configuration training script
└── test.sh            # evaluation script
```

---

## Method–Directory Mapping

| Directory | Paper method | Training views | Input size | Backbone |
|-----------|--------------|----------------|------------|----------|
| `baseline/` | Baseline (rectangular erasing) & Baseline + ROE (oval erasing) ablation | $I^o$ (single view) | $256\times128$ | ResNet-50 |
| `svc_net/` | SVC-Net | $I^o+I^s$ (original + scaled) | $384\times128$ | ResNet-50-IBN-a |
| `ovc_net/` | OVC-Net | $I^o+I^r$ (original + occluded) | $384\times128$ | ResNet-50-IBN-a |
| `osvc_net/` | OSVC-Net | $I^o+I^r+I^s$ (three views) | $384\times128$ | ResNet-50-IBN-a |

The multi-view subprojects support two configurations:

- **Global configuration**: uses the `Baseline` model (single classifier); VCLoss acts on the global feature (`train_global.sh`, feature dim $2048$).
- **BiPA configuration**: uses the `BiPA` model (bipartite split along the height, dual BNNeck/classifier); VCLoss acts on the concatenated upper/lower part features (`train_bipa.sh`, feature dim $4096$).

### Mapping between `baseline/` and the paper's experiments

`baseline/` is the platform for the §4.3 ablation study of Random Oval Erasing (plain ResNet-50, $256\times128$, evaluated on Market-1501 only). By switching the eraser and its parameters, it covers the 8 configurations in Table `tab:roe` of the paper:

| Exp. ID | Erasing strategy | Eraser used | Key config |
|---------|------------------|-------------|------------|
| 00 | Rectangle erasing (Random Erasing, baseline) | `RandomErasing` | — |
| 01 | Rotated oval · single region · zero fill | `RandomOvalErasing` | `shape=oval`, `fill=zero`, single region |
| 02 | Rotated oval · multi-region · zero fill | `RandomOvalErasing` | `shape=oval`, `fill=zero`, multi-region |
| 03 | Rotated oval · multi-region · patch-copy fill | `RandomOvalErasing` | `shape=oval`, `fill=patch`, multi-region |
| 04 | Rotated oval · multi-region · random fill | `RandomOvalErasing` | `shape=oval`, `fill=random`, multi-region |
| 05 | Irregular shape (Blob) · zero fill | `RandomOvalErasing` | `shape=blob`, `fill=zero` |
| 06 | Irregular shape (Blob) · Gaussian-noise fill | `RandomOvalErasing` | `shape=blob`, `fill=noise` |
| 07 | Fully random (shape + fill + count) | `RandomOvalErasing` | `shape=random`, `fill=random` |

Here **00 (rectangular erasing)** corresponds to the pure Baseline in the paper, while **01–07 (oval and its variants)** correspond to Baseline + ROE.

Implementation note: `data/transforms/transforms.py` provides both the standard rectangular eraser `RandomErasing` and the random-oval eraser `RandomOvalErasing`. `data/transforms/build.py`'s `build_transforms` selects the eraser based on `INPUT.RE_SHAPE` — `rect` selects `RandomErasing` (pure Baseline), while `oval`/`blob`/`random` select `RandomOvalErasing` (ROE ablation variants).

---

## Dependencies

This code inherits from `reid-strong-baseline`. The recommended environment is:

| Dependency | Version | Description |
|------------|---------|-------------|
| Python | 3.7 / 3.8 | 3.8 recommended |
| [PyTorch](https://pytorch.org/) | >= 1.2 (with CUDA) | Deep learning framework |
| torchvision | matching PyTorch | Data transforms, model zoo |
| [pytorch-ignite](https://github.com/pytorch/ignite) | **== 0.1.2** | Training engine. Note: `0.2.0`+ has breaking API changes and will fail |
| [yacs](https://github.com/rbgirshick/yacs) | latest | Configuration management |
| numpy | latest | Numerical computation |
| scipy | latest | Outputs `.mat` in visualization scripts |
| matplotlib | latest | Visualization |
| Pillow | latest | Image loading |

### Installation

```bash
# 1. Create and activate a virtual environment (optional)
conda create -n reid python=3.8 -y
conda activate reid

# 2. Install PyTorch and torchvision (pick the command matching your CUDA version at pytorch.org)
pip install torch torchvision

# 3. Install the remaining dependencies
pip install pytorch-ignite==0.1.2 yacs numpy scipy matplotlib pillow
```

> **Note**: pin `pytorch-ignite` to `0.1.2`; newer `0.2.x` versions are incompatible with the training engine due to API changes (this caveat comes from the official `reid-strong-baseline` README).

---

## Dataset Preparation

### Download

| Dataset | Download link | Description |
|---------|---------------|-------------|
| Market-1501 | [Google Drive](https://drive.google.com/file/d/1SMx9IBJORLyNZJbWG_f95Id3nYrU_XR_/view?usp=drive_link) | 6 cameras, 1501 identities, 32668 images |
| DukeMTMC-reID | [Google Drive](https://drive.google.com/file/d/1YNuDMY3phlRNg06yX_ymPpmdn64H-YNi/view?usp=drive_link) | 8 cameras, 1404 identities, 36411 images |
| MSMT17 | [Google Drive](https://drive.google.com/file/d/1voh3E4zZz-48WPoGhgLMiF3KmDZBsfDy/view?usp=drive_link) | 15 cameras, 4101 identities, 126441 images |

### Directory Layout

After downloading and extracting, organize the data as follows (`<ROOT_DIR>` is the parent directory pointed to by `DATASETS.ROOT_DIR`):

```
<ROOT_DIR>/
├── Market/
│   ├── bounding_box_train/
│   ├── query/
│   └── bounding_box_test/
├── Duke/
│   ├── bounding_box_train/
│   ├── query/
│   └── bounding_box_test/
└── MSMT17_V2/
    ├── mask_train_v2/
    ├── mask_test_v2/
    ├── list_train.txt
    ├── list_val.txt
    ├── list_query.txt
    └── list_gallery.txt
```

> The dataset directory names (`Market` / `Duke` / `MSMT17_V2`) are hard-coded in `data/datasets/*.py`; do not rename them. Only make sure the parent directory `DATASETS.ROOT_DIR` is correct.

Set `DATASETS.ROOT_DIR` (e.g. `'/path/to/your/data'`) when training/testing. It can be set in `config/defaults.py`, in `configs/*.yml`, or via command-line `opts`.

---

## Pretrained Weights

| Subproject | Backbone | Pretrained weight | Download link |
|------------|----------|-------------------|---------------|
| `baseline/` | ResNet-50 | `resnet50-19c8e357.pth` | [Google Drive](https://drive.google.com/file/d/1O9W2myi9dBLDFmTvnFaEBGMQyTRwYZMP/view?usp=drive_link) |
| `svc_net/` / `ovc_net/` / `osvc_net/` | ResNet-50-IBN-a | `resnet50_ibn_a.pth.tar` | [Google Drive](https://drive.google.com/file/d/1GE268hKKSYJtkwII1ISIrgwqZRk5pVq0/view?usp=drive_link) |

For `baseline/`, `PRETRAIN_PATH` is set as an absolute path in `configs/softmax_triplet.yml`; update it to the actual location of `resnet50-19c8e357.pth` after download. For the three multi-view subprojects, `PRETRAIN_PATH` is the relative path `'resnet50_ibn_a.pth.tar'`; place the downloaded `resnet50_ibn_a.pth.tar` in the subproject root.

---

## Training

### Single-view baseline (`baseline/`)

`baseline/` is used for the §4.3 erasing ablation study, with entry point `tools/train.py`:

```bash
cd baseline
python3 tools/train.py --config_file='configs/softmax_triplet.yml' \
  MODEL.DEVICE_ID "('0')" \
  DATASETS.NAMES "('market1501')" \
  DATASETS.ROOT_DIR "('/path/to/your/data')" \
  INPUT.RE_PROB "(0.5)" \
  INPUT.RE_SHAPE "('rect')" \
  INPUT.RE_FILL "('zero')" \
  INPUT.RE_MIN_COUNT "(1)" \
  INPUT.RE_MAX_COUNT "(1)" \
  OUTPUT_DIR "('./results/baseline/market')"
```

Switch among the 8 configurations of §4.3:

- **00 rectangular baseline**: `INPUT.RE_SHAPE "('rect')"` (default, `RandomErasing`).
- **01–07 oval variants**: set `INPUT.RE_SHAPE` to one of `oval`/`blob`/`random` to enable `RandomOvalErasing`, combined with `INPUT.RE_FILL` (`zero`/`patch`/`mean`/`noise`/`random`) and `INPUT.RE_MIN_COUNT`/`RE_MAX_COUNT` (single region `1/1`, multi-region `1/3`).

### Multi-view training (`svc_net` / `ovc_net` / `osvc_net`)

The three subprojects share a unified training-script interface:

```bash
cd <subproject>   # svc_net | ovc_net | osvc_net

# Global configuration (Baseline model, VCLoss on global features)
bash train_global.sh <gpu_id> <delta> <vc_type>

# BiPA configuration (BiPA bipartite-part model, VCLoss on part-concatenated features)
bash train_bipa.sh   <gpu_id> <delta> <vc_type>
```

**Arguments:**

| Argument | Meaning | Default | Options |
|----------|---------|---------|---------|
| `gpu_id` | GPU device id | required | `0`, `1`, ... |
| `delta` | VCLoss weight $\delta$ (i.e. `SOLVER.DELTA`) | required | paper default `0.2` |
| `vc_type` | VCLoss variant (i.e. `SOLVER.VC_TYPE`) | `l2` | `l2` \| `ntxent` \| `cosine` \| `kl` |

**Examples:**

```bash
# OSVC-Net (three views) + BiPA + KL consistency loss, δ=0.2, GPU 0
cd osvc_net
bash train_bipa.sh 0 0.2 kl

# OVC-Net (original + occluded) + Global + L2 consistency loss, δ=0.2, GPU 1
cd ../ovc_net
bash train_global.sh 1 0.2 l2
```

**The number of views (`VC_NUM`) is preset by the scripts; no need to specify it manually:**

| Subproject | View combination | `VC_NUM` |
|------------|------------------|----------|
| `svc_net` | $I^o+I^s$ | 2 |
| `ovc_net` | $I^o+I^r$ | 2 |
| `osvc_net` | $I^o+I^r+I^s$ | 3 |

**Occluded-view construction (`ovc_net` / `osvc_net`)**: `INPUT.RE_PROB` is set to `1.1` to guarantee an occluded view is generated for every training sample; `INPUT.RE_SHAPE_MODE=oval` and `INPUT.RE_FILL_MODE=zero` specify the ROE shape and fill.

---

## Testing & Evaluation

Use the unified test script:

```bash
cd <subproject>
bash test.sh <gpu_id> <expname> <dataset>
```

| Argument | Meaning | Example |
|----------|---------|---------|
| `gpu_id` | GPU device id | `0` |
| `expname` | result directory name (relative to `results/`) | `bipa_kl_0.2` |
| `dataset` | dataset name | `market1501` \| `dukemtmc` \| `msmt17` |

`test.sh` invokes `tools/test.py`, which iterates over the checkpoints at `epoch ∈ {51, 54, ..., 78}` under `TEST.WEIGHT` and evaluates each (mAP and CMC Rank-1/5/10). The model type (Baseline or BiPA) is auto-detected by whether the path contains `bipa`.

Evaluation follows the standard protocol: single-query for Market-1501 / DukeMTMC-reID and the official protocol for MSMT17, with no re-ranking and no Test-Time Augmentation.

---

## Key Configurations

Configuration is built by merging `configs/softmax_triplet.yml` over the `config/defaults.py` defaults, further overridden by command-line `opts`. Core hyperparameters:

| Config | Meaning | Default |
|--------|---------|---------|
| `MODEL.NAME` | Backbone | `resnet50` / `resnet50_ibn_a` |
| `MODEL.LAST_STRIDE` | Last stride of stage-4 (1 keeps resolution) | `1` |
| `MODEL.METRIC_LOSS_TYPE` | Metric loss type | `triplet` |
| `MODEL.IF_LABELSMOOTH` | Label smoothing | `on` |
| `INPUT.SIZE_TRAIN` / `SIZE_TEST` | Input resolution | `[384, 128]` (`[256, 128]` in baseline) |
| `INPUT.RE_PROB` | ROE / RE trigger probability | `0.5` (`1.1` for occluded-view construction) |
| `INPUT.RE_SHAPE_MODE` | Erasing shape | `oval` (`rect`/`oval`/`blob`/`random`) |
| `INPUT.RE_FILL_MODE` | Erasing fill | `zero` (`zero`/`patch`/`mean`/`noise`/`random`) |
| `INPUT.RE_MIN_COUNT` / `RE_MAX_COUNT` | Erased-region count range | `1` / `1` |
| `SOLVER.OPTIMIZER_NAME` | Optimizer | `Adam` |
| `SOLVER.BASE_LR` | Base learning rate | `3.5e-4` |
| `SOLVER.IMS_PER_BATCH` | Batch size | `32` (PK sampling: $P=8$, $K=4$) |
| `SOLVER.MAX_EPOCHS` | Total epochs | `80` |
| `SOLVER.STEPS` / `GAMMA` | LR decay epochs / factor | `[30, 50]` / `0.1` |
| `SOLVER.WARMUP_ITERS` / `WARMUP_FACTOR` | LR warm-up | `10` / `0.01` |
| `SOLVER.DELTA` | VCLoss weight $\delta$ | `0.2` |
| `SOLVER.VC_TYPE` | VCLoss variant | `l2` |
| `SOLVER.VC_NUM` | Number of views (groups) | `2` (`3` for three views) |
| `SOLVER.VC_IN_DIM` | VCLoss projector input dim | `2048` (`4096` for BiPA) |
| `SOLVER.VC_TEMPERATURE` | NT-Xent temperature $\tau$ | `0.07` |
| `SOLVER.VC_EPS` | Cosine variance-regularization epsilon | `1e-4` |
| `TEST.NECK_FEAT` | Retrieval feature before/after BNNeck | `after` |
| `TEST.FEAT_NORM` | Whether to $\ell_2$-normalize features | `yes` |

---

## Reproduction

Complete reproduction commands for the main results (assuming data is prepared as above and `DATASETS.ROOT_DIR` is set to the actual path in the scripts):

```bash
# 1) Single-view baseline (§4.3 ablation, Market-1501)
cd baseline
python3 tools/train.py --config_file='configs/softmax_triplet.yml' \
  MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('market1501')" \
  DATASETS.ROOT_DIR "('/path/to/data')" \
  INPUT.RE_PROB "(0.5)" INPUT.RE_SHAPE "('rect')" INPUT.RE_FILL "('zero')" \
  INPUT.RE_MIN_COUNT "(1)" INPUT.RE_MAX_COUNT "(1)" \
  OUTPUT_DIR "('./results/baseline/market')"

# 2) SVC-Net · BiPA · KL · δ=0.2 (Market-1501)
cd ../svc_net
bash train_bipa.sh 0 0.2 kl

# 3) OVC-Net · BiPA · L2 · δ=0.2 (Market-1501)
cd ../ovc_net
bash train_bipa.sh 0 0.2 l2

# 4) OSVC-Net · BiPA · KL · δ=0.2 (Market-1501, main experiment of the paper)
cd ../osvc_net
bash train_bipa.sh 0 0.2 kl

# 5) Evaluation (OSVC-Net · BiPA · KL · δ=0.2 as an example)
bash test.sh 1 bipa_kl_0.2 market1501
```

---

## Citation

If this repository helps your research, please cite the corresponding paper and respect the open-source license of the original `reid-strong-baseline`:

- Luo H., Gu Y., Liao X., et al. *Bag of Tricks and a Strong Baseline for Deep Person Re-identification*. CVPRW 2019.

```bibtex
@InProceedings{Luo_2019_CVPR_Workshops,
  author    = {Luo, Hao and Gu, Youzhi and Liao, Xingyu and Lai, Shenqi and Jiang, Wei},
  title     = {Bag of Tricks and a Strong Baseline for Deep Person Re-Identification},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month     = {June},
  year      = {2019}
}
```
