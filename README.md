# AdPerception

LiDAR-camera 3D perception pipeline with multi-object tracking.  
Built for nuScenes - real-time target < 100ms @ 10 Hz.

## Architecture

```
LiDAR point cloud          6x surround cameras
        |                          |
  Voxelization              Swin-T + FPN
  PTv3 backbone             Lift-Splat-Shoot
        |                          |
   LiDAR BEV feat          Camera BEV feat
        '-------- concat ----------'
                     |
              BEV fusion encoder
                     |
         .-----------'-----------.
    CenterPoint             Seg head
    (detection)          (per-cell BEV)
         |
    SimpleTrack (MOT)
         |
    Tracks + 3D dashboard
```

## Requirements

| Component | Version |
|-----------|---------|
| OS | Ubuntu 24.04 |
| Python | 3.11 |
| CUDA (nvcc) | 12.1 |
| GCC | 12 (CUDA 12.1 does not support GCC 13+) |
| PyTorch | 2.1.2+cu121 |
| GPU | NVIDIA RTX 1000 Ada (compute 8.9) |

---

## Environment setup

### Step 0 - Prerequisites

```bash
# GCC 12 is required - CUDA 12.1 does not support GCC 13+
sudo apt install gcc-12 g++-12

# Verify
gcc-12 --version  # must show 12.x.x
```

### Step 1 - Conda environment

```bash
conda create -n AdPerception python=3.11 -y
conda activate AdPerception
```

### Step 2 - PyTorch (cu121 wheels)

```bash
pip install torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121
```

### Step 3 - spconv (sparse convolutions)

```bash
# cu120 covers the entire CUDA 12.x family - correct for CUDA 12.1
pip install spconv-cu120
```

### Step 4 - PyTorch Geometric

```bash
pip install torch-scatter torch-cluster \
    -f https://data.pyg.org/whl/torch-2.1.2+cu121.html

pip install torch-geometric
```

### Step 5 - All other dependencies

```bash
pip install -r requirements.txt
```

> If `lap` fails to build, use `lapx` instead (Python 3.11 wheels available):
> `pip install lapx`

### Step 6 - Pointcept (PTv3 backbone)

```bash
mkdir -p third_party
cd third_party
git clone https://github.com/Pointcept/Pointcept.git
cd ..
```

### Step 7 - Conda activation hooks

These environment variables are set automatically every time the conda
environment is activated.

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d

cat >> $CONDA_PREFIX/etc/conda/activate.d/adperception_paths.sh << 'HOOKS'
# Pointcept - main package path
export PYTHONPATH="$HOME/AdPerception/third_party/Pointcept:$PYTHONPATH"
# pointops - CUDA custom ops (compiled separately)
export PYTHONPATH="$HOME/AdPerception/third_party/Pointcept/libs:$PYTHONPATH"
# GCC 12 - required by CUDA 12.1 (does not support GCC 13+)
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
HOOKS
```

Apply immediately without restarting:

```bash
conda deactivate && conda activate AdPerception
```

### Step 8 - Compile pointops (CUDA custom ops for PTv3)

pointops is not a standard pip package - it must be compiled from source.
The `--no-build-isolation` flag is mandatory: it forces pip to use the torch
already installed in the conda environment instead of an isolated subprocess
that cannot see torch.

```bash
cd ~/AdPerception/third_party/Pointcept/libs/pointops

# TORCH_CUDA_ARCH_LIST=8.9 targets RTX Ada (compute capability 8.9)
TORCH_CUDA_ARCH_LIST="8.9" \
CC=/usr/bin/gcc-12 \
CXX=/usr/bin/g++-12 \
pip install -e . --no-build-isolation

cd ~/AdPerception
```

#### Known issue - circular import after compilation

If you see this error after compilation:

```
ImportError: cannot import name 'knn_query' from partially initialized module 'pointops'
```

Fix the absolute import in `functions/utils.py`:

```bash
sed -i 's/from pointops import knn_query, ball_query, grouping/from .query import knn_query, ball_query\nfrom .grouping import grouping/' \
    ~/AdPerception/third_party/Pointcept/libs/pointops/functions/utils.py
```

### Step 9 - Verify

```bash
python check_env.py
```

Expected output:

```
==================================================================
  AdPerception - Environment Check
  Python 3.11.x  |  ../anaconda3/envs/AdPerception/bin/python
==================================================================
Package           Status    Constraint      Detected version
----------------  --------  --------------  -------------------------
NumPy             PASS      < 2.0           1.26.4
SciPy             PASS                      1.13.1
scikit-learn      PASS                      1.5.2
PyTorch           PASS      2.1.2+cu121     2.1.2+cu121
torchvision       PASS                      0.16.2+cu121
CUDA available    PASS                      True
GPU name          PASS                      NVIDIA RTX 1000 Ada ...
spconv            PASS      cu120           2.3.6
torch-scatter     PASS                      2.1.2+pt21cu121
torch-cluster     PASS                      1.6.3+pt21cu121
einops            PASS                      0.8.0
timm              PASS                      1.0.9
addict            PASS                      2.4.0
peft              PASS      <= 0.9          0.9.0
accelerate        PASS                      0.33.0
wandb             PASS                      0.17.9
...
PTv3 (Pointcept)  PASS
...
  32/32 passed
```

---

## Project structure

```
AdPerception/
|
├── configs/
│   ├── model/
│   │   ├── bevfusion.yaml          # full architecture config
│   │   └── ptv3_backbone.yaml      # backbone only
│   └── dataset/
│       └── nuscenes.yaml           # paths, splits, voxel params
|
├── csrc/
│   └── bev_pool/                   # CUDA BEV pooling (Phase 2)
│       ├── bev_pool.cpp
│       ├── bev_pool_cuda.cu
│       └── setup.py
|
├── data/
│   └── nuscenes -> /path/to/nuscenes   # symlink - do not copy data
|
├── checkpoints/                    # pretrained weights (git-ignored)
|
├── src/
│   ├── datasets/
│   │   └── nuscenes_loader.py
│   ├── models/
│   │   ├── backbone/
│   │   │   └── ptv3_wrapper.py     # PTv3 wrapper around Pointcept
│   │   ├── camera/
│   │   │   └── lss.py              # Lift-Splat-Shoot
│   │   ├── fusion/
│   │   │   └── bev_encoder.py      # concat + ConvEncoder
│   │   ├── heads/
│   │   │   ├── centerpoint.py      # heatmap detection head
│   │   │   └── seg_head.py         # per-BEV-cell segmentation
│   │   └── pipeline.py             # full model assembly
│   ├── tracking/
│   │   ├── simpletrack_wrapper.py  # SimpleTrack MOT interface
│   │   └── visualization.py        # BEV + 3D dashboard
│   └── utils/
│       ├── voxelizer.py            # point cloud -> spconv tensor
│       ├── postprocess.py          # NMS, heatmap decoding
│       └── metrics.py              # mAP, NDS, mIoU
|
├── tests/
├── third_party/
│   └── Pointcept/                  # git-ignored (own repository)
|   └── SimpleTrack/
|
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dependency conflict notes

Conflicts resolved during setup - do not upgrade these packages:

| Package | Pinned version | Reason |
|---------|---------------|--------|
| `numpy` | 1.26.4 | PyTorch 2.1.2 compiled against NumPy 1.x |
| `peft` | 0.9.0 | peft >= 0.10 requires PyTorch >= 2.4 |
| `matplotlib` | 3.5.3 | nuscenes-devkit requires matplotlib < 3.6 |
| `shapely` | 1.8.5 | nuscenes-devkit requires shapely < 2.0 |
| `lap` | replaced by `lapx` | lap 0.4.0 has no Python 3.11 wheel |
| `gcc` | 12 | CUDA 12.1 does not support GCC 13+ |

---

## Roadmap

### Phase 1 - LiDAR detection + MOT
- [x] `src/utils/voxelizer.py` - point cloud to spconv tensor
- [x] `src/models/backbone/ptv3_wrapper.py` PTv3 feature extraction
- [x] `src/models/heads/centerpoint.py` - heatmap detection head
- [x] `src/models/pipeline.py`           - complete pipeline for Lidar Perception
- [x] `src/tracking/simpletrack_wrapper.py` - SimpleTrack MOT
- [x] `src/tracking/visualization.py` - BEV + 3D dashboard

### Phase 2 - Camera fusion (BEVFusion)
- [ ] `src/models/camera/lss.py` - Lift-Splat-Shoot
- [ ] `src/models/fusion/bev_encoder.py` - BEV fusion encoder
- [ ] `csrc/bev_pool/` - CUDA BEV pooling kernel
- [ ] `src/models/heads/seg_head.py` - BEV segmentation head

### Phase 3 - Benchmark
- [ ] nuScenes 3D detection (mAP, NDS)
- [ ] nuScenes MOT (AMOTA, AMOTP)
- [ ] Inference profiling - target < 100ms @ 10 Hz