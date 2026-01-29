# Prototype-Based Test-Time Adaptation of Vision-Language Models

### Install

- Conda environment of image recognition tasks (recommended).

```
# Create a conda environment
conda create -y -n pta python=3.9

# Activate the environment
conda activate pta

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

# Install requirements
pip install -r requirements.txt
```

- Conda environment of robust point cloud analysis tasks (recommended).

```
# Create a conda environment
conda create -n pta_point python=3.8.16

# Activate the environment
codna activate pta_point

# install torch
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# install dassl
cd PTA_point/dassl/
python setup.py develop # (no need to re-build if the source code is modified)
```

### Pre-trained Weights
ULIP: [Link](https://huggingface.co/datasets/auniquesun/Point-PRC/tree/main/pretrained-weights/ulip)
```
weights
├── ulip
│   ├── pointbert_ulip1.pt
│   ├── pointbert_ulip2.pt
│   ├── slip_base_100ep.pt
```

### Datasets

- Please follow the instructions at docs/DATASETS.md to prepare image recognition benchmarks.
- [Link](https://huggingface.co/datasets/auniquesun/Point-PRC/tree/main/new-3ddg-benchmarks/xset/corruption) for modelnet_c and sonn_c.

```
data
├── caltech-101
├── dtd
├── eurosat
├── fgvc_aircraft
├── food-101
├── imagenet
├── imagenet-adversarial
├── imagenet-rendition
├── imagenet-sketch
├── imagenetv2
├── oxford_flowers
├── oxford_pets
├── stanford_cars
├── sun397
├── ucf101
├── modelnet_c
├── sonn_c
│   ├── obj_bg
│   ├── obj_only
│   ├── hardest
```

### How to Run

#### Corss-Domain Generalization

```
bash scripts/run_cd_benchmark_vit.sh
```

#### OOD Generalization

```
bash scripts/run_ood_benchmark_vit.sh
```

#### Robustness evaluation on ModelNet-C

```
# In eval_pta.sh, you can modify the corruption type and severity, for example, 'add_global_2' indicates the corruption type 'add_global' with severity level 2.
bash ./PTA_point/scripts/eval_pta.sh 0 ulip weights/ulip/pointbert_ulip1.pt modelnet_c obj_only 1024 vitg14 ulip1 so_obj_only_9
```

#### Robustness evaluation on SONN-C (obj_only, obj_bg, hardest)

```
# In eval_pta.sh, you can modify the corruption type and severity, for example, 'add_global_2' indicates the corruption type 'add_global' with severity level 2.
bash ./PTA_point/scripts/eval_pta.sh 0 ulip weights/ulip/pointbert_ulip1.pt sonn_c obj_only 1024 vitg14 ulip1 so_obj_only_9
```



