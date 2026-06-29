# CVLC

The official repository for the **CVLC** algorithm as described in the **Few-Shot Domain Incremental Learning via Continual Vision-Language Consolidation** paper.

## Preparing the datasets

The datasets used in the paper can be downloaded as described in this section. Please note that the "data_path" setting in the .json files must be set to the directories that are used for each dataset.

### CDDB

Please refer to the [project page](https://coral79.github.io/CDDB_web/) of the [CDDB repository](https://github.com/Coral79/CDDB). You can download the dataset from the following [Google Drive](https://drive.google.com/file/d/1NgB8ytBMFBFwyXJQvdVT_yek1EaaEHrg/view?usp=sharing)

### CORe50

The CORe50 dataset can be downloaded from the [Project page](https://vlomonaco.github.io/core50/index.html) of the [CORe50 repository](https://github.com/vlomonaco/core50).

### DomainNet

The DomainNet dataset can be downloaded from [the project page of the Moment Matching for Multi-Source Domain Adaptation](https://ai.bu.edu/M3SDA/).

We provided the split files for each domain for reproducibility. One can remove the DomainNet split files to ???

## Python and libraries

We performed the experiments with the following versions of Python and the main libraries:

- `python==3.14.4`
- `torch==2.12.0`
- `torchvision==0.27.0`
- `numpy==2.4.5`
- `timm==2.4.5`
- `scikit-learn==1.8.0`
- `scipy==1.17.1`
- `einops==0.8.2`

## Running the experiments

You can run the all experiments by using [all.sh](all.sh) script:
```bash
bash all.sh
```

## Ablation studies

You can run the [ablation.sh](ablation.sh) for the ablation studies.
```bash
bash ablation.sh
```

## Acknowledgement

We thank the authors of the following papers and repositories:

- [CL-LoRA](https://github.com/JiangpengHe/CL-LoRA)
- [DCE](https://github.com/Lain810/DCE)
- [SOYO](https://github.com/QWangCV/SOYO)
- [FLOWER]([FLOWER](https://github.com/anwarmaxsum/FLOWER))
- [S3C](https://github.com/JAYATEJAK/S3C) for providing their code, upon which our code is built.
- [BMD](https://github.com/ispc-lab/BMD)
