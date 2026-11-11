# EfficienT-HDR

**An Efficient Transformer-Based Framework via Multi-Exposure Fusion for HDR Reconstruction**

Yu-Shen Huang, Tzu-Han Chen, Cheng-Yen Hsiao, and Shaou-Gang Miaou

EfficienT-HDR reconstructs HDR images from three LDR exposures. It combines YCbCr preprocessing, intersection-aware adaptive fusion, and lightweight Transformer blocks. [Paper](https://arxiv.org/abs/2509.19779)

## Models

| `--model` | Python module | Local checkpoint from prior runs |
| --- | --- | --- |
| `main` | `models/efficient_hdr.py` (formerly V11) | `checkpoints/V11_1/best_checkpoint.pth` |
| `lite` | `models/efficient_hdr_lite.py` (formerly V10) | `checkpoints/V10-3/best_checkpoint.pth` |
| `v9` | `models/hdr_transformer_v9.py` (earlier experiment) | `checkpoints/V9-4/best_checkpoint.pth` |

Both entry points default to `lite`, matching the previous training and evaluation scripts. Checkpoint paths refer to local files. The repository tracks source code and setup instructions. Datasets, weights, evaluation outputs, and paper assets are stored locally.

## Setup

Use Python 3.9. The local environment uses PyTorch 2.1.0 and Torchvision 0.16.0 with CUDA 12.1. For another PyTorch build, install the matching `torch` and `torchvision` packages for your platform before installing the requirements.

```sh
python -m pip install -r requirements.txt
```

## Dataset preparation

Download the [Kalantari and Ramamoorthi dataset](https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/). Place training scenes under `data/Training/<scene>/` and test scenes under `data/Test/EXTRA/<scene>/` or `data/Test/PAPER/<scene>/`. Each scene needs three `.tif` exposures, `exposure.txt`, and `HDRImg.hdr`.

Run the following from the repository root to generate `data/sig17_training_crop128_stride64/`.

```sh
python -m dataset.gen_crop_data --data_root ./data
```

## Training

```sh
python train.py --model main --test_split EXTRA --logdir ./checkpoints/main
python train.py --model lite --test_split EXTRA --logdir ./checkpoints/lite
```

Training uses Weights & Biases. Use `wandb login` to sync runs, or set `WANDB_MODE=offline` to log locally. The default loss uses pretrained VGG16 features. Use `--loss_func 0` for μ-law L1 loss without VGG16.

Resume with the same model variant and `--resume PATH`, pointing to a training checkpoint such as `val_latest_checkpoint.pth`.

## Evaluation

```sh
python test.py --model main --test_split EXTRA --pretrained_model ./checkpoints/main/best_checkpoint.pth --save_dir ./results/main
python test.py --model lite --test_split EXTRA --pretrained_model ./checkpoints/lite/best_checkpoint.pth --save_dir ./results/lite
```

Use `--test_split Real` for scenes under `data/Test/Real`. If you omit `--test_split`, the loader checks `Te`, `Real`, and `EXTRA` in that order, then looks for scenes directly under `data/Test`. Evaluation reports linear and μ-law PSNR and SSIM. Use `--no_cuda` for CPU evaluation or `--no-save_results` to print metrics without writing HDR files.

## Citation

```bibtex
@article{huang2025efficienthdr,
  title={EfficienT-HDR: An Efficient Transformer-Based Framework via Multi-Exposure Fusion for HDR Reconstruction},
  author={Huang, Yu-Shen and Chen, Tzu-Han and Hsiao, Cheng-Yen and Miaou, Shaou-Gang},
  journal={arXiv preprint arXiv:2509.19779},
  year={2025}
}
```

This implementation builds on [HDR-Transformer](https://github.com/ljzycmd/HDR-Transformer) by Zhen Liu, Yinglong Wang, Bing Zeng, and Shuaicheng Liu (ECCV 2022).
