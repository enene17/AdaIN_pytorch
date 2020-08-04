# AdaIN_pytorch

This is an unofficial pytorch implementation of [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868) [Huang+, ECCV2017].

[Official implementation](https://github.com/xunhuang1995/AdaIN-style) is released by the authors.

## Requirements
- Python 3.5+
- Pytorch 1.2.0+

## Usage

### Preprocess

- Download MSCOCO images and Wikiart images.

### Train

```
CUDA_VISIBLE_DEVICES=<gpu_id> python train.py
```

### Test
```
CUDA_VISIBLE_DEVICES=<gpu_id> python test.py --c_image <c_folder_path> --s_image <s_image_name> --output <output_folder_path> --checkpoint <checkpoint_path>
```

## Examples

![Examples](example.png)