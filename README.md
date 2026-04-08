# RUM

RUM is a video object segmentation framework based on SAM2. It improves the robustness of object segmentation in complex scenes through Visual Prototype Memory (VPM), Uncertainty-aware Memory Scheduling (UMS), and Motion-guided Relocalization (MGR).

## File Structure

```text
RUM/
├─ sam2/
│  ├─ modeling/                 # Core model implementation
│  ├─ configs/rum/              # Inference config
│  ├─ configs/rum_training/     # Training config
├─ training/                    # Training entrypoint, loss, trainer, and data utilities
├─ tools/
│  └─ vos_inference.py          # VOS inference script
└─ LICENSE
```

## Environment

Please follow the instruction of [official SAM 2 repo](https://github.com/facebookresearch/sam2?tab=readme-ov-file#installation).

## How To Run

### 1. Inference

Default inference config:

- `configs/rum/rum.yaml`

Example:

```bash
python tools/vos_inference.py \
  --rum_cfg configs/rum/rum.yaml \
  --rum_checkpoint /path/to/rum_checkpoint.pt \
  --base_video_dir /path/to/JPEGImages \
  --input_mask_dir /path/to/Annotations \
  --output_mask_dir /path/to/output_masks
```

### 2. Training

Default training config:

- `sam2/configs/rum_training/rum_training.yaml`

Example:

```bash
python training/train.py \
  -c sam2/configs/rum_training/rum_training.yaml \
  launcher.gpus_per_node=1 \
  dataset.mosev1.img_folder=/path/to/mosev1/images \
  dataset.mosev1.gt_folder=/path/to/mosev1/masks \
  dataset.mosev2.img_folder=/path/to/mosev2/images \
  dataset.mosev2.gt_folder=/path/to/mosev2/masks
```

## License

The code in this repository is released under [Apache 2.0](LICENSE).
