# CPP-Net

Code for "Common Pattern Prior-Driven Semi-Supervised Medical Image Segmentation".

## Requirements

- Python >= 3.8
- PyTorch >= 1.10
- SimpleITK
- medpy
- nibabel
- scikit-image
- tensorboardX
- tqdm
- h5py


## Training

All training scripts support `--root_path` to specify the data directory (default: `./data_split/<dataset>`).

### ACDC

```bash
python code/ACDC_BCP_train.py --root_path ./data_split/ACDC --model unet --labelnum 7 --gpu 0
```

## Testing

```bash
python code/test_ACDC.py --root_path ./data_split/ACDC --model unet --gpu 0
```

## Citation

```bibtex
@article{fang2026common,
  title={Common Pattern Prior-Driven Semi-Supervised Medical Image Segmentation},
  author={Fang, Lexin and Xu, Yunyang and Zhang, Anxin and Li, Xin and Li, Xuemei and Zhang, Caiming},
  journal={IEEE Transactions on Medical Imaging},
  year={2026},
  publisher={IEEE}
}
```
