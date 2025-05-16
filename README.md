🧠 Brain Tumor Segmentation with 2D U-Net (BraTS 2020)

A PyTorch and MONAI-based semantic segmentation project for detecting brain tumor regions from MRI scans. Built as part of a computer vision course semester project.

---

🗂️ Project Overview

This project performs pixel-wise segmentation of brain tumors from multi-modal MRI slices using 2D U-Net architectures. It uses 4-channel MRI inputs and converts RGB masks into 5-class categorical masks.

The project supports both:
- ✅ Custom-built PyTorch U-Net
- ✅ MONAI's ready-to-use medical segmentation UNet

---

📁 Dataset

- Source: BraTS 2020 (MICCAI Challenge)
- Format: `.h5` HDF5 files
  - `image`: (H, W, 4) — modalities: T1, T1ce, T2, FLAIR
  - `mask`: (H, W, 3) — RGB segmentation mask
- Classes (after RGB to index mapping):
  - 0: Background
  - 1: Necrotic core (NCR/NET)
  - 2: Edema (ED)
  - 3: Enhancing tumor (ET)
  - 4: Other tumor regions

Note: For training efficiency, only the first 100 volumes were used.

---

🔧 Setup

```bash
git clone https://github.com/your-username/brats-unet.git
cd brats-unet

pip install -r requirements.txt
```

Or in Colab:

```python
!pip install monai torch torchvision matplotlib h5py pandas
```

---

🚀 Training

```bash
python train.py --model monai --epochs 20 --batch_size 8
```

Best weights are saved as `best_monai_unet2d.pth`.

---

🧪 Evaluation & Visualization

```bash
python inference.py --model monai
```

The script displays:

- Input MRI slice
- Ground truth segmentation
- Predicted mask

---

🧠 Sample Result

| Input        | Ground Truth | Prediction   |
|--------------|---------------|--------------|
| ![input.png] | ![gt.png]     | ![pred.png]  |

---

📊 Results (Val Dice Score)

| Epoch | Dice Score |
|-------|------------|
| 1     | 0.1089     |
| 10    | 0.1996     |
| 20    | 0.2000     |

> Due to limited data (100 volumes, no augmentation), results plateau around 0.2. Full data training is expected to improve performance.

---

📌 Features

- 🧠 4-channel MRI input support
- 🎨 RGB → Class label conversion
- 🔀 Dual model support (PyTorch / MONAI)
- 📈 Dice score evaluation
- 🖼️ Mask visualization
- 🧩 Modular structure

---

🔄 Future Improvements

- Train on full BraTS 2020 dataset
- Add data augmentation
- Explore 3D U-Net
- Try alternative models (e.g., SwinUNet, TransUNet)

---

📄 License

MIT License — for academic and research use.
