# Chromosome-SCL-Encoder

Official implementation of our ICIP 2025 paper:

> **Visual Encoders for Generalized Chromosome Recognition**  
> Ruijia Chang†, Tao Zhou†, Suncheng Xiang, Yujia Wang, Kui Su*, Yin Zhou*, Dahong Qian*, Jun Wang*  
> † Co-first authors. * Corresponding authors.

---

## 📂 Project Structure

```bash
├── cfgs/           # Configuration files for training/evaluation
├── datasets/       # Data loading and preprocessing scripts
├── losses/         # Custom loss functions
├── models/         # Network architectures
├── trainers/       # Training and evaluation pipelines
├── train.py        # Entry point for model training
├── test.py         # Script for evaluation
````
---

## 📦 Pretrained Checkpoints

* `vit_small_patch8_224` 
  🔗 [Download best\_weights.pth](https://drive.google.com/drive/folders/119DsFWtH2I0XKpQS58VDzTxQMmaBFKyS)
* `vit_large_patch14_224` 
  🔗 [Download best\_weights.pth](https://drive.google.com/drive/folders/1czzbnQAmyfm0Vmne43T-qHltJq_9t8_b)
* `davit_small` 
  🔗 [Download best\_weights.pth](https://drive.google.com/drive/folders/1_BItztBrBIRmAe0wJIAmDID3jiHNExvy)
* `davit_large` 
  🔗 [Download best\_weights.pth](https://drive.google.com/drive/folders/1zb5RhXRP-POwGiypDMIAlzllfIly40Pl)
* `swin_small_patch4_window7_224` 
  🔗 [Download best\_weights.pth](https://drive.google.com/drive/folders/1TWL7FMCElxFFHKoAG_dk7UQ-zf1XZiLP)
* `swin_large_patch4_window7_224` 
  🔗 [Download best\_weights.pth](https://drive.google.com/drive/folders/1kt5YKHqtBd4R_BW74y5MXLmrUTb8N5UN)
* `convnextv2_large` 
  🔗 [Download best\_weights.pth](https://drive.google.com/drive/folders/1bmNrq2JgQMzccEgdo3nVNALL49y5qrFq)
* `resnet50` 
  🔗 [Download best\_weights.pth](https://drive.google.com/drive/folders/13l_2QI9EMfyaX9qPbalUhzjLme-AViSg)
---


## 🚀 Getting Started

### 1. Install dependencies

We recommend using a virtual environment (e.g., conda):

```bash
pip install -r requirements.txt
```
---

### 2. Prepare the dataset

Prepare your chromosome classification data in **LabelMe JSON format**, and organize it under a `data/` folder.
The dataset should be split into three separate JSON annotation files:

```
data/
├── train.json
├── val.json
└── test.json
```

Each JSON file should follow the standard LabelMe structure, containing image paths and corresponding annotations.

---

### 3. Train the model

You can modify the training config in the `cfgs/` directory.

```bash
python train.py 
-encoder=vit_small_patch8_224 
-input_size=224 
-data_aug=rot-vflip-jit -save_dir=vit_small_patch8_224_aug  
-epochs=100 
-steps_per_epoch=3000 
-lr=0.00001 
```

---

### 4. Evaluate the model

```bash
python test.py 
-encoder=vit_small_patch8_224 -im_path=your_data_root_path 
-input_size=224 
-save_dir=checkpoints/vit_small_patch8_224_aug
-load=checkpoints/vit_small_patch8_224_aug/best_weights.pth
```

## 📄 Citation

If you use this code or build on our work, please cite:

```bibtex
xx
```

