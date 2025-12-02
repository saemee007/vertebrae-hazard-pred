# Enhancing Vertebral Fracture Prediction Using Multi-Task Deep Learning  

Official implementation of the paper:  
"Enhancing Vertebral Fracture Prediction Using Multi-Task Deep Learning: Computed Tomography Imaging of Bone and Muscle"

<p align="center">
  <img src="figure/main_figure_v4.png" alt="Main Figure" width="700"/>
</p>

---

## 🛠️ Environment Setup

We recommend using **Python 3.9+** and setting up a virtual environment:

```bash
git clone https://github.com/saemee007/vertebrae-hazard-pred.git
cd vertebrae-hazard-pred/

# Install dependencies
pip install -r requirements.txt
```
---
## 📦 Data Setup

Due to medical data privacy, we do not distribute the dataset publicly.  


To reproduce our results, organize the data as follows:

```bash
data/
├── cls/
│   └── nifti/
│       ├── 001_0000.nii.gz
│       ├── 002_0000.nii.gz
│       ├── ...
└── pred/
    └── nifti/
        ├── 001_0000.nii.gz
        ├── 002_0000.nii.gz
        ├── ...

```
- Based on the above data structure, you need to add CSV files containing the labels for each dataset. Refer to `data/datasets/example_cls_data.csv` and `data/datasets/example_pred_data.csv` for guidance on how to structure your label CSV files.

- And update config with your local data paths.

---
## 🔄 Data Preprocessing

Run the preprocessing pipeline to prepare the data for training:
```bash
python data_preprocess.py --config config/data_preprocess_config.yaml
```
---
## 🧠 Training

To train the multi-task model:
```bash
python train_MTL.py --config config/config.yaml --cls_data data/datasets/cls_data.csv  --pred_data data/datasets/pred_data.csv --slices_select random --view axial --task MTL --gpus 0 --split 0 --sort_slice --nashmtl 
```
---
## 📋 Citation

If you find this work helpful, please cite:
```bibbex
@article{yourbibtex2025,
  title={Enhancing Vertebral Fracture Prediction Using Multi-Task Deep Learning: Computed Tomography Imaging of Bone and Muscle},
  author={Kong et al.},
  journal={...},
  year={2025}
}
```

---
## 📧 Contact

For questions, please contact:
📮 saemee0007@gmail.com
