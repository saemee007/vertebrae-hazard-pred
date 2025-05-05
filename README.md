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
git clone https://github.com/saemee007/vertebral-fracture-prediction.git
cd vertebral-fracture-prediction

# Install dependencies
pip install -r requirements.txt
```

---
## 📦 Data Setup

Due to medical data privacy, we do not distribute the dataset publicly.
To reproduce our results:

Obtain the CT scan data with bone and muscle segmentation labels.
Organize the data as follows:

```bash
data/
├── images/
│   ├── patient_001.nii.gz
│   ├── ...
├── labels/
│   ├── patient_001_label.json  # Includes fracture label, segmentation annotations
│   ├── ...
```

Update configs/path_config.yaml with your local data paths.

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
python train_MTL.py --config config/config.yaml m--cls_data data/datasets/_cls_data.csv  --pred_data data/datasets/pred_data.csv --slices_select random --view axial --task MTL --gpus 0 --split 0 --sort_slice --nashmtl 
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
