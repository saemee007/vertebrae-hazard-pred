# Enhancing Vertebral Fracture Prediction Using Multi-Task Deep Learning  
**Computed Tomography Imaging of Bone and Muscle**

Official implementation of the paper:  
**"Enhancing Vertebral Fracture Prediction Using Multi-Task Deep Learning: Computed Tomography Imaging of Bone and Muscle"**

---

## 📁 Repository Structure
├── data/ # Data directory (not included in repo)
├── preprocessing/ # Data preprocessing scripts
├── models/ # Model definitions
├── training/ # Training scripts
├── evaluation/ # Evaluation scripts
├── utils/ # Utility functions
├── configs/ # Configuration files
└── main.py # Entry point for training/evaluation

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

Obtain the CT scan data with bone and muscle segmentation labels (see paper for details).
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
python preprocessing/preprocess.py --config configs/path_config.yaml
```

This will:

Normalize CT images
Align muscle and bone segmentations
Generate input tensors and labels

---
## 🧠 Training

To train the multi-task model:
```bash
python main.py --mode train --config configs/train_config.yaml
```
Features:

Multi-task architecture: fracture classification + segmentation
Configurable loss balancing
Optional pretraining

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