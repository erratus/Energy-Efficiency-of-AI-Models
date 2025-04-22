# 💡 Sustainable AI: Carbon Footprint Analysis of Popular AI Models

This repository contains an in-depth comparative analysis of various AI models based on their **energy consumption**, **training time**, **test accuracy**, and **loss metrics**. The goal is to identify energy-efficient alternatives for sustainable AI development.

## 📊 Project Overview

As the demand for large-scale AI models grows, so does their environmental impact. This study evaluates the carbon footprint of multiple popular AI architectures to guide researchers and developers in adopting more sustainable AI practices.

## 🔍 Models Evaluated

- **Improved BERT**
- **CNN**
- **DistilBERT**
- **Improved GLAM**
- **GPT-2**
- **ViT (Vision Transformer)**
- **XGBoost + ResNet18**

Each model was tested under controlled conditions, and metrics such as training time, energy usage (in Joules), accuracy, and loss were recorded.

## 📈 Key Metrics

| Model               | Training Time (s) | Energy (J)   | Test Accuracy (%) | Test Loss  |
|---------------------|-------------------|--------------|-------------------|------------|
| Improved BERT       | 767.12            | 15,921.49    | 60.74             | 1.0975     |
| CNN                 | 329.52            | 3,000.64     | 72.12             | 0.7920     |
| DistilBERT          | 703.70            | 16,591.18    | 53.36             | 1.4873     |
| Improved GLAM       | 205.48            | 1,938.10     | 74.76             | 1.3930     |
| GPT-2               | 1443.05           | 33,751.45    | 55.84             | 1.7020     |
| ViT                 | 5019.22           | 351,345.45   | 93.88             | 0.1927     |
| XGBoost + ResNet18  | 35.70             | 786.73       | 61.31             | -          |

## 🧪 Technologies Used

This project leverages a variety of Python libraries across machine learning, data analysis, and energy profiling:

### 🔍 Core Libraries for Model Training & Evaluation
- **PyTorch** (`torch`, `torchvision`, `torchaudio`) – Deep learning framework for training models
- **Transformers** (`transformers`, `tokenizers`) – Pre-trained NLP models like BERT, GPT-2, DistilBERT
- **scikit-learn** – Classical ML models (e.g., XGBoost), evaluation metrics
- **XGBoost** – Gradient boosting algorithm used with ResNet18
- **Timm** – PyTorch image models including ViT

### ⚡ Energy & Performance Profiling
- **nvidia-ml-py**, **pynvml**, **nvidia-smi** – GPU power and usage tracking
- **codecarbon** (implied by energy metrics) – Carbon footprint estimation (you can list if used)

### 📊 Data Handling & Visualization
- **Pandas**, **NumPy** – Data manipulation and numerical operations
- **Matplotlib**, **Seaborn** – Plotting and visualization of metrics

### 📚 Datasets & Utilities
- **Datasets** (HuggingFace) – Easy access to benchmark datasets
- **Huggingface Hub** – Model and dataset hosting
- **TQDM** – Progress bars during training


## 📌 Findings

- **ViT** achieves the highest accuracy but is extremely power-hungry.
- **CNN** and **Improved GLAM** strike a balance between accuracy and low energy consumption.
- **XGBoost + ResNet18** is the most lightweight and energy-efficient for moderate accuracy requirements.

## 📁 Folder Structure

├── graphs/  # Visualizations of results  <br>
├── helpers/ # Dataset used for experiments <br>
├── models/ # Model training scripts <br>
├── results/ # Tabulated outputs and logs <br> 
└── README.md # Project documentation <br>


## 📝 Citation

If you use this work, please consider citing or referencing the project in your research or presentation.

## 🤝 Contributing

Feel free to open issues or submit pull requests if you'd like to enhance the study or extend it to more models.

## 📧 Contact

For questions or collaboration, contact: **[Your Email or GitHub handle]**
