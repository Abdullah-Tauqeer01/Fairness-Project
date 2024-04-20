# Chest X-Ray Debiasing Using Conditional Generative Adversarial Networks

## Introduction

This project aims to mitigate biases in chest X-ray analysis by employing a conditional generative adversarial network (CGAN) to generate synthetic embedding data for underrepresented groups. The goal is to improve fairness and reduce disparities in the model's predictive performance across different protected groups, specifically focusing on race.

## Dataset

The project uses the MIMIC-CXR dataset, which consists of chest radiographs and associated labels for various pathologies. Instead of working with the original images, the project utilizes vector embeddings of the chest X-rays to reduce computational complexity.

## Methodology

1. **Data Preprocessing**: The dataset is preprocessed to balance the representation of different subgroups by undersampling techniques.

2. **Classification Model**: A Multi-Layer Perceptron (MLP) is trained on the embeddings for binary classification of each label.

3. **Undersampling**: Undersampling is applied to balance the dataset with respect to both race and label by selectively removing excess data from overrepresented groups.

4. **Conditional Generative Adversarial Network (CGAN)**: A tailored CGAN architecture is implemented to generate synthetic embeddings for underrepresented groups. The generator incorporates residual blocks, and the discriminator has a simple structure to prevent mode collapse.

5. **Debiasing**: The trained CGAN model is deployed to generate synthetic embeddings for underrepresented groups, which are then integrated into the original dataset to enhance its diversity and balance.

## Evaluation Methods

Two evaluation methods are employed to assess fairness and performance:

1. **True Positive Rate (TPR) Disparity**: The Difference between the maximum and minimum value of the marginal overall Area Under the receiver operating characteristic Curve (∆mAUC) is minimized to reduce disparities in predictive performance across subgroups.

2. **Maximizing Performance for the Worst-Performing Subgroup**: The differences between the maximum and minimum values of traditional evaluation metrics, such as TPR, false negative rate (FNR), and Brier scores (∆TPR, ∆FNR, and ∆BS), are minimized to ensure adequate performance for the worst-performing subgroup.

## Files

- `cgan.ipynb`: Contains the implementation of the CGAN architecture, including the generator and discriminator models, as well as the training loop and code for generating synthetic embeddings.
- `fairness.py`: Includes functions for data preprocessing, balancing the dataset, training the classification model, and evaluating the model's fairness and performance using the specified metrics.
- `main.py`: The main script that orchestrates the workflow, including data preprocessing, CGAN training, synthetic data generation, and model evaluation.
- `helper.py`: Contains helper functions and classes used throughout the project, such as data loading, preprocessing utilities, and model architectures.

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- TensorFlow

## Usage

1. Clone the repository: `git clone https://github.com/Abdullah-Tauqeer01/Fairness-Project.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Download the embedding and csv data from this link: "https://drive.google.com/drive/folders/1uQiCY41178LY_9tm1mFD9yQZ4o5A-wCi?usp=drive_link"
4.Paste data in data directory
5. Update the file paths and configurations in the code to point to the correct locations of the dataset files.
6.Run fairness.py to get result of FPR and AUC for all data before data augumentation
7. Run the main script: `python main.py`
8.Use cgan.ipynb to genrate the embeddings
9.Use synthesized data fro classification using fairness.py


## Acknowledgments

- [MIMIC-CXR Dataset](https://physionet.org/content/mimic-cxr/2.0.0/)