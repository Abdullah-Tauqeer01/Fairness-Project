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

The project evaluates the fairness and performance of the binary classification models using the following metrics:

1. **False Positive Rate (FPR) Disparity**: The FPR is calculated for each protected group (race) using the `calculate_fpr` function. The FPR disparity across different protected groups is reported to assess the fairness of the model's predictions.

2. **Training Accuracy**: The overall training accuracy of the binary classification model is reported for both the balanced and unbalanced datasets.

3. **Area Under the Curve (AUC)**: The AUC is calculated for each protected group using the `calculate_auc_fpr` function. The AUC disparity across different protected groups is reported as an additional fairness metric.

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

1. Begin by cloning the repository: `git clone https://github.com/Abdullah-Tauqeer01/Fairness-Project.git`.
2. Install the necessary dependencies by executing: `pip install -r requirements.txt`.
3. Download the embedding and CSV data from the following link: [Data Link](https://drive.google.com/drive/folders/1uQiCY41178LY_9tm1mFD9yQZ4o5A-wCi?usp=drive_link).
4. Create a directory named "Data" and place the downloaded data into it.
5. Update the file paths and configurations within the code to correctly reference the locations of the dataset files.
6. Run the `fairness.py` script to obtain the results of False Positive Rate (FPR) and Area Under the Curve (AUC) for all data before data augmentation.
7. Execute the main script by running: `python main.py`.
8. Utilize `cgan.ipynb` to generate the embeddings.
9. Employ the synthesized data for classification purposes using `fairness.py`.

## Acknowledgments

- [MIMIC-CXR Dataset](https://physionet.org/content/mimic-cxr/2.0.0/)
