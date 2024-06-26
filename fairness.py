# -*- coding: utf-8 -*-
"""
fairness.ipynb
"""

import pandas as pd
from IPython.display import clear_output
import io
import os
import glob
import zipfile
import shutil

import numpy as np
import random as python_random
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as sklm
from sklearn.utils import shuffle

seed=19
np.random.seed(seed)
python_random.seed(seed)

# from google.colab import drive
# drive.mount('/content/drive')

# base_path="/content/drive/MyDrive/"

"""### Start here if you have the CSV file"""

df = pd.read_csv("processed_mimic_df.csv")

df.columns

"""### Race prediction using embedding

### Neural Net
"""

# Importing necessary libraries
import numpy as np
import tensorflow as tf  # Import TensorFlow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

def train_model(dataframe, target_column):
    """
    Trains a neural network model to predict a binary outcome based on embeddings using GPU.

    :param dataframe: A pandas DataFrame containing an 'embeddings' column and a binary target column.
    :param target_column: The name of the column in dataframe that is the target binary outcome to predict.
    :return: A tuple of the trained model and its accuracy on the test set.
    """
    # Isolate the embeddings and the target column
    X = np.stack(dataframe['embeddings'].values)
    y = dataframe[target_column].values

    # Encode labels if they are not already numeric
    if not np.issubdtype(y.dtype, np.number):
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the model on GPU
    with tf.device('/GPU:0'):  # Specify GPU device
        model = Sequential([
            Dense(512, input_dim=X.shape[1], activation='relu'),
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        # Compile the model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Train the model
        history = model.fit(X_train, y_train, epochs=10, batch_size=50, validation_split=0.2, verbose=1)

        # Evaluate the model on the test data
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    return model, accuracy

## real one

import pandas as pd
import numpy as np

# Read the CSV file
# df = pd.read_csv("C:\\Users\\atauqeer\\Desktop\\Project\\processed_mimic_df.csv")
# df = pd.read_csv("processed_mimic_df.csv")

# Load the .npz file(Uncomment line for first run)
npz_data = np.load("embeddings_and_paths_2.npz", mmap_mode='r')
# npz_data = np.load("C:\\Users\\atauqeer\\Desktop\\Project\\embeddings_and_paths_2.npz", mmap_mode='r')

file_paths = npz_data['file_paths']
embeddings = npz_data['embeddings']

# Reshape embeddings and select corresponding file paths
all_embeddings = embeddings.reshape(-1, 1376)
selected_file_paths = df['path'].to_list()
file_names = [path.split('/')[-1] for path in selected_file_paths]
file_paths =[i.split('\\')[-1] for i in file_paths]
file_names_set = set(file_names)
print(1,"\n\n\n\n")
selected_indices = [idx for idx, path in enumerate(file_paths) if path in file_names_set]
print(2,"\n\n\n\n")
selected_embeddings = all_embeddings[selected_indices]

# Create final DataFrame
final_df = pd.DataFrame({
    'file_path': [file_paths[idx] for idx in selected_indices],
    'embeddings': [emb.tolist() for emb in selected_embeddings]
})
## for unbalanced
df['path']  = [path.split('/')[-1] for path in df['path']]
final_df = pd.merge(df, final_df, left_on='path', right_on='file_path', how='inner')

### Balancing Dataframe

# Function to balance dataset by disease, race, and label

def balance_dataset_by_disease_race_label(df, disease_labels, races=['BLACK/AFRICAN AMERICAN', 'WHITE']):
    """
    Balances DataFrame 'df' by disease labels and races. Returns dictionary with balanced DataFrames for each disease.

    Parameters:
    df (DataFrame): Input DataFrame.
    disease_labels (list): List of disease labels.
    races (list): List of races to consider. Default is ['BLACK/AFRICAN AMERICAN', 'WHITE'].

    Returns:
    dict: Balanced DataFrames for each disease label.
    """
    balanced_datasets = {}

    for disease_label in disease_labels:
        sampled_dfs = []
        min_counts = []

        for label in [0, 1]:
            for race in races:
                count = df[(df[disease_label] == label) & (df['race'] == race)]['subject_id'].nunique()
                min_counts.append(count)

        min_count = min(min_counts)

        for label in [0, 1]:
            for race in races:
                cases_df = df[(df[disease_label] == label) & (df['race'] == race)]
                sampled_patients = cases_df['subject_id'].drop_duplicates().sample(n=min_count, random_state=42)
                sampled_cases_df = cases_df[cases_df['subject_id'].isin(sampled_patients)]
                sampled_dfs.append(sampled_cases_df)

        balanced_df = pd.concat(sampled_dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        balanced_datasets[f'{disease_label}_balanced'] = balanced_df

    return balanced_datasets

balanced_datasets = balance_dataset_by_disease_race_label(final_df, final_df.columns[11:24])

balanced_df = balanced_datasets[list(balanced_datasets.keys())[8]]
# final_balanced_df = pd.merge(balanced_df[["path", ]], final_df, left_on='path', right_on='path', how='inner')

balanced_df.columns


from sklearn.metrics import roc_auc_score, roc_curve

def calculate_auc_fpr(dataframe, target_column, protected_group_column, trained_model):
    """
    Calculate the AUC and FPR for each protected group in the dataframe.

    :param dataframe: A pandas DataFrame containing an 'embeddings' column, a binary target column,
        and a column specifying the protected group.
    :param target_column: The name of the column in the dataframe that is the target binary outcome to predict.
    :param protected_group_column: The name of the column specifying the protected group.
    :param trained_model: The trained model to use for predictions.
    :return: A dictionary containing AUC and FPR for each protected group.
    """
    results = {}
    protected_groups = dataframe[protected_group_column].unique()
    for group in protected_groups:
        # Filter dataframe for the specific protected group
        group_df = dataframe[dataframe[protected_group_column] == group]
        # Isolate embeddings and target column
        X_group = np.stack(group_df['embeddings'].values)
        y_group = group_df[target_column].values
        # Predict probabilities
        y_pred_proba = trained_model.predict(X_group)
        # Calculate AUC
        auc = roc_auc_score(y_group, y_pred_proba)
        # Calculate FPR
        fpr = roc_curve(y_group, y_pred_proba)
        fpr = fpr[1]  # FPR for positive class
        # Store results
        results[group] = {'AUC': auc, 'FPR': fpr}
    return results


from sklearn.metrics import confusion_matrix

def calculate_fpr(df, target_column, protected_group_column, trained_model):
    """
    Calculate False Positive Rate (FPR) for each protected group.
    """
    embeddings = df['embeddings']  # Assuming 'embeddings' contains the necessary data
    embeddings = np.array([[np.array(sublist) for sublist in sublist_list] for sublist_list in embeddings])

    predictions = trained_model.predict(embeddings)
    predictions = (predictions > 0.5).astype(int)
    true_labels = df[target_column]
    groups = df[protected_group_column].unique()
    fpr_results = {}

    for group in groups:
        group_indices = df[df[protected_group_column] == group].index
        group_predictions = predictions[group_indices]
        group_true_labels = true_labels[group_indices]
        print(group_predictions, group_true_labels, type(group_predictions), type(group_true_labels))
        tn, fp, fn, tp = confusion_matrix(group_true_labels, group_predictions).ravel()
        fpr = fp / (fp + tn) if (fp + tn) != 0 else 0  # Avoid division by zero
        fpr_results[group] = fpr

    return fpr_results


def train_and_evaluate_models(balanced_df, unbalanced_df, dataset_index):
    """
    Trains models on both balanced and unbalanced datasets for the specified dataset index,
    calculates FPR disparity, and displays training accuracy and FPR disparity metrics.

    Args:
    - balanced_datasets: Dictionary of balanced DataFrames.
    - unbalanced_df: DataFrame of the unbalanced dataset.
    - dataset_index: Index of the dataset to process.
    """
    keys = list(balanced_datasets.keys())
    if dataset_index < len(keys):
        k = keys[dataset_index]
        # print(f"Dataset: {k}")

        # # Process balanced dataset
        # balanced_df = balanced_datasets[k]
        trained_model_balanced, accuracy_balanced = train_model(balanced_df, target_column="No_Finding")
        auc_fpr_results_balanced = calculate_auc_fpr(balanced_df, target_column="No_Finding", protected_group_column='race', trained_model=trained_model_balanced)

        # Process unbalanced dataset
        trained_model_unbalanced, accuracy_unbalanced = train_model(unbalanced_df, target_column="No_Finding")
        auc_fpr_results_unbalanced = calculate_auc_fpr(unbalanced_df, target_column="No_Finding", protected_group_column='race', trained_model=trained_model_unbalanced)

        # Display results for balanced dataset
        print("Balanced Dataset:")
        print(f"Training Accuracy: {accuracy_balanced:.4f}")
        print("FPR Disparity:")
        for group, metrics in auc_fpr_results_balanced.items():
            print(f"Protected Group: {group}, AUC: {metrics['AUC']:.4f}")

        fpr_results_unbalanced = calculate_fpr(unbalanced_df, target_column="No_Finding", protected_group_column='race', trained_model=trained_model_unbalanced)
        
        # Display results for unbalanced dataset
        print("\nUnbalanced Dataset:")
        print(f"Training Accuracy: {accuracy_unbalanced:.4f}")
        print("FPR Disparity:")
        for group, metrics in auc_fpr_results_unbalanced.items():
            print(f"Protected Group: {group}, AUC: {metrics['AUC']:.4f}")
        print()
        

        # Display training accuracy and FPR disparity for the unbalanced dataset
        print("Unbalanced Dataset:")
        print(f"Training Accuracy: {accuracy_unbalanced:.4f}")
        print("FPR Disparity:")
        for group, fpr in fpr_results_unbalanced.items():
            print(f"Protected Group: {group}, FPR: {fpr:.4f}")
        print()
        
    else:
        print("Invalid dataset index.")


final_df.to_csv('unbalanced_df.csv.gz', compression='gzip', index=False)
balanced_df.to_csv('balanced_df.csv.gz', compression='gzip', index=False)

# array = final_df.values
# np.savez_compressed('unbalanced_df.npz', df_array=array)
# array = balanced_df.values
# np.savez_compressed('balanced_df.npz', df_array=array)
print('training',"\n\n\n\n")
train_and_evaluate_models(balanced_df, unbalanced_df=final_df, dataset_index=8)

balanced_df.columns


def evaluate_model_fpr(balanced_datasets, unbalanced_df, target_dataset_index):
    """
    Evaluates FPR disparity and training accuracy for both balanced and unbalanced datasets
    at the specified index. Only the unbalanced dataset processing is included as per the provided code.

    Args:
    - balanced_datasets: Dictionary containing balanced DataFrames keyed by disease labels.
    - unbalanced_df: DataFrame containing the unbalanced dataset.
    - target_dataset_index: Index of the dataset to evaluate (0-based).
    """
    # keys = list(balanced_datasets.keys())
    # if target_dataset_index < len(keys):
    #     k = keys[target_dataset_index]
    #     print(f"Dataset: {k}")

    # Train model and calculate FPR on the unbalanced dataset
    trained_model_unbalanced, accuracy_unbalanced = train_model(unbalanced_df, target_column="No_Finding")

    # Display training accuracy and FPR disparity for the unbalanced dataset
    print("Unbalanced Dataset:")
    print(f"Training Accuracy: {accuracy_unbalanced:.4f}")
    print("FPR Disparity:")
    for group, fpr in fpr_results_unbalanced.items():
        print(f"Protected Group: {group}, FPR: {fpr:.4f}")
    print()

evaluate_model_fpr(balanced_df, unbalanced_df=final_df, target_dataset_index=8)

