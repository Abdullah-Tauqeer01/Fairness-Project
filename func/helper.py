
######## Imports

import numpy as np
import random as python_random
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as sklm
from sklearn.utils import shuffle
# Importing necessary libraries
import numpy as np
import tensorflow as tf  # Import TensorFlow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from ast import literal_eval
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix


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
        history = model.fit(X_train, y_train, epochs=10, batch_size=50, validation_split=0.2, verbose=0)

        # Evaluate the model on the test data
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    return model, accuracy


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


def train_and_evaluate_models(balanced_df, unbalanced_df, dataset_index,balanced_datasets,target):
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
        trained_model_balanced, accuracy_balanced = train_model(balanced_df, target_column=target)
        auc_fpr_results_balanced = calculate_auc_fpr(balanced_df, target_column=target, protected_group_column='race', trained_model=trained_model_balanced)

        # Process unbalanced dataset
        trained_model_unbalanced, accuracy_unbalanced = train_model(unbalanced_df, target_column=target)
        auc_fpr_results_unbalanced = calculate_auc_fpr(unbalanced_df, target_column=target, protected_group_column='race', trained_model=trained_model_unbalanced)

        # Display results for balanced dataset
        print("Balanced Dataset:")
        print(f"Training Accuracy: {accuracy_balanced:.4f}")
        print("FPR Disparity:")
        for group, metrics in auc_fpr_results_balanced.items():
            print(f"Protected Group: {group}, AUC: {metrics['AUC']:.4f}")

        fpr_results_unbalanced = calculate_fpr(unbalanced_df, target_column=target, protected_group_column='race', trained_model=trained_model_unbalanced)
        
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


def evaluate_model_fpr(balanced_datasets, unbalanced_df, target_dataset_index, target):
    """
    Evaluates FPR disparity and training accuracy for both balanced and unbalanced datasets
    at the specified index. Only the unbalanced dataset processing is included as per the provided code.

    Args:
    - balanced_datasets: Dictionary containing balanced DataFrames keyed by disease labels.
    - unbalanced_df: DataFrame containing the unbalanced dataset.
    - target_dataset_index: Index of the dataset to evaluate (0-based).
    """
    # Train model and calculate FPR on the unbalanced dataset
    trained_model_unbalanced, accuracy_unbalanced = train_model(unbalanced_df, target_column=target)

    # Initialize lists to store FPR results
    protected_groups = []
    fpr_values = []

    # Calculate FPR for each protected group in unbalanced dataset
    # Assuming you have a function fpr_calculator that calculates FPR for different groups
    # You would need to replace it with actual logic
    # fpr_results_unbalanced = fpr_calculator(trained_model_unbalanced, unbalanced_df, protected_groups)
    # for group, fpr in fpr_results_unbalanced.items():
    #     protected_groups.append(group)
    #     fpr_values.append(fpr)

    # Create new columns in the DataFrame to store evaluation results
    unbalanced_df[f"Accuracy_Target{target_dataset_index}"] = accuracy_unbalanced
    for i, group in enumerate(protected_groups):
        unbalanced_df[f"FPR_{group}_Target{target_dataset_index}"] = fpr_values[i]

    return unbalanced_df



############################################## CLASSES AND FUNCTIONS FOR CGAN ###############################

def string_to_float_list(string):
    string = string.strip('[]')
    numbers = string.split(', ')
    float_list = [float(num) for num in numbers]
    return float_list
    


#################### Simpler version of GAN ######################

# class Discriminator(nn.Module):
#     def __init__(self, embedding_dim, race_dim, hidden_dim):
#         super(Discriminator, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(embedding_dim + race_dim, hidden_dim),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(hidden_dim, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, embeddings, conditions):
#        # print(conditions.shape, )
#         conditions = conditions.float()
#         combined = torch.cat([embeddings, conditions], dim=1)
#         return self.net(combined)

# class Generator(nn.Module):
#     def __init__(self, embedding_dim, race_dim, hidden_dim):
#         super(Generator, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(embedding_dim + race_dim, hidden_dim),
#             nn.ReLU(True),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(True),
#             nn.Linear(hidden_dim, embedding_dim),
#             nn.Tanh()
#         )

#     def forward(self, embeddings, conditions):
       
#         conditions = conditions.float()
#         combined = torch.cat([embeddings, conditions], dim=1)
#         return self.net(combined)




class Discriminator(nn.Module):
    def __init__(self, embedding_dim, race_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim + race_dim,1),
#            nn.LeakyReLU(0.2, inplace=True),
#            nn.Linear(hidden_dim, hidden_dim),
#            nn.LeakyReLU(0.2, inplace=True),
#            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, embeddings, conditions):
        conditions = conditions.float()
        combined = torch.cat([embeddings, conditions], dim=1)
        return self.net(combined)


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(True),
            nn.BatchNorm1d(out_features),
            nn.Linear(out_features, out_features),
            nn.ReLU(True),
            nn.BatchNorm1d(out_features)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, embedding_dim, race_dim, hidden_dim):
        super(Generator, self).__init__()
        self.initial = nn.Linear(embedding_dim + race_dim, hidden_dim)
        self.residual_blocks = nn.Sequential(
            ResidualBlock(hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim)
        )
        self.final = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Tanh()
        )

    def forward(self, embeddings, conditions):
        conditions = conditions.float()
        combined = torch.cat([embeddings, conditions], dim=1)
        x = self.initial(combined)
        x = self.residual_blocks(x)
        return self.final(x)


import pandas as pd
from torch.utils.data import Dataset
import torch

class EmbeddingDataset(Dataset):
    def __init__(self, npz_file):
        data = pd.read_csv(npz_file, compression='gzip')
       # print(data.head())
       # print(data.columns)
        self.embeddings = data['embeddings']
        self.races = data['race']

        self.race_to_index = {race: idx for idx, race in enumerate(np.unique(self.races))}

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
       # print(idx, "---------------------")
        embedding_value = string_to_float_list(self.embeddings[idx])
        embedding = torch.tensor(embedding_value, dtype=torch.float)
        race_index = torch.tensor(self.race_to_index[self.races[idx]], dtype=torch.long)

        return embedding, race_index




def generate_embeddings(generator, num_examples, race_index, embedding_dim, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    generator.eval()

    noise = torch.randn(num_examples, embedding_dim, device=device)

    conditions = torch.full((num_examples,), race_index, dtype=torch.long, device=device)
    conditions= F.one_hot(conditions, num_classes=6 )
    with torch.no_grad():
        
        generated_embeddings = generator(noise, conditions)

    return generated_embeddings