

from func.helper import train_model,balance_dataset_by_disease_race_label,calculate_auc_fpr,calculate_fpr,train_and_evaluate_models,evaluate_model_fpr,balance_dataset_by_disease_race_label


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
# Importing necessary libraries
import numpy as np
import tensorflow as tf  # Import TensorFlow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, roc_curve


from sklearn.metrics import confusion_matrix





seed=19
np.random.seed(seed)
python_random.seed(seed)


"""### Start here if you have the CSV file"""

df = pd.read_csv("processed_mimic_df.csv")

df.columns

"""### Race prediction using embedding

### Neural Net
"""


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
selected_indices = [idx for idx, path in enumerate(file_paths) if path in file_names_set]

selected_embeddings = all_embeddings[selected_indices]

# Create final DataFrame
final_df = pd.DataFrame({
    'file_path': [file_paths[idx] for idx in selected_indices],
    'embeddings': [emb.tolist() for emb in selected_embeddings]
})
df['path']  = [path.split('/')[-1] for path in df['path']]
final_df = pd.merge(df, final_df, left_on='path', right_on='file_path', how='inner')



balanced_datasets = balance_dataset_by_disease_race_label(final_df, final_df.columns[11:24])
print("before balance","\n\n\n")
balanced_df = balanced_datasets[list(balanced_datasets.keys())[8]]

print("after balance","\n\n\n")



if not os.path.exists('unbalanced_df.csv.gz'):
    final_df.to_csv('unbalanced_df.csv.gz', compression='gzip', index=False)
if not os.path.exists('balanced_df.csv.gz'):
    balanced_df.to_csv('balanced_df.csv.gz', compression='gzip', index=False)



# array = final_df.values
# np.savez_compressed('unbalanced_df.npz', df_array=array)
# array = balanced_df.values
# np.savez_compressed('balanced_df.npz', df_array=array)
print('training',"\n\n\n\n")
train_and_evaluate_models(balanced_df, unbalanced_df=final_df, dataset_index=8, balanced_datasets=balanced_datasets)


balanced_df.columns



evaluate_model_fpr(balanced_df, unbalanced_df=final_df, target_dataset_index=8)

