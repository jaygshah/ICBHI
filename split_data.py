import pandas as pd 
import numpy as np
import librosa
import argparse
import os
from sklearn.model_selection import train_test_split, KFold

DATA_PATH = './ICBHI_final_database/'
PROCESSED_DATA_PATH = './processed_data/'

def extract_mfcc(row):
    path = PROCESSED_DATA_PATH + row['processed_filename']
    soundArr, sample_rate = librosa.load(path)
    return librosa.feature.mfcc(y=soundArr, sr=sample_rate)

def main(split_type='mfcc'):
    processed_data = pd.read_csv('./processed_data.csv')
    train_test_data = pd.read_csv(f'./ICBHI_challenge_train_test.txt', names=['filename', 'split'], delimiter='\t')

    split_merged_data = pd.merge(processed_data, train_test_data, on='filename')

    train_data = split_merged_data[split_merged_data['split'] == 'train']
    test_data = split_merged_data[split_merged_data['split'] == 'test']

    print(f"Original | train: {train_data.shape[0]}, test: {test_data.shape[0]}")

    unique_pids = sorted(test_data.pid.unique())
    index_75_percent = int(len(unique_pids) * 0.75)
    threshold_pid = unique_pids[index_75_percent]

    train_data = pd.concat([train_data, test_data[test_data.pid <= threshold_pid]])
    test_data = test_data[test_data.pid > threshold_pid]

    print(f"After thresholding original test set at 75% | train: {train_data.shape[0]}, test: {test_data.shape[0]}")

    # TODO: check if any pid overlaps between train, val, and test

    X_train, X_val, y_train, y_val = train_test_split(train_data, train_data.disease, test_size=0.1, random_state=42, stratify=train_data.disease)
    print(f"After split | train: {X_train.shape[0]}, val: {X_val.shape[0]}, test: {test_data.shape[0]}")

    if split_type == 'mfcc':
            
        os.makedirs(f"./{split_type}", exist_ok=True)
        
        x_train = np.array(X_train.apply(extract_mfcc, axis=1).tolist())
        x_val = np.array(X_val.apply(extract_mfcc, axis=1).tolist())
        x_test = np.array(test_data.apply(extract_mfcc, axis=1).tolist())
        y_train = np.array(y_train)
        y_val = np.array(y_val)
        y_test = np.array(test_data.disease)

        np.save(f'./{split_type}/x_train.npy', x_train)
        np.save(f'./{split_type}/x_val.npy', x_val)
        np.save(f'./{split_type}/x_test.npy', x_test)
        np.save(f'./{split_type}/y_train.npy', y_train)
        np.save(f'./{split_type}/y_val.npy', y_val)
        np.save(f'./{split_type}/y_test.npy', y_test)
    
    else:
        raise Exception("Invalid split type.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Choose the type of processing.')
    parser.add_argument('--split_type', type=str, choices=['mfcc'], required=False, default="mfcc", help='Type of data processing to perform.')
    args = parser.parse_args()
    main(args.split_type)