import pandas as pd 
import os
import csv
from sklearn.model_selection import train_test_split, KFold

# paths
DATA_PATH = './ICBHI_final_database/'
PROCESSED_DATA_PATH = './processed_data/'
HELDOUT_TEST_PATH = './heldout_test/'
os.makedirs(HELDOUT_TEST_PATH, exist_ok=True)  # Create the heldout test directory if it doesn't exist

def main():
    
    processed_data = pd.read_csv('./processed_data.csv')
    train_test_data = pd.read_csv(f'./ICBHI_challenge_train_test.txt', names=['filename', 'split'], delimiter='\t')

    # Merge the processed data with the train-test split information
    split_merged_data = pd.merge(processed_data, train_test_data, on='filename')

    train_data = split_merged_data[split_merged_data['split'] == 'train']
    test_data = split_merged_data[split_merged_data['split'] == 'test']

    print(f"Original | train: {train_data.shape[0]}, test: {test_data.shape[0]}")

    # Get unique patient IDs from the test data and determine the threshold for 75% split
    # This is to supplement the train set with test data (only for experiments)
    unique_pids = sorted(test_data.pid.unique())
    index_75_percent = int(len(unique_pids) * 0.75)
    threshold_pid = unique_pids[index_75_percent]

    # Add 75% of the test data to the train data based on the patient ID threshold
    train_data = pd.concat([train_data, test_data[test_data.pid <= threshold_pid]])
    test_data = test_data[test_data.pid > threshold_pid]

    print(f"After thresholding original test set at 75% | train: {train_data.shape[0]}, test: {test_data.shape[0]}")

    test_data_wavs = test_data['processed_filename'].tolist()
    
    with open(f"./heldout_test.csv", "w") as outfile:
        csvwriter = csv.writer(outfile)
        headers = ["audio_path", "disease"]
        csvwriter.writerow(headers)

        for wav in test_data_wavs:
            # Copy the audio files to the heldout test directory
            os.system(f"cp {PROCESSED_DATA_PATH}{wav} {HELDOUT_TEST_PATH}/{wav}")

            # disease label for the audio file and write to the CSV
            dx = processed_data.loc[processed_data['processed_filename'] == wav, 'disease'].values[0]
            csvwriter.writerow([f"{HELDOUT_TEST_PATH}{wav}", dx])

if __name__ == "__main__":
    main()