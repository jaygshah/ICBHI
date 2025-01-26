import os
import librosa
import pandas as pd
import soundfile as sf
from tqdm import tqdm
import shutil

#####################DOWLOAD DATASET#####################
# Refer to https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip
# unzip ICBHI_final_database.zip
# rm -rf ICBHI_final_database.zip
#####################DOWLOAD DATASET#####################

DATA_PATH = './ICBHI_final_database/'
PROCESSED_DATA_PATH = './processed_data/'
patient_data = pd.read_csv(f'./ICBHI_Challenge_diagnosis.txt', delimiter='\t', names=['pid', 'disease'])

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"The directory {DATA_PATH} does not exist.")

filenames = []
for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith('.txt'):
            filename = file.split('.')[0]
            wav_file = os.path.join(root, filename + '.wav')
            if os.path.exists(wav_file):
                filenames.append(filename)

all_files_data = []
for filename in filenames:
    file_data = pd.read_csv(DATA_PATH + filename + '.txt', sep='\t', names=['start', 'end', 'crackles', 'wheezes'])
    name_parts = filename.split('_')
    file_data['pid'] = name_parts[0]
    file_data['mode'] = name_parts[-2]
    file_data['filename'] = filename
    all_files_data.append(file_data)

files_df = pd.concat(all_files_data).reset_index(drop=True)

patient_data.pid = patient_data.pid.astype('int32')
files_df.pid = files_df.pid.astype('int32')

merged_data = pd.merge(files_df, patient_data, on='pid')
merged_data['disease'] = merged_data['disease'].apply(lambda x: 0 if x == 'Healthy' else 1)

# healthy: 322, sick: 6576
print(f"healthy: {merged_data[merged_data['disease'] == 0].shape[0]}, sick: {merged_data[merged_data['disease'] == 1].shape[0]}")

if os.path.exists(PROCESSED_DATA_PATH):
    user_input = input("The processed data directory already exists. Do you want to remove it? (y/n): ")
    if user_input.lower() == 'y':
        print("Removing existing processed data directory...")
        shutil.rmtree(PROCESSED_DATA_PATH)
    else:
        print("Exiting without removing the processed data directory.")
        exit()

os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

processed_count = 0
processed_filenames = []
for index, row in tqdm(merged_data.iterrows(), total=merged_data.shape[0], desc="Processing audio files"):
    
    max_duration = 8
    start_time = row['start']
    end_time = row['end']
    filename = row['filename']
    
    if end_time - start_time > max_duration:
        end_time = start_time + max_duration
    
    audio_file_path = os.path.join(DATA_PATH, filename + '.wav')
    
    if index > 0 and merged_data.iloc[index - 1]['filename'] == filename:
        file_suffix = merged_data.iloc[:index]['filename'].tolist().count(filename)
    else:
        file_suffix = 0
    
    new_filename = f"{filename}_{file_suffix}.wav"
    # merged_data.at[index, 'filename'] = new_filename
    processed_filenames.append(new_filename)
    save_path = os.path.join(PROCESSED_DATA_PATH, new_filename)
    processed_count += 1
    
    sample_rate = 22050
    audio_data, _ = librosa.load(audio_file_path, sr=sample_rate)

    start_index = min(int(start_time * sample_rate), len(audio_data))
    end_index = min(int(end_time * sample_rate), len(audio_data))
    audio_segment = audio_data[start_index:end_index]
    
    required_length = max_duration * sample_rate
    padded_audio = librosa.util.pad_center(data=audio_segment, size=required_length)
    
    sf.write(file=save_path, data=padded_audio, samplerate=sample_rate)

# Processed 6898 audio files.
print(f"Processed {processed_count} audio files.")

# add a new column to the merged_data dataframe
merged_data['processed_filename'] = processed_filenames
merged_data.to_csv(f'processed_data.csv', index=False)