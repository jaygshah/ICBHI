## Downloading dataset

Download the zip file from [here](https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip) along with [diagnosis labels](https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_Challenge_diagnosis.txt) and [train/test split](https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_challenge_train_test.txt) information.

## 🛠️ Requirements and Installation
* Python >= 3.8
* Pytorch == 2.4.1
* CUDA Version >= 12.6
* Install required packages:
```bash
conda create -n icbhi python=3.8 -y
conda activate icbhi
pip3 install -r requirements.txt
pip3 install torch torchvision torchaudio
```

## Data Preparation
First, we extract the relevant segment from all audio files, pad it to a fixed length, and saves it to the `processed_data` directory.
```
python3 data_preprocess.py
```
Next, we extract Mel-frequency cepstral coefficients (MFCCs) from audio files and split the data into training, validation, and test sets. 
Note: I have supplemented train set here with some samples from test set (only for experiments purpose)
```
python3 split_data.py --split_type mfcc
```

`python3 prepare_heldout_set.py` was used to create the heldout_test set from ICBHI dataset.

## Training
```
python3 train.py --config_file ./configs/train_config.yaml
```
The script uses a `custom_model` by-default. Change to other supported models such as resnet18, resnet34 or resnet50 in config file.

## Inference
```
python3 inference.py --config_file ./configs/inference_config.yaml
```
For evaluating on custom data, folder directory should have structure as `heldout_test` set here with a supporting `heldout_test.csv` with label information (for evaluating)

```
heldout_test
├── audio1.wav
├── audio2.wav
├──...
```

## Logs

`mfcc_custom_saved_models/logs/training.log`: training log using custom model with early stopping and default training-config setting

`heldout_test_custom_results/logs/testing.log`: inference log on heldout_test set from ICBHI dataset using trained checkpoints

`heldout_test_custom_results/confusion_matrix.png`: confusion matrix of predicted labels on heldout_test set