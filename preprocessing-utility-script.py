# %% [code]
# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import subprocess

package_name = 'iisignature'

# Run pip install command
subprocess.run(['pip', 'install', package_name])

import os
import numpy as np
import time
import iisignature
import h5py
from tqdm import tqdm

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

train_columns = ['eeg_id','eeg_sub_id','eeg_label_offset_seconds','spectrogram_id',
                 'spectrogram_sub_id','spectrogram_label_offset_seconds','label_id','patient_id']


TARGET_COLUMNS = ['seizure_vote','lpd_vote','gpd_vote','lrda_vote','grda_vote','other_vote']
CLASS_NAMES = ['Seizure', 'LPD', 'GPD', 'LRDA','GRDA', 'Other']
LABEL2NAME = dict(enumerate(CLASS_NAMES))
NAME2LABEL = {v:k for k, v in LABEL2NAME.items()}

EEG_PATH_TEMPL = '/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/'
SP_PATH_TEMPL = '/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/'

WIN_SIZE =  10 # 10 seconds
EEG_FR = 200 # 200 samples per seconds
EEG_T = WIN_SIZE*EEG_FR
CHAINS = {
    'LL' : [("Fp1","F7"),("F7","T3"),("T3","T5"),("T5","O1")],
    'RL' : [("Fp2","F8"),("F8","T4"),("T4","T6"),("T6","O2")],
    'LP' : [("Fp1","F3"),("F3","C3"),("C3","P3"),("P3","O1")],
    'RP' : [("Fp2","F4"),("F4","C4"),("C4","P4"),("P4","O2")]
}
SP_WIN = 600 # 10 minutes = 600 seconds
EGG_WIN = 50 # 50 seconds in total

LABELED_SECS = 10




def get_data(file):
    """Takes in file and returns a pandas data frame of the data
        File should be a .csv """
    
    return pd.read_csv(file)


def get_eeg_sp_data(train_row):
    """Gets EEG and Spectogram data from a specific row in the dataset"""
    
    eeg_id = train_row.eeg_id
    sp_id = train_row.spectrogram_id
    
    eeg_parquet = pd.read_parquet(f'{EEG_PATH_TEMPL}{eeg_id}.parquet')
    sp_parquet = pd.read_parquet(f'{SP_PATH_TEMPL}{sp_id}.parquet')
    
    # offset of data
    eeg_offset = int(train_row.eeg_label_offset_seconds + 20) #only 10 central seconds from 50 secs were labeled, which should be seconds 20-30 in the sample
    sp_offset = int(train_row.spectrogram_label_offset_seconds )
    
    # get spectrogram data
    sp = sp_parquet.loc[(sp_parquet.time>=sp_offset)&(sp_parquet.time<sp_offset+SP_WIN)]
    sp = sp.loc[:, sp.columns != 'time']
    sp = {
        "LL": sp.filter(regex='^LL', axis=1),
        "RL": sp.filter(regex='^RL', axis=1),
        "RP": sp.filter(regex='^RP', axis=1),
        "LP": sp.filter(regex='^LP', axis=1)}
    
    # calculate eeg data
    eeg_data = eeg_parquet.iloc[eeg_offset*EEG_FR:(eeg_offset+WIN_SIZE)*EEG_FR]
    # print(eeg_data.keys()) # Has keys Index(['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz', 'Pz',
                            # 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2', 'EKG']
    # assert 0 == 1
    
    eeg = pd.DataFrame({})
    for chain in CHAINS.keys():
        for s_i, signals in enumerate(CHAINS[chain]):
            diff=eeg_data[signals[0]]-eeg_data[signals[1]] # Subtracts relevant fields as in the image above
            diff.ffill(inplace = True)
            eeg[f"{chain}: {signals[0]} - {signals[1]}"] = diff
    
    return eeg, sp, train_row[TARGET_COLUMNS].values


def signature(data, level=2):
    """Performs signature on data"""
    return iisignature.sig(data, level)

def change_sp_to_array(sp_dict, sig=True):
    """Takes in a dictionary of sp_data and 
       converts it to a numpy array"""
    if not sig:
        return np.array(list(sp_dict.values()))
    
    else:
        return np.array([signature(val) for val in sp_dict.values()])
                      

def create_h5_file(h5name, datasets, dataset_names):
    if not os.path.exists("hdf5"):
        os.mkdir("hdf5")
                        
    h5f = h5py.File(f'hdf5/{h5name}', 'w') 
    # for name, data in zip(dataset_names, datasets):
    #   h5f.create_dataset(name, data=data, compression="gzip")
    h5f.create_dataset(dataset_names[0], data=datasets[0], compression="gzip")
    h5f.create_dataset(dataset_names[1], data=datasets[1], compression="gzip")
    h5f.create_dataset(dataset_names[2], data=datasets[2], compression="gzip")
    
    h5f.close()
    

def process_as_h5(file, num_examples=None):
    """Takes in the dataset, performs signature and then stores data
        Choose number of of examples. It i suseful to choose a small number
        initially for testing"""
                    
    data = get_data(file)
    eeg_arr = []
    sp_arr = []
    target_arr = []
    if num_examples is None:
        num_examples = len(data)
                    
    for i in tqdm(range(num_examples)):
        exp_row = data.iloc[i]
        eeg_data, sp_dict, targets = get_eeg_sp_data(exp_row)
        eeg_arr.append(signature(eeg_data.to_numpy()))
        sp_arr.append(change_sp_to_array(sp_dict))
        target_arr.append(np.asfarray(targets))
        print(f"target {i} length: {len(targets)}")
        print(f"target {i} length: {type(targets[0])}")
    
    h5name = f"processed_dataset_{num_examples}.h5"
    eeg_arr = np.array(eeg_arr)
    sp_arr = np.array(sp_arr)
    target_arr = np.array(target_arr)
    
    
    
    ds_names = [f"eeg_train_data_{num_examples}", f"sp_train_data_{num_examples}", f"target_train_data_{num_examples}"]
    
    create_h5_file(h5name, [eeg_arr, sp_arr, target_arr], ds_names)
    print("Files created!")





if __name__ == "__main__":
    
    #for dirname, _, filenames in os.walk('/kaggle/input'):
    #    for filename in filenames:
    #        print(os.path.join(dirname, filename))

    train_path = '/kaggle/input/hms-harmful-brain-activity-classification/train.csv'
    process_as_h5(train_path, num_examples=10)
        
        
        
        