# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-16T12:40:53.885066Z","iopub.execute_input":"2024-03-16T12:40:53.885483Z","iopub.status.idle":"2024-03-16T12:40:54.095864Z","shell.execute_reply.started":"2024-03-16T12:40:53.885452Z","shell.execute_reply":"2024-03-16T12:40:54.094753Z"}}
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import subprocess
import pkg_resources
import os
import numpy as np
import time
import h5py
from tqdm import tqdm


#installs iisignature
package_name = 'iisignature'

try:
    pkg_resources.get_distribution(package_name)
    print(f"{package_name} is already installed.")
except pkg_resources.DistributionNotFound:
    # The package is not installed; install it.
    print(f"{package_name} not found. Installing...")
    subprocess.run(['pip', 'install', package_name], check=True)

import iisignature as isig

# %% [code] {"execution":{"iopub.status.busy":"2024-03-16T12:40:55.738271Z","iopub.execute_input":"2024-03-16T12:40:55.738682Z","iopub.status.idle":"2024-03-16T12:40:55.750749Z","shell.execute_reply.started":"2024-03-16T12:40:55.738654Z","shell.execute_reply":"2024-03-16T12:40:55.749635Z"},"jupyter":{"outputs_hidden":false}}
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
    'RP' : [("Fp2","F4"),("F4","C4"),("C4","P4"),("P4","O2")],
    'other' : [("Fz","Cz"), ("Cz", "Pz"), ("EKG")]
}
SP_WIN = 600 # 10 minutes = 600 seconds
EGG_WIN = 50 # 50 seconds in total

LABELED_SECS = 10

# %% [code] {"execution":{"iopub.status.busy":"2024-03-16T12:40:55.914307Z","iopub.execute_input":"2024-03-16T12:40:55.914949Z","iopub.status.idle":"2024-03-16T12:40:55.919195Z","shell.execute_reply.started":"2024-03-16T12:40:55.914918Z","shell.execute_reply":"2024-03-16T12:40:55.918089Z"},"jupyter":{"outputs_hidden":false}}
def get_data(file):
    """Takes in file and returns a pandas data frame of the data
        File should be a .csv """
    
    return pd.read_csv(file)

# %% [code] {"execution":{"iopub.status.busy":"2024-03-16T12:40:56.025504Z","iopub.execute_input":"2024-03-16T12:40:56.026893Z","iopub.status.idle":"2024-03-16T12:40:56.041648Z","shell.execute_reply.started":"2024-03-16T12:40:56.026841Z","shell.execute_reply":"2024-03-16T12:40:56.040418Z"},"jupyter":{"outputs_hidden":false}}
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
            if len(signals) == 2:
                diff=eeg_data[signals[0]]-eeg_data[signals[1]] # Subtracts relevant fields as in the image above
                diff.ffill(inplace = True) # forward fills in the casse of nan values
                eeg[f"{chain}: {signals[0]} - {signals[1]}"] = diff
            
            elif len(signals) == 1:
                sig=eeg_data[signals[0]]
                sig.ffill(inplace = True) 
                eeg[f"{chain}: {signals[0]}"] = sig
                
                
    
    return eeg, sp, train_row[TARGET_COLUMNS].values

# %% [code] {"jupyter":{"outputs_hidden":false}}


# %% [code] {"execution":{"iopub.status.busy":"2024-03-16T12:40:56.372282Z","iopub.execute_input":"2024-03-16T12:40:56.373478Z","iopub.status.idle":"2024-03-16T12:40:56.378346Z","shell.execute_reply.started":"2024-03-16T12:40:56.373441Z","shell.execute_reply":"2024-03-16T12:40:56.376921Z"},"jupyter":{"outputs_hidden":false}}
def signature(data, level=2, chunk_len=None):
    """Performs signature on data needs data to be 3d,
    
    if input to isig.sig is (num_examples, num_time_steps, dim_of_data_at_each_time_step)
    then the output will be (num_examples, isig.sig_length(dim_of_data_at_each_time_step, level))
    
    chunk_len will split the timesteps (i.e. the second dimension) into chunks of length chunk_len
    
    
    """
    assert len(data.shape) == 3, f"data needs to be 3d. data is {data.shape} If it is 2d, reshape so first dim is 1"
    assert chunk_len < data.shape[1] f"chunk length is bigger than the number of time steps"
    
    sig_len = isig.siglength(data.shape[2], level)
    assert sig_len > 0 , "Too many elements in each chunk. Signature package thinks the num of elements is negative lol"
    
    if chunk_len is None:        
        return isig.sig(data, level) # np.array(sig_arr)
        
    else:
        num_whole_chunks = (data.shape[1] // chunk_len)
        remainder = data.shape[1] % chunk_len
            
        #sig_arr = np.zeros((data.shape[0], num_whole_chunks + int(remainder != 0), sig_len)
        sig_arr = []
        
        for j in range(num_whole_chunks):
            sig_output = isig.sig(data[:,j*chunk_len: (j+1)*chunk_len,:], level)
            sig_arr.append(sig_output.reshape(sig_output.shape[0], 1, sig_output.shape[1]))

        sig_output = isig.sig(data[:,num_whole_chunks*chunk_len:,:], level)
        sig_arr.append(sig_output.reshape(sig_output.shape[0], 1, sig_output.shape[1]))
            
        
        return np.concatenate(sig_arr, axis=1)
        

# %% [code] {"execution":{"iopub.status.busy":"2024-03-16T12:40:56.493541Z","iopub.execute_input":"2024-03-16T12:40:56.493964Z","iopub.status.idle":"2024-03-16T12:40:56.500716Z","shell.execute_reply.started":"2024-03-16T12:40:56.493933Z","shell.execute_reply":"2024-03-16T12:40:56.499369Z"},"jupyter":{"outputs_hidden":false}}
def change_sp_to_array(sp_dict, sig=True):
    """Takes in a dictionary of sp_data and 
       converts it to a numpy array"""    
    if not sig:
        return np.array(list(sp_dict.values()))
    
    else:
        return np.array([signature(val) for val in sp_dict.values()])

# %% [code] {"execution":{"iopub.status.busy":"2024-03-16T12:40:56.775940Z","iopub.execute_input":"2024-03-16T12:40:56.776949Z","iopub.status.idle":"2024-03-16T12:40:56.782437Z","shell.execute_reply.started":"2024-03-16T12:40:56.776911Z","shell.execute_reply":"2024-03-16T12:40:56.781586Z"},"jupyter":{"outputs_hidden":false}}
def create_h5_file(h5name, datasets, dataset_names):
    if not os.path.exists("hdf5"):
        os.mkdir("hdf5")
                        
    h5f = h5py.File(f'hdf5/{h5name}', 'w') 
    for name, data in zip(dataset_names, datasets):
        h5f.create_dataset(name, data=data, compression="gzip")
    h5f.close()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-16T12:40:56.784340Z","iopub.execute_input":"2024-03-16T12:40:56.784687Z","iopub.status.idle":"2024-03-16T12:40:56.797546Z","shell.execute_reply.started":"2024-03-16T12:40:56.784652Z","shell.execute_reply":"2024-03-16T12:40:56.796355Z"},"jupyter":{"outputs_hidden":false}}
def process_as_h5(file, num_examples=None, apply_sig=False):
    """Takes in the dataset, performs signature and then stores data
        Choose number of of examples. It i suseful to choose a small number
        initially for testing"""
                    
    data = get_data(file)
    eeg_arr = []
    sp_arr = []
    target_arr = []
    num_votes_arr = []
    if num_examples is None:
        num_examples = len(data)
                    
    for i in tqdm(range(num_examples)):
        exp_row = data.iloc[i]
        eeg_data, sp_dict, targets = get_eeg_sp_data(exp_row)
#         print(eeg_data.shape)
#         print(signature(eeg_data.to_numpy()).shape)
        if apply_sig:
            eeg_arr.append(signature(eeg_data.to_numpy()))
            sp_arr.append(change_sp_to_array(sp_dict, sig=True))
        else:
            eeg_arr.append(eeg_data.to_numpy())
            sp_arr.append(change_sp_to_array(sp_dict, sig=False))
        total_votes = targets.sum()
        num_votes_arr.append(total_votes)
        target_arr.append(np.asfarray(targets) / total_votes)
    
    h5name = f"processed_dataset_{num_examples}.h5"
    eeg_arr = np.array(eeg_arr)
    sp_arr = np.array(sp_arr)
    target_arr = np.array(target_arr)
    num_votes_arr = np.array(num_votes_arr)
    
    ds_names = [f"eeg", f"sp", 
                f"targets", f"num_votes"]
    
    create_h5_file(h5name, [eeg_arr, sp_arr, target_arr, num_votes_arr], ds_names)
    print("Files created!")

def get_data_from_h5(filename):
    with h5py.File(filename, 'r') as file:
        # List all groups

        print("Keys: %s" % file.keys())

        file_keys = list(file.keys())

        a_group_key = list(file.keys())[0]

        eeg_data = np.array(file[f"eeg"])
        sp_data = np.array(file[f"sp"])
        targets = np.array(file[f"targets"])
        num_votes = np.array(file[f"num_votes"])
        num_votes = num_votes.reshape((len(num_votes), -1))
        return eeg_data, sp_data, targets, num_votes

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-16T12:40:56.814038Z","iopub.execute_input":"2024-03-16T12:40:56.814457Z","iopub.status.idle":"2024-03-16T12:40:58.074884Z","shell.execute_reply.started":"2024-03-16T12:40:56.814426Z","shell.execute_reply":"2024-03-16T12:40:58.073584Z"}}
if __name__ == "__main__":
#     train_path = '/kaggle/input/hms-harmful-brain-activity-classification/train.csv'
#     data_file = get_data(train_path)
#     process_as_h5(train_path, num_examples=10)
    
#     h5_file = '/kaggle/working/hdf5/processed_dataset_10.h5'
#     eeg_data, sp_data, targets, num_votes = get_data_from_h5(h5_file)
#     eeg_data = np.transpose(eeg_data,(0,2,1))
    
#     sig_arr = []
#     for i in range(eeg_data.shape[0]):
#         sig_arr.append(signature(eeg_data[i,:,:], level=2, chunk_len=100))
    
#     print(sig_arr)

    rand_arr = np.random.rand(4, 5, 6)
    print(signature(rand_arr, chunk_len).shape)
        
    
    
    
    

# %% [code] {"jupyter":{"outputs_hidden":false}}


# %% [code] {"jupyter":{"outputs_hidden":false}}
