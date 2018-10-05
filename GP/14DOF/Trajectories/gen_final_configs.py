import numpy as np
import os

raw_path = 'raw/'
pca_path = 'processed/unscaled/'
gen_path = 'final_configs/'

for subject in os.listdir(raw_path):

    config_data = []

    for trial in os.listdir(raw_path+subject):
        trial_data = np.load(raw_path+subject+'/'+trial)
        config = trial_data[-1, :]
        config = np.insert(config, 0, int(trial[:2]))
        config_data.append(config)

    config_data = np.array(config_data)
    np.save(gen_path+subject+'_final_configs_raw', config_data)

for subject in os.listdir(pca_path):

    config_data = []

    for trial in os.listdir(pca_path+subject):
        trial_data = np.load(pca_path+subject+'/'+trial)
        config = trial_data[-1, :]
        config = np.insert(config, 0, int(trial[:2]))
        config_data.append(config)

    config_data = np.array(config_data)
    np.save(gen_path+subject+'_final_configs_pca', config_data)
