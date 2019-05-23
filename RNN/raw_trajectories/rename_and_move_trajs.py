import os, shutil
import numpy as np


for _, dirs, _ in os.walk('.', topdown = True):
    for d_name in dirs:
        if d_name.split(' ')[0] == 'Subject':
            for _, _, files in os.walk(d_name):
                files_temp = [f_name.split('_')[0] for f_name in files]
                grasps = list(set(files_temp))
                for grasp in grasps:
                    trial_cnt = 1
                    grasp_trials = [f_name for f_name in files if grasp == f_name.split('_')[0]]
                    for trial in grasp_trials:
                        new_name = str(int(d_name.split(' ')[1])) + '_' + str(int(trial.split(' ')[0])) + '_' + str(trial_cnt)
                        shutil.copy(d_name + '/' + trial, 'all_trajs/' + new_name + '.npy')
                        trial_cnt += 1
                    # input('')
                # for f_name in files:
                #     new_name = d_name.split(' ')[1] + '_' + f_name.split(' ')[0]
                #     shutil.copy(d_name + '/' + f_name, 'all_trajs/' + new_name + '.npy')
            # input('')
