import numpy as np
from scipy import signal
import os

trajs = []
q = 6
sample_size = 20
for root0, files0, names0 in os.walk('../../../HUSTdataset/Subjects', topdown = True):
    cnt0 = 0
    for file0 in files0:
        if (os.path.exists(os.path.join('../../../HUSTdataset/Subjects', file0))):
            # print(file0)
            f_temp = file0.split(' ')[0] + ' ' + file0.split(' ')[1] # jsut formatting the folder names correctly here bc im too lazy to go back and do it for all of them in the actual location
            if not os.path.exists('raw/'+f_temp):
                os.makedirs('raw/'+f_temp)
            if not os.path.exists('processed/scaled'):
                os.makedirs('processed/scaled')
            if not os.path.exists('processed/unscaled'):
                os.makedirs('processed/unscaled')
            if not os.path.exists('processed/unscaled/'+f_temp):
                os.makedirs('processed/unscaled/'+f_temp)
            if not os.path.exists('processed/scaled/'+f_temp):
                os.makedirs('processed/scaled/'+f_temp)    #only needed for the first run to make the directories
            subject_path = '.././../HUSTdataset/Subjects' + '/' + file0
            PC = np.load('../PCAs/'+f_temp.split(' ')[0]+'_'+f_temp.split(' ')[1]+'_PCA.npy')
            for root, files, names in os.walk(subject_path, topdown = True):
                for file in files:
                    if (os.path.exists(os.path.join(subject_path, file))):
                        for root1, files1, names1 in os.walk(subject_path + '/' + file, topdown = True):
                            grasp_trajs = []
                            for file1 in files1:
                                for name1 in range(1, 4):
                                    # print(file0, file, file1, name1)
                                    r = np.genfromtxt(subject_path + '/' + file + '/' + file1 + '/' + str(name1) + '.txt', delimiter='\t')
                                    r = r[:, 1:17]
                                    np.save('raw/'+f_temp+'/'+file+'_'+file1+'_'+str(name1)+'_raw', r)

                                    r_temp = np.copy(r)
                                    r_temp = np.matmul(r_temp, PC[:, :q])
                                    np.save('processed/unscaled/'+f_temp+'/'+file+'_'+file1+'_'+str(name1)+'_processed_unscaled', r_temp)
                                    new_traj = np.zeros((sample_size, 1));
                                    for row in r_temp.T:
                                        new_traj = np.append(new_traj, signal.resample(row, sample_size).reshape((1, sample_size)).T, axis = 1)
                                    new_traj = np.delete(new_traj, 0, 1)
                                    r_temp = new_traj
                                    np.save('processed/scaled/'+f_temp+'/'+file+'_'+file1+'_'+str(name1)+'_processed_scaled', r_temp)
