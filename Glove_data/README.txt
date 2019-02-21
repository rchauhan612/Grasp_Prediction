To convert the raw data from the glove found in /PERSON/results using
/PERSON/read_angles.py:

1. Go from 5 DOF to 16: /PERSON/expand_angles.py
2. PCA conversion: /PERSON/gen_glove_pca.py
3. Regenerate trajectories with PC: /PERSON/PCAs/Trajectories/gen_traj_pca.py
    -use the argument 'gen' to show them
    -use the argument 'plot' to show them
4. Create GP from the trajectories: /PERSON/GPs/gen_glove_gp.py
    -use the argument 'plot' to show them
5. Run prediction test: /PERSON/GPs/glove_pred.py
6. Plot confusion matrix: /PERSON/GPs/gen_comf_comp.py

Run all of the above from the /PERSON directory.
