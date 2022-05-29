import numpy as np

with open('../../../Data/ABAW4/training_set_annotations.txt', 'r') as file:
    lines = file.readlines()
    lines = lines[1:]
    lines = [l.strip() for l in lines]
    lines = [l.split(',') for l in lines]
AUs = [np.array([float(x) for x in l[4:]]) for l in lines]
ids_list = [i for i, x in enumerate(AUs) if -1 not in x]
new_list = np.array([AUs[i] for i in ids_list])
weight = 1.0 / np.sum(new_list, axis=0)
weight = weight / weight.sum() * new_list.shape[1]
weight = weight.T

np.savetxt('./train_weight.txt', weight, fmt='%f', delimiter='\t')
