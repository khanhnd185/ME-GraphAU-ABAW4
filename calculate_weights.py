import numpy as np

with open('../../../Data/ABAW4/validation_set_annotations.txt', 'r') as file:
    lines = file.readlines()
    lines = lines[1:]
    lines = [l.strip() for l in lines]
    lines = [l.split(',') for l in lines]
AUs = [np.array([float(x) for x in l[4:]]) for l in lines]
expr = [int(l[3]) for l in lines]

ids_list = [i for i, x in enumerate(AUs) if -1 not in x]
new_list = np.array([AUs[i] for i in ids_list])
weight = 1.0 / np.sum(new_list, axis=0)
weight = weight / weight.sum() * new_list.shape[1]
weight = weight.T
np.savetxt('./valid_weight_au.txt', weight, fmt='%f', delimiter='\t')

ids_list = [i for i, x in enumerate(expr) if x != -1]
weight = np.zeros(8)
for i in range(8):
    weight[i] = len([x for x in expr if x==i])
weight = 1.0 / weight
weight = weight / weight.sum() * 8
weight = weight.T
np.savetxt('./valid_weight_ex.txt', weight, fmt='%f', delimiter='\t')


with open('../../../Data/ABAW4/training_set_annotations.txt', 'r') as file:
    lines = file.readlines()
    lines = lines[1:]
    lines = [l.strip() for l in lines]
    lines = [l.split(',') for l in lines]
AUs = [np.array([float(x) for x in l[4:]]) for l in lines]
expr = [int(l[3]) for l in lines]

ids_list = [i for i, x in enumerate(AUs) if -1 not in x]
new_list = np.array([AUs[i] for i in ids_list])
weight = 1.0 / np.sum(new_list, axis=0)
weight = weight / weight.sum() * new_list.shape[1]
weight = weight.T
np.savetxt('./train_weight_au.txt', weight, fmt='%f', delimiter='\t')

ids_list = [i for i, x in enumerate(expr) if x != -1]
weight = np.zeros(8)
for i in range(8):
    weight[i] = len([x for x in expr if x==i])
weight = 1.0 / weight
weight = weight / weight.sum() * 8
weight = weight.T
np.savetxt('./train_weight_ex.txt', weight, fmt='%f', delimiter='\t')

