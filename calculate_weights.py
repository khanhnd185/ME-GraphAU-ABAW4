import numpy as np

with open('../../../Data/ABAW4/validation_set_annotations.txt', 'r') as file:
    lines = file.readlines()
    lines = lines[1:]
    lines = [l.strip() for l in lines]
    lines = [l.split(',') for l in lines]
AUs = [np.array([float(x) for x in l[4:]]) for l in lines]
expr = [int(l[3]) for l in lines]
valence = [float(l[1]) for l in lines]

ids_list_au = [i for i, x in enumerate(AUs) if -1 not in x]
new_list = np.array([AUs[i] for i in ids_list_au])
weight = 1.0 / np.sum(new_list, axis=0)
weight = weight / weight.sum() * new_list.shape[1]
weight = weight.T
np.savetxt('./valid_weight_au.txt', weight, fmt='%f', delimiter='\t')

ids_list_ex = [i for i, x in enumerate(expr) if x != -1]
weight = np.zeros(8)
for i in range(8):
    weight[i] = len([x for x in expr if x==i])
weight = 1.0 / weight
weight = weight / weight.sum() * 8
weight = weight.T
np.savetxt('./valid_weight_ex.txt', weight, fmt='%f', delimiter='\t')

ids_list_va = [i for i, x in enumerate(valence) if x != -5]
print('va: ' + str(len(ids_list_va)))
print('ex: ' + str(len(ids_list_ex)))
print('au: ' + str(len(ids_list_au)))
len_va = float(len(ids_list_va))
len_ex = float(len(ids_list_ex))
len_au = float(len(ids_list_au))
weight_va = 1. / len_va
weight_ex = 1. / len_ex
weight_au = 1. / len_au
_avg = (weight_va + weight_ex + weight_au) / 3
weight_va = weight_va / _avg
weight_ex = weight_ex / _avg
weight_au = weight_au / _avg
weight_all = np.array([weight_va, weight_ex, weight_au])
np.savetxt('./valid_weight.txt', weight_all, fmt='%f', delimiter='\t')

with open('../../../Data/ABAW4/training_set_annotations.txt', 'r') as file:
    lines = file.readlines()
    lines = lines[1:]
    lines = [l.strip() for l in lines]
    lines = [l.split(',') for l in lines]
AUs = [np.array([float(x) for x in l[4:]]) for l in lines]
expr = [int(l[3]) for l in lines]
valence = [float(l[1]) for l in lines]

ids_list_au = [i for i, x in enumerate(AUs) if -1 not in x]
new_list = np.array([AUs[i] for i in ids_list_au])
weight = 1.0 / np.sum(new_list, axis=0)
weight = weight / weight.sum() * new_list.shape[1]
weight = weight.T
np.savetxt('./train_weight_au.txt', weight, fmt='%f', delimiter='\t')

ids_list_ex = [i for i, x in enumerate(expr) if x != -1]
weight = np.zeros(8)
for i in range(8):
    weight[i] = len([x for x in expr if x==i])
weight = 1.0 / weight
weight = weight / weight.sum() * 8
weight = weight.T
np.savetxt('./train_weight_ex.txt', weight, fmt='%f', delimiter='\t')

ids_list_va = [i for i, x in enumerate(valence) if x != -5]
print('va: ' + str(len(ids_list_va)))
print('ex: ' + str(len(ids_list_ex)))
print('au: ' + str(len(ids_list_au)))
len_va = float(len(ids_list_va))
len_ex = float(len(ids_list_ex))
len_au = float(len(ids_list_au))
weight_va = 1. / len_va
weight_ex = 1. / len_ex
weight_au = 1. / len_au
_avg = (weight_va + weight_ex + weight_au) / 3
weight_va = weight_va / _avg
weight_ex = weight_ex / _avg
weight_au = weight_au / _avg
weight_all = np.array([weight_va, weight_ex, weight_au])
np.savetxt('./train_weight.txt', weight_all, fmt='%f', delimiter='\t')
