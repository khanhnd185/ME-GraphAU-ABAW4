import numpy as np
import os

read_list_name = '../../../Data/ABAW4/validation_set_annotations.txt'
save_list_name = 'validation_au_relation_annotations.txt'

with open(read_list_name, 'r') as file:
    lines = file.readlines()
    lines = lines[1:]
    lines = [l.strip() for l in lines]
    lines = [l.split(',') for l in lines]
aus = np.array([np.array([float(x) for x in l[4:]]) for l in lines])

le, class_num = aus.shape
new_aus = np.zeros((le, class_num * class_num))
for j in range(class_num):
    for k in range(class_num):
        new_aus[:,j*class_num+k] = 2 * aus[:,j] + aus[:,k]
np.savetxt(os.path.join(save_list_name), new_aus, fmt='%d')
