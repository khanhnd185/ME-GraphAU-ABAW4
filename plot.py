import numpy as np
import pandas as pd
from collections import Counter
from matplotlib import pyplot as plt
import seaborn as sns

expression = ['Neutral', 'Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Other']
with open('../../../Data/ABAW4/training_set_annotations.txt', 'r') as file:
    lines = file.readlines()
    lines = lines[1:]
    lines = [l.strip() for l in lines]
    lines = [l.split(',') for l in lines]
path = [l[0] for l in lines]
valence = [float(l[1]) for l in lines]
arousal = [float(l[2]) for l in lines]
expr = [int(l[3]) for l in lines]
AUs = [np.array([float(x) for x in l[4:]]) for l in lines]

total = len(path)
only_au = [i for i in range(total) if -1 not in AUs[i] and -1 == expr[i] and (-5.0 == valence[i] or -5.0 == arousal[i])]
only_ex = [i for i in range(total) if -1 in AUs[i] and -1 != expr[i] and (-5.0 == valence[i] or -5.0 == arousal[i])]
only_va = [i for i in range(total) if -1 in AUs[i] and -1 == expr[i] and (-5.0 != valence[i] and -5.0 != arousal[i])]
au_ex_va = [i for i in range(total) if -1 not in AUs[i] and -1 != expr[i] and (-5.0 != valence[i] and -5.0 != arousal[i])]
ex_au = [i for i in range(total) if -1 not in AUs[i] and -1 != expr[i] and (-5.0 == valence[i] or -5.0 == arousal[i])]
va_ex = [i for i in range(total) if -1 in AUs[i] and -1 != expr[i] and (-5.0 != valence[i] and -5.0 != arousal[i])]
va_au = [i for i in range(total) if -1 not in AUs[i] and -1 == expr[i] and (-5.0 != valence[i] and -5.0 != arousal[i])]
noone = [i for i in range(total) if -1 in AUs[i] and -1 == expr[i] and (-5.0 == valence[i] or -5.0 == arousal[i])]

all_au = [i for i in range(total) if -1 not in AUs[i]]
all_ex = [i for i in range(total) if -1 != expr[i]]
all_va = [i for i in range(total) if (-5.0 != valence[i] and -5.0 != arousal[i])]

print('Total examples: ' + str(total))
print('None: ' + str(len(noone)))
print('Only AU: ' + str(len(only_au)))
print('Only EX: ' + str(len(only_ex)))
print('Only VA: ' + str(len(only_va)))
print('EX AU: ' + str(len(ex_au)))
print('VA EX: ' + str(len(va_ex)))
print('VA AU ' + str(len(va_au)))
print('All: ' + str(len(au_ex_va)))
print('All AU: ' + str(len(all_au)))
print('All EX: ' + str(len(all_ex)))
print('All VA: ' + str(len(all_va)))

ex = [expression[expr[i]] for i in all_ex]
# fig, axes = plt.subplots(2,1)

v = [valence[i] for i in all_va]
a = [arousal[i] for i in all_va]

#uniform_data = np.random.rand(10, 12)
#plt.scatter(a, v)
#fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(9, 4))
plt.hist2d(a, v)
plt.colorbar()

# pd.DataFrame({'Expression' : ex}).groupby('Expression', as_index=True).size().plot(kind = 'bar', ax=ax1)

plt.show()

