# importing package
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Plot learning curve')
parser.add_argument('--arc', '-a', type=str, help='choose architecture')
parser.add_argument('--factor', default='all', type=str, help="choose factor to plot")
args = parser.parse_args()

arc = args.arc
print(args.factor)
df1 = pd.read_csv(arc + '_1.csv')
df2 = pd.read_csv(arc + '_2.csv')
i = np.arange(20)

df = pd.concat(
    [df1, df2],
    axis=0,
    join="outer",
    ignore_index=True,
    keys=df1.keys(),
    copy=True,
)

print(df)  
for k in df.keys()[1:]:
    if (args.factor not in k) and (args.factor != 'all'):
        print('skip')
        continue
    else:
        plt.plot(i, df[k], label=k)

plt.legend()
plt.show()