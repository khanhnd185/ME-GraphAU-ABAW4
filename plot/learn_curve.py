# importing package
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

arc = 'swinbase'
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
    plt.plot(i, df[k], label=k)
plt.legend()
plt.show()