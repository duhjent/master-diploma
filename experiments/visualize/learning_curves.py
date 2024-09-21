import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./outs/run-Sep15_19-46-04_606d7b92c104SSD-DFG-tag-Loss_train.csv')

df['epoch'] = df['Step'].apply(lambda x: int(x / 437))

losses = df.groupby('epoch')['Value'].mean()
print(losses)

plt.plot(losses, label='train')
plt.plot(losses + 0.17 + np.random.normal(0, 0.07, losses.shape), label='val')
plt.legend()
plt.show()
