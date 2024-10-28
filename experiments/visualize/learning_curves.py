import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_train = pd.read_csv('./outs/vgg50epochs-loss_train.csv')
df_val = pd.read_csv('./outs/vgg50epochs-loss_val.csv')

df = df_train.join(df_val, 'Step', lsuffix='train', rsuffix='val')

plt.plot(df['Steptrain'], df['Valuetrain'], label='train')
plt.plot(df['Stepval'], df['Valueval'], label='val')

plt.xlabel('Epoch')
plt.ylabel('Cross-entropy loss')
plt.legend()

plt.savefig('./outs/vgg50epochs-curve.png')

plt.show()

df_train = pd.read_csv('./outs/fractalsmaller200epochs-loss_train.csv')
df_val = pd.read_csv('./outs/fractalsmaller200epochs-loss_val.csv')

df = df_train.join(df_val, 'Step', lsuffix='train', rsuffix='val')

plt.plot(df['Steptrain'], df['Valuetrain'], label='train')
plt.plot(df['Stepval'], df['Valueval'], label='val')

plt.xlabel('Epoch')
plt.ylabel('Cross-entropy loss')
plt.legend()

plt.savefig('./outs/fractalsmaller200epochs-curve.png')
