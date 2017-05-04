import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

filepath = '../../data/learning_curves/ngsim_5_sec.npz'
f = np.load(filepath)
sizes, train_scores, val_scores = f['sizes'], f['train_scores'], f['val_scores']
loss = 'ce'
idx = 0 if loss == 'ce' else 1
plt.plot(sizes, train_scores[:, idx], c='blue', label='train')
plt.plot(sizes, val_scores[:, idx], c='red', label='val')
plt.legend()
plt.xlabel('training set sizes')
plt.ylabel('{} loss'.format(loss))
plt.show()