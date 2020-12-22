from numpy import genfromtxt, savetxt
import numpy as np
import pandas as pd

labels = genfromtxt('res.csv', delimiter=',')
labels = labels.astype(int)
labels_df = pd.DataFrame(data=labels)
labels_df.index += 1
labels_df.to_csv('submission3.csv')
