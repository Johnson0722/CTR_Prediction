import pandas as pd

train = pd.read_csv('train_baseline_raw.csv', chunksize=1000)
for data in train:
    data.to_csv('train_baseline_debug.csv', index=False)
    break