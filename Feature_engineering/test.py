# coding:utf-8
import pandas as pd
import numpy as np

ids = pd.read_csv('/home/johnso/PycharmProjects/News_recommendation/CTR_prediction/avazu_CTR/test.csv')['id'].values

click = pd.read_csv('/home/johnso/PycharmProjects/News_recommendation/CTR_prediction/FM_baseline/FM_FTRL_v1.csv')['click'].values

print(len(ids))
print(len(click))

pd.DataFrame(np.array([ids, click]).T, columns=['id','click']).to_csv('FM_FTRL_v1.csv', index=False)



