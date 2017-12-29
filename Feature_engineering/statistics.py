import pandas as pd
from collections import Counter
import pickle

# train_data = pd.read_csv('/home/johnso/PycharmProjects/News_recommendation/CTR_prediction/avazu_CTR/train.csv',
#                          chunksize=1000)
#
# device_ids = []
#
# num = 0
# # batch_size data
# for data in train_data:
#     device_ids.extend(list(data['device_id'].values))
#     num += 1
#     if num % 10000 == 0:
#         print('{} has finished'.format(num))
#
#
# device_ids_dict = Counter(device_ids)
#
# with open('device_id_count.pkl', 'wb') as f:
#     pickle.dump(device_ids_dict,f)


with open('device_id_count.pkl', 'rb') as f:
    device_id_count = pickle.load(f)
    device_id_frequency = set(device_id_count.values())
    print(device_id_frequency)
    print(len(device_id_frequency))