# coding:utf-8
import pandas as pd
import pickle
import logging
from scipy.sparse import coo_matrix
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)


def one_hot_representation(sample, fields_dict, isample):
    """
    One hot presentation for every sample data
    :param fields_dict: fields value to array index
    :param sample: sample data, type of pd.series
    :param isample: sample index
    :return: sample index
    """
    index = []
    for field in fields_dict:
        # get index of array
        if field == 'hour':
            field_value = int(str(sample[field])[-2:])
        else:
            field_value = sample[field]
        ind = fields_dict[field][field_value]
        index.append([isample,ind])
    return index


def train_sparse_data_generate(train_data, fields_dict):
    sparse_data = []
    # batch_index
    ibatch = 0
    for data in train_data:
        labels = []
        indexes = []
        for i in range(len(data)):
            sample = data.iloc[i,:]
            click = sample['click']
            # get labels
            if click == 0:
                label = 0
            else:
                label = 1
            labels.append(label)
            # get indexes
            index = one_hot_representation(sample,fields_dict, i)
            indexes.extend(index)
        sparse_data.append({'indexes':indexes, 'labels':labels})
        ibatch += 1
        if ibatch % 200 == 0:
            logging.info('{}-th batch has finished'.format(ibatch))
    with open('../avazu_CTR/train_sparse_data_frac_0.01.pkl','wb') as f:
        pickle.dump(sparse_data, f)

def test_sparse_data_generate(test_data, fields_dict):
    sparse_data = []
    # batch_index
    ibatch = 0
    for data in test_data:
        ids = []
        indexes = []
        for i in range(len(data)):
            sample = data.iloc[i,:]
            ids.append(sample['id'])
            index = one_hot_representation(sample,fields_dict, i)
            indexes.extend(index)
        sparse_data.append({'indexes':indexes, 'id':ids})
        ibatch += 1
        if ibatch % 200 == 0:
            logging.info('{}-th batch has finished'.format(ibatch))
    with open('../avazu_CTR/test_sparse_data_frac_0.01.pkl','wb') as f:
        pickle.dump(sparse_data, f)


# generate batch indexes
if __name__ == '__main__':

    fields = ['hour', 'C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21',
              'banner_pos', 'site_id' ,'site_domain', 'site_category', 'app_domain',
              'app_id', 'app_category', 'device_model', 'device_type', 'device_id',
              'device_conn_type']
    batch_size = 512
    train = pd.read_csv('../avazu_CTR/train_frac_0.01.csv', chunksize=batch_size)
    test = pd.read_csv('../avazu_CTR/test.csv', chunksize=batch_size)
    # loading dicts
    fields_dict = {}
    for field in fields:
        with open('dicts/'+field+'.pkl','rb') as f:
            fields_dict[field] = pickle.load(f)

    test_sparse_data_generate(test, fields_dict)











