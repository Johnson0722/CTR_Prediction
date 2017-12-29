# coding:utf-8
import numpy as np
import pandas as pd
import pickle


def one_hot_representation(sample, fields_dict, array_length):
    """
    One hot presentation for every sample data
    :param fields_dict: fields value to array index
    :param sample: sample data, type of pd.series
    :param array_length: length of one-hot representation
    :return: one-hot representation, type of np.array
    """
    array = np.zeros([array_length])
    idx = []
    for field in fields_dict:
        # get index of array
        if field == 'hour':
            field_value = int(str(sample[field])[-2:])
        else:
            field_value = sample[field]
        ind = fields_dict[field][field_value]
        array[ind] = 1
        idx.append(ind)
    return array,idx[:21]



if __name__ == '__main__':
    fields_train = ['hour', 'C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21',
              'banner_pos', 'site_id' ,'site_domain', 'site_category', 'app_domain',
              'app_id', 'app_category', 'device_model', 'device_type', 'device_id',
              'device_conn_type','click']

    fields_test = ['hour', 'C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21',
                   'banner_pos', 'site_id' ,'site_domain', 'site_category', 'app_domain',
                   'app_id', 'device_id', 'app_category', 'device_model', 'device_type',
                   'device_conn_type']

    train = pd.read_csv('/home/johnso/PycharmProjects/News_recommendation/CTR_prediction/avazu_CTR/train.csv',
                        chunksize=100)
    test = pd.read_csv('/home/johnso/PycharmProjects/News_recommendation/CTR_prediction/avazu_CTR/test.csv',
                        chunksize=100)
    # loading dicts
    fields_train_dict = {}
    for field in fields_train:
        with open('dicts/'+field+'.pkl','rb') as f:
            fields_train_dict[field] = pickle.load(f)

    fields_test_dict = {}
    for field in fields_test:
        with open('dicts/'+field+'.pkl','rb') as f:
            fields_test_dict[field] = pickle.load(f)


    train_array_length = max(fields_train_dict['click'].values()) + 1
    test_array_length = train_array_length - 2
    # initialize the model

    for data in test:
        # data['click'] = np.zeros(100)
        # data.to_csv('a.csv',mode='a')
        sample = data.iloc[3,:]
        print(one_hot_representation(sample, fields_test_dict, test_array_length))

        break





