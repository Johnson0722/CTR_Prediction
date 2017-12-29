import pandas as pd
import pickle
from collections import Counter

train = pd.read_csv('/home/johnso/PycharmProjects/News_recommendation/CTR_prediction/avazu_CTR/train.csv',chunksize=20000)
test = pd.read_csv('/home/johnso/PycharmProjects/News_recommendation/CTR_prediction/avazu_CTR/test.csv',chunksize=20000)

C14 = dict()
C17 = dict()
C19 = dict()
C21 = dict()
site_id = dict()
site_domain = dict()
app_id = dict()
app_domain = dict()
device_model = dict()
device_id = dict()
device_ip = dict()

count = 0
for data in train:
    C14_list = data['C14'].values
    for k,v in Counter(C14_list).items():
        if k in C14.keys():
            C14[k] += v
        else:
            C14[k] = v

    C17_list = data['C17'].values
    for k,v in Counter(C17_list).items():
        if k in C17.keys():
            C17[k] += v
        else:
            C17[k] = v

    C19_list = data['C19'].values
    for k,v in Counter(C19_list).items():
        if k in C19.keys():
            C19[k] += v
        else:
            C19[k] = v

    C21_list = data['C21'].values
    for k,v in Counter(C21_list).items():
        if k in C21.keys():
            C21[k] += v
        else:
            C21[k] = v

    site_id_list = data['site_id'].values
    for k,v in Counter(site_id_list).items():
        if k in site_id.keys():
            site_id[k] += v
        else:
            site_id[k] = v

    site_domain_list = data['site_domain'].values
    for k,v in Counter(site_domain_list).items():
        if k in site_domain.keys():
            site_domain[k] += v
        else:
            site_domain[k] = v

    app_id_list = data['app_id'].values
    for k,v in Counter(app_id_list).items():
        if k in app_id.keys():
            app_id[k] += v
        else:
            app_id[k] = v

    app_domain_list = data['app_domain'].values
    for k,v in Counter(app_domain_list).items():
        if k in app_domain.keys():
            app_domain[k] += v
        else:
            app_domain[k] = v

    device_model_list = data['device_model'].values
    for k,v in Counter(device_model_list).items():
        if k in device_model.keys():
            device_model[k] += v
        else:
            device_model[k] = v

    device_id_list = data['device_id'].values
    for k,v in Counter(device_id_list).items():
        if k in device_id.keys():
            device_id[k] += v
        else:
            device_id[k] = v

    device_ip_list = data['device_ip'].values
    for k,v in Counter(device_ip_list).items():
        if k in device_ip.keys():
            device_ip[k] += v
        else:
            device_ip[k] = v

    count += 1
    if count % 100 == 0:
        print('{} has finished'.format(count))


for data in test:
    C14_list = data['C14'].values
    for k,v in Counter(C14_list).items():
        if k in C14.keys():
            C14[k] += v
        else:
            C14[k] = v

    C17_list = data['C17'].values
    for k,v in Counter(C17_list).items():
        if k in C17.keys():
            C17[k] += v
        else:
            C17[k] = v

    C19_list = data['C19'].values
    for k,v in Counter(C19_list).items():
        if k in C19.keys():
            C19[k] += v
        else:
            C19[k] = v

    C21_list = data['C21'].values
    for k,v in Counter(C21_list).items():
        if k in C21.keys():
            C21[k] += v
        else:
            C21[k] = v

    site_id_list = data['site_id'].values
    for k,v in Counter(site_id_list).items():
        if k in site_id.keys():
            site_id[k] += v
        else:
            site_id[k] = v

    site_domain_list = data['site_domain'].values
    for k,v in Counter(site_domain_list).items():
        if k in site_domain.keys():
            site_domain[k] += v
        else:
            site_domain[k] = v

    app_id_list = data['app_id'].values
    for k,v in Counter(app_id_list).items():
        if k in app_id.keys():
            app_id[k] += v
        else:
            app_id[k] = v

    app_domain_list = data['app_domain'].values
    for k,v in Counter(app_domain_list).items():
        if k in app_domain.keys():
            app_domain[k] += v
        else:
            app_domain[k] = v

    device_model_list = data['device_model'].values
    for k,v in Counter(device_model_list).items():
        if k in device_model.keys():
            device_model[k] += v
        else:
            device_model[k] = v

    device_id_list = data['device_id'].values
    for k,v in Counter(device_id_list).items():
        if k in device_id.keys():
            device_id[k] += v
        else:
            device_id[k] = v

    device_ip_list = data['device_ip'].values
    for k,v in Counter(device_ip_list).items():
        if k in device_ip.keys():
            device_ip[k] += v
        else:
            device_ip[k] = v


# save dictionaries
with open('field2count/C14.pkl','wb') as f:
    pickle.dump(C14,f)

with open('field2count/C17.pkl','wb') as f:
    pickle.dump(C17,f)

with open('field2count/C19.pkl','wb') as f:
    pickle.dump(C19,f)

with open('field2count/C21.pkl','wb') as f:
    pickle.dump(C21,f)

with open('field2count/site_id.pkl','wb') as f:
    pickle.dump(site_id,f)

with open('field2count/site_domain.pkl','wb') as f:
    pickle.dump(site_domain,f)

with open('field2count/app_id.pkl','wb') as f:
    pickle.dump(app_id,f)

with open('field2count/app_domain.pkl','wb') as f:
    pickle.dump(app_domain,f)

with open('field2count/device_model.pkl','wb') as f:
    pickle.dump(device_model,f)

with open('field2count/device_id.pkl','wb') as f:
    pickle.dump(device_id,f)

with open('field2count/device_ip.pkl','wb') as f:
    pickle.dump(device_ip,f)