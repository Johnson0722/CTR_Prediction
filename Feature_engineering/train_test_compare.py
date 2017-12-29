import pandas as pd
import pickle

fields = ['hour', 'C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21',
          'banner_pos', 'site_id','site_domain', 'site_category','app_id','app_domain',
          'app_category', 'device_model', 'device_type',
          'device_conn_type']

data = pd.read_csv('/home/johnso/PycharmProjects/News_recommendation/CTR_prediction/avazu_CTR/test.csv')

C1_v = set(data['C1'].values)

C14_v = set(data['C14'].values)

C15_v = set(data['C15'].values)

C16_v = set(data['C16'].values)

C17_v = set(data['C17'].values)

C18_v = set(data['C18'].values)

C19_v = set(data['C19'].values)

C20_v = set(data['C20'].values)

C21_v = set(data['C21'].values)

hour_v = set(data['hour'].values)

banner_pos_v = set(data['banner_pos'].values)

site_id_v = set(data['site_id'].values)

site_domain_v = set(data['site_domain'].values)

site_category_v = set(data['site_category'].values)

app_id_v = set(data['app_id'].values)

app_domain_v = set(data['app_domain'].values)

app_category_v = set(data['app_category'].values)

device_model_v = set(data['device_model'].values)

device_type_v = set(data['device_type'].values)

device_conn_type_v = set(data['device_conn_type'].values)


with open('sets/click.pkl','rb') as f:
    click = pickle.load(f)

with open('sets/hour.pkl','rb') as f:
    hour = pickle.load(f)

with open('sets/C1.pkl','rb') as f:
    C1 = pickle.load(f)

with open('sets/C14.pkl','rb') as f:
    C14 = pickle.load(f)

with open('sets/C15.pkl','rb') as f:
    C15 = pickle.load(f)

with open('sets/C16.pkl','rb') as f:
    C16 = pickle.load(f)

with open('sets/C17.pkl','rb') as f:
    C17 = pickle.load(f)

with open('sets/C18.pkl','rb') as f:
    C18 = pickle.load(f)

with open('sets/C19.pkl','rb') as f:
    C19 = pickle.load(f)

with open('sets/C20.pkl','rb') as f:
    C20 = pickle.load(f)

with open('sets/C21.pkl','rb') as f:
    C21 = pickle.load(f)

with open('sets/banner_pos.pkl','rb') as f:
    banner_pos = pickle.load(f)

with open('sets/site_id.pkl','rb') as f:
    site_id = pickle.load(f)

with open('sets/site_domain.pkl','rb') as f:
    site_domain = pickle.load(f)

with open('sets/site_category.pkl','rb') as f:
    site_category = pickle.load(f)

with open('sets/app_id.pkl','rb') as f:
    app_id = pickle.load(f)

with open('sets/app_domain.pkl','rb') as f:
    app_domain = pickle.load(f)

with open('sets/app_category.pkl','rb') as f:
    app_category = pickle.load(f)

with open('sets/device_id_count.pkl','rb') as f:
    device_id_count = pickle.load(f)

with open('sets/device_id_frequency.pkl','rb') as f:
    device_id_frequency = pickle.load(f)

with open('sets/device_model.pkl','rb') as f:
    device_model = pickle.load(f)

with open('sets/device_type.pkl','rb') as f:
    device_type = pickle.load(f)

with open('sets/device_conn_type.pkl','rb') as f:
    device_conn_type = pickle.load(f)


for field in fields:
    print(field)
    print(len(eval(field+'_v').difference(eval(field))))
    print(eval(field+'_v').difference(eval(field)))
