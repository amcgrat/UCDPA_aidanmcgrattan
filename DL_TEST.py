from sklearn.model_selection import RandomizedSearchCV
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from category_encoders import *
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.neural_network import MLPClassifier
from sodapy import Socrata

#import csv files.
#results_df_sample = pd.read_csv('C:/Users/amcgrat/Desktop/UCD PROGRAM/Project/HPCPS/HCPCS_CODES_ALL_SAMPLE.csv')
hcpcs_cs_cat = pd.read_csv('C:/Users/amcgrat/Desktop/UCD PROGRAM/Project/HPCPS/HCPCS_CODES_ALL_1.csv')

client = Socrata("data.cms.gov", None)
results = client.get('efgi-jnkv',
                     select ='hcpcs_cd,'
                             'psps_submitted_charge_amt,'
                             'hcpcs_initial_modifier_cd,'
                             'hcpcs_second_modifier_cd,'
                             'carrier_num,'
                             'pricing_locality_cd,'
                             'type_of_service_cd,'
                             'place_of_service_cd,'
                             'provider_spec_cd,'
                             'psps_submitted_service_cnt,'
                             'psps_denied_services_cnt,'
                             'psps_hcpcs_asc_ind_cd,'
                             'hcpcs_betos_cd',
                     where = 'psps_submitted_charge_amt > 5',
                     limit = 3400000)
results_df_sample = pd.DataFrame.from_records(results)
client.download_attachments("efgi-jnkv", download_dir="~/Desktop")
client.close()

results_df_sample =  results_df_sample.sample(frac = 0.25,random_state=123)

results_df_sample['hcpcs_cd']=results_df_sample.hcpcs_cd.str.pad(5,side='left',fillchar='0')

#join dataframe to code categories dataframe.
results_df_sample=pd.merge(results_df_sample, hcpcs_cs_cat, on=['hcpcs_cd'], how='left')

#Replace nulls in denied services count with zero.
results_df_sample['psps_denied_services_cnt'].replace(np.NaN,'0',inplace=True)
results_df_sample['hcpcs_initial_modifier_cd'].replace(np.NaN,'NA',inplace=True)
results_df_sample['hcpcs_second_modifier_cd'].replace(np.NaN,'NA',inplace=True)
results_df_sample['hcpcs_betos_cd'].replace(np.NaN,'NA',inplace=True)
results_df_sample['psps_hcpcs_asc_ind_cd'].replace(np.NaN,'NA',inplace=True)
results_df_sample['pricing_locality_cd'].replace(np.NaN,'NA',inplace=True)

results_df_sample['psps_denied_services_cnt'] = results_df_sample['psps_denied_services_cnt'].astype(float)
results_df_sample['psps_submitted_charge_amt'] = results_df_sample['psps_submitted_charge_amt'].astype(float)
results_df_sample['psps_submitted_service_cnt'] = results_df_sample['psps_submitted_service_cnt'].astype(float)


#calculate charge per service and create new column
results_df_sample['chg_per_svc'] = results_df_sample['psps_submitted_charge_amt']/results_df_sample['psps_submitted_service_cnt']

results_df_sample['sub_chg_amt_binn'] = pd.qcut(results_df_sample['psps_submitted_charge_amt'],q = 60,labels=False,duplicates='drop')
results_df_sample['sub_svc_cnt_binn'] = pd.qcut(results_df_sample['psps_submitted_service_cnt'],q = 60,labels=False,duplicates='drop')
results_df_sample['chg_per_svc_binn'] = pd.qcut(results_df_sample['chg_per_svc'],q = 60,labels=False,duplicates='drop')

#create denied column and convert to int.  This will function as the target/label feature
results_df_sample['denied'] = results_df_sample['psps_denied_services_cnt']>0
results_df_sample['denied'] = results_df_sample["denied"].astype(int)

for col in ['hcpcs_cd',
            'carrier_num',
            'pricing_locality_cd',
            'type_of_service_cd',
            'place_of_service_cd',
            'provider_spec_cd',
            'psps_hcpcs_asc_ind_cd',
            'hcpcs_betos_cd',
            'hcpcs_initial_modifier_cd',
            'hcpcs_second_modifier_cd',
            'cd_categories',
            'sub_chg_amt_binn',
            'sub_svc_cnt_binn',
            'chg_per_svc_binn']:
    results_df_sample[col] = results_df_sample[col].astype('category')


y = results_df_sample['denied']

results_df_sample.drop(['psps_submitted_service_cnt',
                        'chg_per_svc',
                        'denied'
                        ], inplace=True,axis = 1)

X_train, X_test, y_train, y_test = train_test_split(results_df_sample, y, stratify=y,test_size=0.20, random_state=123)

print(results_df_sample[results_df_sample.isnull().any(axis=1)][null_columns].head())

print(X_train.info())
print(X_test.info())

WOE_encoder = WOEEncoder()
X_train_enc = WOE_encoder.fit_transform(X_train, y_train)
X_test_enc = WOE_encoder.transform(X_test)

scaler = MinMaxScaler()
X_train_enc_scaled = pd.DataFrame(scaler.fit_transform(X_train_enc, y_train))
X_test_enc_scaled = pd.DataFrame(scaler.transform(X_test_enc))

mlp = MLPClassifier(hidden_layer_sizes=(13,13,13,13),max_iter=1000,learning_rate= 'adaptive',random_state=123)
print(mlp.get_params())
mlp.fit(X_train_enc_scaled,y_train)
mlp_pred = mlp.predict(X_test_enc_scaled)
print("MLP Accuracy :", accuracy_score(y_test, mlp_pred))
print(confusion_matrix(y_test,mlp_pred))
print(classification_report(y_test,mlp_pred))