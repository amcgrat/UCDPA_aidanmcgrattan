import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sodapy import Socrata
from category_encoders import *
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as auc
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV



pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


#import csv files.
results_df_sample = pd.read_csv('C:/Users/amcgrat/Desktop/UCD PROGRAM/Project/HPCPS/HCPCS_CODES_ALL_SAMPLE.csv')
hcpcs_cs_cat = pd.read_csv('C:/Users/amcgrat/Desktop/UCD PROGRAM/Project/HPCPS/HCPCS_CODES_ALL_1.csv')

#join dataframe to code categories dataframe.
results_df_sample=pd.merge(results_df_sample, hcpcs_cs_cat, on=['hcpcs_cd'], how='left')


#Replace nulls in denied services count with zero.
results_df_sample['psps_denied_services_cnt'].replace(np.NaN,'0',inplace=True)
results_df_sample['hcpcs_initial_modifier_cd'].replace(np.NaN,'NA',inplace=True)
results_df_sample['hcpcs_second_modifier_cd'].replace(np.NaN,'NA',inplace=True)
results_df_sample['hcpcs_betos_cd'].replace(np.NaN,'NA',inplace=True)
results_df_sample['psps_hcpcs_asc_ind_cd'].replace(np.NaN,'NA',inplace=True)
results_df_sample['pricing_locality_cd'].replace(np.NaN,'NA',inplace=True)



#change datatypes for submitted amount and denied service count columns.
#cols = ['psps_denied_services_cnt', 'psps_submitted_charge_amt','psps_submitted_service_cnt']
#results_df_sample[cols] = results_df_sample[cols].apply(pd.to_numeric, errors='coerce', axis=1)

results_df_sample = results_df_sample.astype({'psps_denied_services_cnt': np.int32,
                                              'psps_submitted_charge_amt': np.int32,
                                              'psps_submitted_service_cnt':np.int32})


#calculate charge per service and create new column
results_df_sample['chg_per_svc'] = results_df_sample['psps_submitted_charge_amt']/results_df_sample['psps_submitted_service_cnt']

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
            'cd_categories']:
    results_df_sample[col] = results_df_sample[col].astype('category')


print(results_df_sample.info())
#Try this for first try.  May create a lot of columns
#categorical_lst = results_df_sample.select_dtypes(include=['category']).columns.tolist()
#encoder=ce.BinaryEncoder(cols=categorical_lst)
#df_bin_spec_cd = encoder.fit_transform(results_df_sample[categorical_lst])
#print(df_bin_spec_cd.head(20))

#Will create a reduced list of columns, one per category transformed.  May cause target leakage and overfitting
#LOOE_encoder = LeaveOneOutEncoder()
#train_looe = LOOE_encoder.fit_transform(results_df_sample[categorical_lst], results_df_sample['denied'])
#test_looe = LOOE_encoder.transform(results_df_sample[categorical_lst])
#print(test_looe.head(20))

categorical_lst = results_df_sample.select_dtypes(include=['category']).columns.tolist()
X_num = results_df_sample[['psps_submitted_charge_amt','psps_submitted_service_cnt','chg_per_svc']]

X_cat = results_df_sample[categorical_lst]
y = results_df_sample['denied']

frames = [X_num,X_cat]

X = pd.concat(frames,axis=1)

X['lg_sub_chg'] = np.log(X['psps_submitted_charge_amt'])
X['lg_sub_scv_cnt'] = np.log(X['psps_submitted_service_cnt'])
X['lg_chg_per_svc'] = np.log(X['chg_per_svc'])

X.drop(['psps_submitted_charge_amt','psps_submitted_service_cnt','chg_per_svc'], inplace=True,axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,test_size=0.3, random_state=123)

#Will not increase number of columns, one per category transformed.  May reduce target leakage and overfitting present in LOO
CBE_encoder = CatBoostEncoder()
X_train_enc = CBE_encoder.fit_transform(X_train, y_train)
X_test_enc = CBE_encoder.transform(X_test)

model = LogisticRegression(C=1,max_iter=200)
model.fit(X_train_enc, y_train)
model_pred = model.predict_proba(X_test_enc)[:, 1]

score = auc(y_test, model_pred)
print("score: ", score)

