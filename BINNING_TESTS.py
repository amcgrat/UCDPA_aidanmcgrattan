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
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report



pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


#import csv files.
results_df_sample = pd.read_csv('C:/Users/amcgrat/Desktop/UCD PROGRAM/Project/HPCPS/HCPCS_CODES_ALL_SAMPLE_001.csv')
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



results_df_sample = results_df_sample.astype({'psps_denied_services_cnt': np.int32,
                                              'psps_submitted_charge_amt': np.int32,
                                              'psps_submitted_service_cnt':np.int32})


#calculate charge per service and create new column
results_df_sample['chg_per_svc'] = results_df_sample['psps_submitted_charge_amt']/results_df_sample['psps_submitted_service_cnt']


results_df_sample['sub_chg_amt_binn'] = pd.qcut(results_df_sample['psps_submitted_charge_amt'],q = 25,labels=False,duplicates='drop')
results_df_sample['sub_svc_cnt_binn'] = pd.qcut(results_df_sample['psps_submitted_service_cnt'],q = 25,labels=False,duplicates='drop')
results_df_sample['chg_per_svc_binn'] = pd.qcut(results_df_sample['chg_per_svc'],q = 25,labels=False,duplicates='drop')

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

results_df_sample.drop(['psps_submitted_charge_amt',
                        'psps_submitted_service_cnt',
                        'chg_per_svc','denied',
                        'psps_denied_services_cnt',
                        'chg_per_svc_binn'], inplace=True,axis = 1)



X_train, X_test, y_train, y_test = train_test_split(results_df_sample, y, stratify=y,test_size=0.25, random_state=123)

#Will not increase number of columns, one per category transformed.  May reduce target leakage and overfitting present in LOO
CBE_encoder = CatBoostEncoder()
X_train_enc = CBE_encoder.fit_transform(X_train, y_train)
X_test_enc = CBE_encoder.transform(X_test)

scaler = MinMaxScaler()
X_train_enc_scaled = pd.DataFrame(scaler.fit_transform(X_train_enc, y_train))
X_test_enc_scaled = pd.DataFrame(scaler.transform(X_test_enc))



print(X_train_enc_scaled.head(10))

model = LogisticRegression(C=1,max_iter=100)
model.fit(X_train_enc_scaled, y_train)
model_pred = model.predict_proba(X_test_enc_scaled)[:, 1]

score = auc(y_test, model_pred)
print("score: ", score)

#Kernel Ridge
from sklearn.kernel_ridge import KernelRidge

model_kr = KernelRidge(kernel="rbf", alpha=1e-3, gamma=0.05)
model_kr.fit(X_train_enc_scaled, y_train)
model_pred_kr = model_kr.predict(X_test_enc_scaled)

score_kr = auc(y_test, model_pred_kr)
print("score kr: ", score_kr)




#from sklearn.svm import SVC
#params_grid = {'C':[100],'gamma':[1],'kernel':['rbf']}
#grid = GridSearchCV(SVC(),params_grid,refit=True,verbose = 3)
#grid.fit(X_train_enc_scaled, y_train)
#print(grid.best_params_)
#print(grid.best_estimator_)

#grid_pred = grid.predict(X_test_enc_scaled)

#print(classification_report(y_test,grid_pred))

#score_svc_grid = auc(y_test, grid_pred)
#print("score SVC grid: ", score_svc_grid)

from sklearn.svm import SVC
params_grid = {'alpha':[10,1,0.1,0.001,0.0001],'gamma':[1,0.5,0.1,0.01],'kernel':['rbf','linear','poly','cosine']}
grid = GridSearchCV(KernelRidge(),params_grid,refit=True,verbose = 3)
grid.fit(X_train_enc_scaled, y_train)
print(grid.best_params_)
print(grid.best_estimator_)

grid_pred = grid.predict(X_test_enc_scaled)

print(classification_report(y_test,grid_pred))

score_svc_grid = auc(y_test, grid_pred)
print("score SVC grid: ", score_svc_grid)