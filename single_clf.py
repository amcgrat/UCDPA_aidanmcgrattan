import numpy as np
import pandas as pd
from category_encoders import *
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score as auc,confusion_matrix,ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import classification_report,accuracy_score,precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def display_settings():
    pd.set_option('display.max_rows', 20)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

display_settings()

#import csv files.
results_df_sample = pd.read_csv('C:/Users/amcgrat/Desktop/UCD PROGRAM/Project/HPCPS/HCPCS_CODES_ALL_SAMPLE_20.csv')
independent_df_sample = pd.read_csv('C:/Users/amcgrat/Desktop/UCD PROGRAM/Project/HPCPS/HCPCS_CODES_ALL_SAMPLE_001.csv')
hcpcs_cs_cat = pd.read_csv('C:/Users/amcgrat/Desktop/UCD PROGRAM/Project/HPCPS/HCPCS_CODES_ALL_1.csv')

#pad codes hcpcs_cd with zeros
results_df_sample['hcpcs_cd']=results_df_sample.hcpcs_cd.str.pad(5,side='left',fillchar='0')

#join dataframe to code categories dataframe.
results_df_sample=pd.merge(results_df_sample, hcpcs_cs_cat, on=['hcpcs_cd'], how='left')
results_df_sample.rename(columns={'cd_categories_x': 'cd_categories'}, inplace=True)
print(results_df_sample.info())

#Replace nulls in denied services count with zero.
results_df_sample['psps_denied_services_cnt'].replace(np.NaN,'0',inplace=True)
results_df_sample['hcpcs_initial_modifier_cd'].replace(np.NaN,'NA',inplace=True)
results_df_sample['hcpcs_second_modifier_cd'].replace(np.NaN,'NA',inplace=True)
results_df_sample['hcpcs_betos_cd'].replace(np.NaN,'NA',inplace=True)
results_df_sample['psps_hcpcs_asc_ind_cd'].replace(np.NaN,'NA',inplace=True)
results_df_sample['pricing_locality_cd'].replace(np.NaN,'NA',inplace=True)

#results_df_sample['sub_chg_log']= np.log(results_df_sample['psps_submitted_charge_amt'])
#results_df_sample['sub_svc_cnt']= np.log(results_df_sample['psps_submitted_service_cnt'])

results_df_sample = results_df_sample.astype({'psps_denied_services_cnt': np.int32,
                                              'psps_submitted_charge_amt': np.int32,
                                              'psps_submitted_service_cnt':np.int32})


#calculate charge per service and create new column
results_df_sample['chg_per_svc'] = results_df_sample['psps_submitted_charge_amt']/results_df_sample['psps_submitted_service_cnt']

results_df_sample['sub_chg_amt_binn'] = pd.qcut(results_df_sample['psps_submitted_charge_amt'],q = 60,labels=False,duplicates='drop')
results_df_sample['sub_svc_cnt_binn'] = pd.qcut(results_df_sample['psps_submitted_service_cnt'],q = 60,labels=False,duplicates='drop')
results_df_sample['chg_per_svc_binn'] = pd.qcut(results_df_sample['chg_per_svc'],q = 60,labels=False,duplicates='drop')

#create denied column and convert to int.  This will function as the target/label feature
results_df_sample['accepted'] = results_df_sample['psps_denied_services_cnt']<1
results_df_sample['accepted'] = results_df_sample["accepted"].astype(int)


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




#df_accepted = results_df_sample[results_df_sample['accepted'] ==1]
#df_denied = results_df_sample[results_df_sample['accepted'] !=1]
#df_accepted =  df_accepted.sample(frac = 0.50,random_state=123)
#print(df_denied.shape)
#print(df_accepted.shape)
#results_df_sample = df_denied.append(df_accepted)
#print(results_df_sample.shape)

y = results_df_sample['accepted']


results_df_sample.drop(['psps_submitted_charge_amt',
                        'psps_submitted_service_cnt',
                        'chg_per_svc',
                        'accepted',
                        #'cd_categories_y',
                        'psps_denied_services_cnt'], inplace=True,axis = 1)

print(results_df_sample.head())




X_train, X_test, y_train, y_test = train_test_split(results_df_sample, y, stratify=y,test_size=0.15, random_state=123)


RF_pipeline = Pipeline(steps=[('target_encoder', TargetEncoder()),('scaler',MinMaxScaler()),('RF_clf', RandomForestClassifier(n_estimators= 500, min_samples_split= 9, min_samples_leaf= 1, max_features ='auto', max_depth= 175, bootstrap= True,class_weight='balanced'))])

RF_pipeline.fit(X_train,y_train)
RF_pred = RF_pipeline.predict(X_test)
print('RF Accuracy :',accuracy_score(y_test,RF_pred))
print('RF F1 :',f1_score(y_test,RF_pred))
print( confusion_matrix(y_test,RF_pred))
print(classification_report(y_test,RF_pred))


