import numpy as np
import pandas as pd
from category_encoders import *
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score as auc,confusion_matrix,ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


#import csv files.
results_df_sample = pd.read_csv('C:/Users/amcgrat/Desktop/UCD PROGRAM/Project/HPCPS/HCPCS_CODES_ALL_SAMPLE.csv')
independent_df_sample = pd.read_csv('C:/Users/amcgrat/Desktop/UCD PROGRAM/Project/HPCPS/HCPCS_CODES_ALL_SAMPLE_001.csv')
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



results_df_sample.drop(['psps_submitted_charge_amt',
                        'psps_submitted_service_cnt',
                        'chg_per_svc',
                        'denied',
                        'psps_denied_services_cnt'], inplace=True,axis = 1)



X_train, X_test, y_train, y_test = train_test_split(results_df_sample, y, stratify=y,test_size=0.20, random_state=123)

print(X_train.info())
print(X_test.info())




#TE_encoder = TargetEncoder()
#X_train_enc = TE_encoder.fit_transform(X_train, y_train)
#X_test_enc = TE_encoder.transform(X_test)

#WOE_encoder = WOEEncoder()
#X_train_enc = WOE_encoder.fit_transform(X_train, y_train)
#X_test_enc = WOE_encoder.transform(X_test)

#CBE_encoder = CatBoostEncoder()
#X_train_enc = CBE_encoder.fit_transform(X_train, y_train)
#X_test_enc = CBE_encoder.transform(X_test)

#scaler = MinMaxScaler()
#X_train_enc_scaled = pd.DataFrame(scaler.fit_transform(X_train_enc, y_train))
#X_test_enc_scaled = pd.DataFrame(scaler.transform(X_test_enc))



KNN = KNeighborsClassifier(algorithm='kd_tree', metric='manhattan', n_neighbors=9,weights='distance')

SVC_ = SVC(C=11,kernel='rbf',gamma=1)

RFC = RandomForestClassifier()

KNN_pipeline = Pipeline(steps=[('target_encoder', TargetEncoder()),
                                 ('scaler',MinMaxScaler()),
                                 ('knn_clf', KNeighborsClassifier(algorithm='kd_tree', metric='manhattan', n_neighbors=9,weights='distance'))
                                 ])
KNN_pipeline.fit(X_train,y_train)
KNN_pred = KNN_pipeline.predict(X_test)
print('KNN Accuracy :',accuracy_score(y_test,KNN_pred))
print( confusion_matrix(y_test,KNN_pred))
print(classification_report(y_test,KNN_pred))

SVC_pipeline = Pipeline(steps=[('target_encoder', TargetEncoder()),
                                 ('scaler',MinMaxScaler()),
                                 ('knn_clf', SVC(C=11,kernel='rbf',gamma=1))
                                 ])
SVC_pipeline.fit(X_train,y_train)
SVC_pred = SVC_pipeline.predict(X_test)
print('SVC Accuracy :',accuracy_score(y_test,SVC_pred))
print( confusion_matrix(y_test,SVC_pred))
print(classification_report(y_test,SVC_pred))

RF_pipeline = Pipeline(steps=[('woe_encoder', WOEEncoder()),
                                 ('scaler',MinMaxScaler()),
                                 ('knn_clf', RandomForestClassifier(bootstrap=True, max_depth=175, max_features='auto', min_samples_leaf=1, min_samples_split=9, n_estimators= 2000))
                                 ])
RF_pipeline.fit(X_train,y_train)
RF_pred = RF_pipeline.predict(X_test)
print('RF Accuracy :',accuracy_score(y_test,RF_pred))
print( confusion_matrix(y_test,RF_pred))
print(classification_report(y_test,RF_pred))

