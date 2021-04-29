import numpy as np
import pandas as pd
from category_encoders import *
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score as auc,confusion_matrix,ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,accuracy_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC

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

#Will not increase number of columns, one per category transformed.  May reduce target leakage and overfitting present in LOO
#CBE_encoder = CatBoostEncoder(cols =['hcpcs_cd',
                                #'carrier_num',
                                #'pricing_locality_cd',
                                #'type_of_service_cd',
                                #'place_of_service_cd',
                                #'provider_spec_cd',
                                #'psps_hcpcs_asc_ind_cd',
                                #'hcpcs_betos_cd',
                                #'hcpcs_initial_modifier_cd',
                                #'hcpcs_second_modifier_cd',
                                #'cd_categories'])
#X_train_enc = CBE_encoder.fit_transform(X_train, y_train)
#X_test_enc = CBE_encoder.transform(X_test)

#TE_encoder = TargetEncoder()
#X_train_enc = TE_encoder.fit_transform(X_train, y_train)
#X_test_enc = TE_encoder.transform(X_test)

#WOE_encoder = WOEEncoder()
#X_train_enc = WOE_encoder.fit_transform(X_train, y_train)
#X_test_enc = WOE_encoder.transform(X_test)

CBE_encoder = CatBoostEncoder()
X_train_enc = CBE_encoder.fit_transform(X_train, y_train)
X_test_enc = CBE_encoder.transform(X_test)

scaler = MinMaxScaler()
X_train_enc_scaled = pd.DataFrame(scaler.fit_transform(X_train_enc, y_train))
X_test_enc_scaled = pd.DataFrame(scaler.transform(X_test_enc))

#LR = LogisticRegression(C=10,max_iter=100,solver = 'lbfgs')
#LR.fit(X_train_enc_scaled, y_train)
#LR_pred = LR.predict(X_test_enc_scaled)
#print('LR Accuracy :',accuracy_score(y_test,LR_pred))
#print(LR.get_params())

#print( confusion_matrix(y_test,LR_pred))
#print(classification_report(y_test,LR_pred))


#from sklearn.svm import SVC, LinearSVC, NuSVC
#LSVC = LinearSVC()
#LSVC.fit(X_train_enc_scaled,y_train)
#LSVC_pred = LSVC.predict(X_test_enc_scaled)
#print("LSVC Accuracy :", accuracy_score(y_test, LSVC_pred,))
#print(LSVC.get_params())

#print( confusion_matrix(y_test,LSVC_pred))
#print(classification_report(y_test,LSVC_pred))


from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(algorithm='kd_tree', metric='manhattan', n_neighbors=9,weights='distance')
KNN.fit(X_train_enc_scaled,y_train)
KNN_pred = KNN.predict(X_test_enc_scaled)
print('KNN Accuracy :',accuracy_score(y_test,KNN_pred))
print(KNN.get_params)

print( confusion_matrix(y_test,KNN_pred))
print(classification_report(y_test,KNN_pred))


#SGDC = SGDClassifier()
#SGDC.fit(X_train_enc_scaled,y_train)
#SGDC_pred = SGDC.predict(X_test_enc_scaled)
#print("SGDC Accuracy :", accuracy_score(y_test, SGDC_pred))
#print(SGDC.get_params())

#print( confusion_matrix(y_test,SGDC_pred))
#print(classification_report(y_test,SGDC_pred))

#GnB = GaussianNB()
#GnB.fit(X_train_enc_scaled,y_train)
#GnB_pred = GnB.predict(X_test_enc_scaled)
#print("GnB Accuracy :", accuracy_score(y_test, GnB_pred))
#print(GnB.get_params())

#print( confusion_matrix(y_test,GnB_pred))
#print(classification_report(y_test,GnB_pred))


#SVC = SVC(C=11,kernel='rbf',gamma=1)
#SVC.fit(X_train_enc_scaled,y_train)
#SVC_pred = SVC.predict(X_test_enc_scaled)
#print("SVC Accuracy :", accuracy_score(y_test, SVC_pred))
#print(SVC.get_params())

#print( confusion_matrix(y_test,SVC_pred))
#print(classification_report(y_test,SVC_pred))


#NSVC = NuSVC()
#NSVC.fit(X_train_enc_scaled,y_train)
#NSVC_pred = NSVC.predict(X_test_enc_scaled)
#print("NSVC Accuracy :", accuracy_score(y_test, NSVC_pred))
#print(NSVC.get_params())

#print( confusion_matrix(y_test,NSVC_pred))
#print(classification_report(y_test,NSVC_pred))


#from sklearn.ensemble import RandomForestClassifier

#RFC = RandomForestClassifier()
#RFC.fit(X_train_enc_scaled,y_train)
#RFC_pred = RFC.predict(X_test_enc_scaled)
#print("RF Accuracy :", accuracy_score(y_test, RFC_pred))
#print (RFC.get_params())

#print( confusion_matrix(y_test,RFC_pred))
#print(classification_report(y_test,RFC_pred))


