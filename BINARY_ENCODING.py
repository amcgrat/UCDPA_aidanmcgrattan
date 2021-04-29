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


#results_df_sample['sub_chg_amt_binn'] = pd.qcut(results_df_sample['psps_submitted_charge_amt'],q = 60,labels=False,duplicates='drop')
#results_df_sample['sub_svc_cnt_binn'] = pd.qcut(results_df_sample['psps_submitted_service_cnt'],q = 60,labels=False,duplicates='drop')


#create denied column and convert to int.  This will function as the target/label feature
results_df_sample['denied'] = results_df_sample['psps_denied_services_cnt']>0
results_df_sample['denied'] = results_df_sample["denied"].astype(int)


y = results_df_sample['denied']

#Replace nulls in denied services count with zero.
results_df_sample['psps_denied_services_cnt'].replace(np.NaN,'0',inplace=True)
results_df_sample['hcpcs_initial_modifier_cd'].replace(np.NaN,'NA',inplace=True)
results_df_sample['hcpcs_second_modifier_cd'].replace(np.NaN,'NA',inplace=True)
results_df_sample['hcpcs_betos_cd'].replace(np.NaN,'NA',inplace=True)
results_df_sample['psps_hcpcs_asc_ind_cd'].replace(np.NaN,'NA',inplace=True)
results_df_sample['pricing_locality_cd'].replace(np.NaN,'NA',inplace=True)

results_df_sample['sub_chg_log']= np.log(results_df_sample['psps_submitted_charge_amt'])
results_df_sample['sub_svc_cnt']= np.log(results_df_sample['psps_submitted_service_cnt'])

results_df_sample.drop(['denied',
                        'psps_denied_services_cnt',
                        'psps_submitted_charge_amt',
                        'psps_submitted_service_cnt'], inplace=True,axis = 1)

print(results_df_sample.info())


X_train, X_test, y_train, y_test = train_test_split(results_df_sample, y, stratify=y,test_size=0.20, random_state=123)



encoder= ce.BaseNEncoder(cols =['hcpcs_cd',
                                'sub_chg_log',
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
                                ],return_df=True,base=8)
X_train_enc = encoder.fit_transform(X_train, y_train)
X_test_enc = encoder.transform(X_test)



print(X_train_enc.shape)

#Will not increase number of columns, one per category transformed.  May reduce target leakage and overfitting present in LOO
#CBE_encoder = CatBoostEncoder()
#X_train_enc = CBE_encoder.fit_transform(X_train, y_train)
#X_test_enc = CBE_encoder.transform(X_test)

#TE_encoder = TargetEncoder()
#X_train_enc = TE_encoder.fit_transform(X_train, y_train)
#X_test_enc = TE_encoder.transform(X_test)

#WOE_encoder = WOEEncoder()
#X_train_enc = WOE_encoder.fit_transform(X_train, y_train)
#X_test_enc = WOE_encoder.transform(X_test)

scaler = MinMaxScaler()
X_train_enc_scaled = pd.DataFrame(scaler.fit_transform(X_train_enc, y_train))
X_test_enc_scaled = pd.DataFrame(scaler.transform(X_test_enc))

LR = LogisticRegression(C=10,max_iter=100,solver = 'lbfgs')
LR.fit(X_train_enc_scaled, y_train)
LR_pred = LR.predict(X_test_enc_scaled)
print('LR Accuracy :',accuracy_score(y_test,LR_pred))

from sklearn.svm import SVC, LinearSVC, NuSVC
LSVC = LinearSVC()
LSVC.fit(X_train_enc_scaled,y_train)
LSVC_pred = LSVC.predict(X_test_enc_scaled)
print("LSVC Accuracy :", accuracy_score(y_test, LSVC_pred,))

from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=10)
KNN.fit(X_train_enc_scaled,y_train)
KNN_pred = KNN.predict(X_test_enc_scaled)
print('KNN Accuracy :',accuracy_score(y_test,KNN_pred))

SGDC = SGDClassifier()
SGDC.fit(X_train_enc_scaled,y_train)
SGDC_pred = SGDC.predict(X_test_enc_scaled)
print("SGDC Accuracy :", accuracy_score(y_test, SGDC_pred))

GnB = GaussianNB()
GnB.fit(X_train_enc_scaled,y_train)
GnB_pred = GnB.predict(X_test_enc_scaled)
print("GnB Accuracy :", accuracy_score(y_test, GnB_pred))

BnB = BernoulliNB()
BnB.fit(X_train_enc,y_train)
BnB_pred = BnB.predict(X_test_enc_scaled)
print("BnB Accuracy :", accuracy_score(y_test, BnB_pred))

MnB = MultinomialNB()
MnB.fit(X_train_enc_scaled,y_train)
MnB_pred = MnB.predict(X_test_enc_scaled)
print("MnB Accuracy :", accuracy_score(y_test, MnB_pred))

SVC = SVC()
SVC.fit(X_train_enc_scaled,y_train)
SVC_pred = SVC.predict(X_test_enc_scaled)
print("SVC Accuracy :", accuracy_score(y_test, SVC_pred))

NSVC = NuSVC()
NSVC.fit(X_train_enc_scaled,y_train)
NSVC_pred = NSVC.predict(X_test_enc_scaled)
print("NSVC Accuracy :", accuracy_score(y_test, NSVC_pred))

from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(n_estimators=200, random_state=123,class_weight='balanced')
RFC.fit(X_train_enc_scaled,y_train)
RFC_pred = RFC.predict(X_test_enc_scaled)
print("RF Accuracy :", accuracy_score(y_test, RFC_pred))
print("RF Fetaure Importance :",RFC.feature_importances_)

#print( confusion_matrix(y_test,RFC_pred))
#print(classification_report(y_test,RFC_pred))

print(X_test_enc_scaled.head(10))