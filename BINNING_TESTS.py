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
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

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

classifiers = [('LogisticRegression',LogisticRegression()),
               ('LinearSVC',LinearSVC()),
               ('KNeighborsClassifier',KNeighborsClassifier()),
               ('SGDClassifier',SGDClassifier()),
               ('Gaussian',GaussianNB()),
               ('SVC',SVC(C=11,kernel='rbf',gamma=1)),
               ('NuSVC',NuSVC()),
               ('RandomForestClassifier',RandomForestClassifier(n_estimators= 200,random_state=123))]

encoders = [('WOEEncoder',WOEEncoder()),
            ('TargetEncoder',TargetEncoder()),
            ('CatBoostEncoder',CatBoostEncoder())]

enc_name = []
clf_name = []
score = []

for index, encoder in enumerate (encoders):
    for index, classifier in enumerate (classifiers):
        pipe = Pipeline(steps=[('encoder', encoder[1]),
                            ('scaler', MinMaxScaler()),
                            ('classifier', classifier[1])])
        pipe.fit(X_train, y_train)
        #print(encoder,classifier, " model score: %.3f" % pipe.score(X_test, y_test))
        pipe_pred = pipe.predict(X_test)

        enc_name.append(encoder[0])
        clf_name.append(classifier[0])
        score.append(accuracy_score(y_test,pipe_pred))

        print(encoder[0],classifier[0],accuracy_score(y_test,pipe_pred))

baseline = pd.DataFrame()
baseline['Encoder'] = enc_name
baseline['Classifier'] = clf_name
baseline['Accuracy Score'] = score

print(baseline.head(10))

