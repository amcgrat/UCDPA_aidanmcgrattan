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
from sklearn.metrics import classification_report,accuracy_score,precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def display_settings():
    pd.set_option('display.max_rows', 20)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

display_settings()

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

def get_classifiers():
        classifiers=[('LogisticRegression',LogisticRegression()),
                    ('LinearSVC',LinearSVC(random_state=123)),
                    ('KNeighborsClassifier',KNeighborsClassifier()),
                    ('SGDClassifier',SGDClassifier(random_state=123)),
                    ('Gaussian',GaussianNB()),
                    ('SVC',SVC(random_state=123)),
                    ('NuSVC',NuSVC(random_state=123)),
                    ('RandomForestClassifier',RandomForestClassifier(random_state=123)),
                    ('ADABoost',AdaBoostClassifier()),
                    ('GradientBoostingClassifier',GradientBoostingClassifier())]
        return classifiers

def get_encoders():
        encoders=[('WOEEncoder',WOEEncoder()),
                 ('TargetEncoder',TargetEncoder()),
                 ('CatBoostEncoder',CatBoostEncoder())]
        return encoders

enc_name=[]
clf_name=[]
acc_score=[]
F1_score=[]

def evaluate_models():
        encoders=get_encoders()
        classifiers=get_classifiers()
        for index, encoder in enumerate (encoders):
            for index, classifier in enumerate (classifiers):
                pipe = Pipeline(steps=[('encoder', encoder[1]),
                                       ('scaler', MinMaxScaler()),
                                       ('classifier', classifier[1])])
                pipe.fit(X_train, y_train)
                pipe_pred=pipe.predict(X_test)
                #print(encoder[0],classifier[0],accuracy_score(y_test,pipe_pred),f1_score(y_test, pipe_pred))
                enc_name.append(encoder[0])
                clf_name.append(classifier[0])
                acc_score.append(accuracy_score(y_test,pipe_pred))
                F1_score.append(f1_score(y_test, pipe_pred))

evaluate_models()

def model_results():
        baseline=pd.DataFrame()
        baseline['Encoder']=enc_name
        baseline['Classifier']=clf_name
        baseline['Accuracy Score']=acc_score
        baseline['f1 Score']=F1_score
        baseline.to_csv('C:/Users/amcgrat/Desktop/UCD PROGRAM/Project/HPCPS/clf_enc_eval_5.csv',index=False)
        baseline_clf=baseline.iloc[baseline.groupby(['Classifier']).apply(lambda x: x['Accuracy Score'].idxmax())]
        return baseline_clf


def display_top_clf():
        baseline_top_10=model_results()
        baseline_top_10.sort_values(by='Accuracy Score', ascending=False,inplace=True)
        print(baseline_top_10)
        return baseline_top_10

display_top_clf()