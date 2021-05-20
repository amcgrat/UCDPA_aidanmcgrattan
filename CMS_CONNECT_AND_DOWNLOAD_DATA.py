import numpy as np
import pandas as pd
from category_encoders import *
import category_encoders as ce
import matplotlib.pyplot as plt
import seaborn as sns
from category_encoders import *
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier,StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
from sklearn.model_selection import train_test_split,cross_validate,GridSearchCV,RandomizedSearchCV
from sklearn.pipeline import Pipeline
from time import time
import re
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def set_display():
    '''
        Sets Display Options including warnings
        Parameters:
            N/A
        Returns:
            N/A
    '''
    pd.set_option('display.max_rows', 20)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    #Suppress 'FutureWarning' messages
    warnings.simplefilter(action='ignore', category=FutureWarning)

set_display()

def api_connection(n):
    '''
        Connects to API and downloads datafile and attachments
        Parameters:
            n(int): a whole number
        Returns:
            DataFrame: Pandas Dataframe with downloaded data from API
        Outputs:
            attachment: attachments from API
    '''
    from sodapy import Socrata
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
                             'hcpcs_betos_cd',
                     where = 'psps_submitted_charge_amt > 5',
                     limit = n)
    results_df = pd.DataFrame.from_records(results)

    client.download_attachments("efgi-jnkv", download_dir='C:/Users/amcgrat/Desktop/UCD PROGRAM/Project/HPCPS')

    client.close()
    return results_df

results_df = api_connection(3400000)

def create_sample(df,n):
    '''
            Creates random sample of initial dataframe download
            Parameters:
                n(int): a decimal integer between 0 and 1
                DataFrame: Pandas DataFrame
            Returns:
                DataFrame: Random sample of dataframe
    '''
    results_df_sample =  results_df.sample(frac = n,random_state=123)
    #pad hcpcs_cd codes with zeros
    results_df_sample['hcpcs_cd']=results_df_sample.hcpcs_cd.str.pad(5,side='left',fillchar='0')
    # Import cd codes csv file
    hcpcs_cs_cat = pd.read_csv('C:/Users/amcgrat/Desktop/UCD PROGRAM/Project/HPCPS/HCPCS_CODES_ALL_1.csv')
    # join dataframe to hcpcs code categories
    results_df_sample = pd.merge(results_df_sample, hcpcs_cs_cat, on=['hcpcs_cd'], how='left')
    # output file to disk
    #results_df_sample.to_csv('C:/Users/amcgrat/Desktop/UCD PROGRAM/Project/HPCPS/HCPCS_CODES_ALL_SAMPLE_20.csv',index=False)
    return results_df_sample

results_sample = create_sample(results_df,0.2)

def code_groups(df):
    results_sample['hcpcs_groups'] = results_sample.hcpcs_cd.apply(lambda x: (x[-1:]+x[:1]) if re.match(r'\d{4}[a-zA-Z]{1}',x) else x[:2])

    return results_sample

results_sample=code_groups(results_sample)


def remove_nulls(df):
    '''
                Remove null values from selected columns
                Parameters:
                    DataFrame: Pandas DataFrame
                Returns:
                    DataFrame: Pandas DataFrame
    '''
    df['psps_denied_services_cnt'].replace(np.NaN,'0',inplace=True)
    df['hcpcs_initial_modifier_cd'].replace(np.NaN,'NA',inplace=True)
    df['hcpcs_second_modifier_cd'].replace(np.NaN,'NA',inplace=True)
    df['hcpcs_betos_cd'].replace(np.NaN,'NA',inplace=True)
    df['psps_hcpcs_asc_ind_cd'].replace(np.NaN,'NA',inplace=True)
    df['pricing_locality_cd'].replace(np.NaN,'NA',inplace=True)
    return df

results_sample=remove_nulls(results_sample)

def change_data_types(df):
    #change datatypes
    results_chg = results_sample.astype({'psps_denied_services_cnt': np.float64,
                                         'psps_submitted_charge_amt': np.float64,
                                         'psps_submitted_service_cnt': np.float64})
    return results_chg

results_sample = change_data_types(results_sample)

#prepend_string to column values
def pre_pend_cd(df):
    results_sample['carrier_num'] = 'cn_' + results_sample['carrier_num'].astype(str)
    results_sample['place_of_service_cd'] = 'pos_' + results_sample['place_of_service_cd'].astype(str)

    return results_sample

results_sample = pre_pend_cd(results_sample)

#add new columns
def add_new_cols(df,n):
    #create calculated field
    results_sample['chg_per_svc'] = results_sample['psps_submitted_charge_amt'] / results_sample['psps_submitted_service_cnt']

    #create bins to convert continuous features
    results_sample['sub_chg_amt_binn'] = pd.qcut(results_sample['psps_submitted_charge_amt'], q=60, labels=False,duplicates='drop')
    results_sample['sub_svc_cnt_binn'] = pd.qcut(results_sample['psps_submitted_service_cnt'], q=60,labels=False,duplicates='drop')
    results_sample['chg_per_svc_binn'] = pd.qcut(results_sample['chg_per_svc'], q=60, labels=False,duplicates='drop')

    #print(results_sample['chg_per_svc_binn'].value_counts())

    #results_sample['sub_chg_amt_binn'] = 'sub_' + results_sample['sub_chg_amt_binn'].astype(str)
    #results_sample['sub_svc_cnt_binn'] = 'svc_' + results_sample['sub_svc_cnt_binn'].astype(str)
    #results_sample['chg_per_svc_binn'] = 'cps_' + results_sample['chg_per_svc_binn'].astype(str)


    #create denied column and convert to int.  This will function as the traget/label feature
    results_sample['accepted'] = results_sample['psps_denied_services_cnt'] < 1
    results_sample['accepted'] = results_sample["accepted"].astype(int)


    return results_sample

results_sample = add_new_cols(results_sample,5)

def creat_den_accpt(df):
    df_denied = df[results_sample['accepted'] ==1]
    df_accepted = df[results_sample['accepted'] !=1]
    return df_denied,df_accepted


def creat_kde_df(df):
    kde_df_table = df[['psps_submitted_charge_amt','psps_submitted_service_cnt','chg_per_svc','accepted']]
    kde_df_table = kde_df_table.replace({'accepted': {1: 'accepted', 0: 'denied'}})
    return kde_df_table

def cat_df(code,n):
    df_denied,df_accepted = creat_den_accpt(results_sample)
    code_top_n_denied = df_denied[code].value_counts(sort=True).nlargest(n)
    code_top_n_accepted = df_accepted[code].value_counts(sort=True).nlargest(n)
    all_codes = pd.DataFrame(code_top_n_denied.append(code_top_n_accepted))
    all_codes = pd.DataFrame(all_codes.index)
    all_codes = pd.DataFrame.drop_duplicates(all_codes)
    cat_freq_counts =  results_sample.loc[:,[code,'accepted']]
    cat_freq_counts['class']= results_sample['accepted']
    cat_freq_counts = cat_freq_counts.replace({'class': {1: 'accepted', 0: 'denied'}})
    cat_freq_df = cat_freq_counts.groupby([code,'class'])['accepted'].count().reset_index()
    cat_freq_df.sort_values(by=['accepted'],ascending=False, inplace=True)
    cat_freq_counts_1 = pd.merge(cat_freq_df, all_codes, left_on=[code], right_on = [0],how='inner')

    return cat_freq_counts_1

def show_plts():

    hcpcs_cd = cat_df('hcpcs_cd', 25)
    carrier_num = cat_df('carrier_num', 25)
    pricing_locality_cd = cat_df('pricing_locality_cd', 25)
    type_of_service_cd = cat_df('type_of_service_cd', 25)
    place_of_service_cd = cat_df('place_of_service_cd', 25)
    provider_spec_cd = cat_df('provider_spec_cd', 25)
    psps_hcpcs_asc_ind_cd = cat_df('psps_hcpcs_asc_ind_cd', 25)
    hcpcs_betos_cd = cat_df('hcpcs_betos_cd', 25)
    hcpcs_initial_modifier_cd = cat_df('hcpcs_initial_modifier_cd', 25)
    hcpcs_second_modifier_cd = cat_df('hcpcs_second_modifier_cd', 25)
    cd_categories = cat_df('hcpcs_groups', 25)
    #sub_chg_amt_binn = cat_df('sub_chg_amt_binn', 25)
    #sub_svc_cnt_binn = cat_df('sub_svc_cnt_binn', 25)
    #chg_per_svc_binn = cat_df('chg_per_svc_binn', 25)

    fig, axes = plt.subplots(3, 4)

    sns.lineplot(data = hcpcs_cd,x = 'hcpcs_cd',y = 'accepted',hue = 'class',ax = axes[0,0],legend=False)
    sns.lineplot(data = pricing_locality_cd,x = 'pricing_locality_cd',y = 'accepted',hue = 'class',ax = axes[0,1],legend = False)
    sns.lineplot(data = carrier_num,x = 'carrier_num',y = 'accepted',hue = 'class',ax = axes[0,2],legend = False)
    sns.lineplot(data = type_of_service_cd,x = 'type_of_service_cd',y = 'accepted',hue = 'class',ax = axes[0,3],legend = False)
    sns.lineplot(data = place_of_service_cd,x = 'place_of_service_cd',y = 'accepted',hue = 'class',ax = axes[1,0],legend = False)
    sns.lineplot(data = provider_spec_cd,x = 'provider_spec_cd',y = 'accepted',hue = 'class',ax = axes[1,1],legend = False)
    sns.lineplot(data = hcpcs_betos_cd,x = 'hcpcs_betos_cd',y = 'accepted',hue = 'class',ax = axes[1,2],legend = False)
    sns.lineplot(data = hcpcs_initial_modifier_cd,x = 'hcpcs_initial_modifier_cd',y = 'accepted',hue = 'class',ax = axes[1,3],legend = False)
    sns.lineplot(data = hcpcs_second_modifier_cd,x = 'hcpcs_second_modifier_cd',y = 'accepted',hue = 'class',ax = axes[2,0],legend = False)
    sns.lineplot(data = cd_categories,x = 'hcpcs_groups',y = 'accepted',hue = 'class',ax = axes[2,1],legend = False)

    kde_df_table = creat_kde_df(results_sample)
    palette = {'accepted': 'dodgerblue','denied': 'orangered'}

    sns.kdeplot(data=kde_df_table, x="psps_submitted_charge_amt", hue='accepted', log_scale=True, fill=True,bw_adjust=.75, ax=axes[2,2],legend=False,palette=palette)
    sns.kdeplot(data=kde_df_table, x="psps_submitted_service_cnt", hue='accepted', log_scale=True, fill=True,bw_adjust=.75, ax=axes[2,3], legend=False,palette=palette)

    for i in range (0,3):
        for j in range(0,4):

            axes[i,j].set_ylabel("value frequency")
            axes[i,j].set_xticklabels("")
            axes[i,j].set_yticklabels("")
            axes[i,j].set_yticks([])
            axes[i,j].set_xticks([])
            axes[i,j].set_facecolor("aliceblue")

    axes[2,2].set_ylabel("density")
    axes[2,3].set_ylabel("density")
    #axes[3,1].set_ylabel("density")
    plt.suptitle("Accepted/Denied Frequencies",fontsize =16)

    fig.legend(title='', bbox_to_anchor=(0.1,0.875),  labels=['Accepted', 'Denied'])

    plt.show()

show_plts()

def create_target(df):
    y = results_sample['accepted']
    return y
y = create_target(results_sample)



def del_cols(df):
    results_sample.drop([#'psps_submitted_charge_amt',
                        #'psps_submitted_service_cnt',
                        'hcpcs_cd',
                        'chg_per_svc',
                        'accepted',
                        #'hcpcs_groups',
                        'cd_categories',
                        'sub_chg_amt_binn',
                        'sub_svc_cnt_binn',
                        #'hcpcs_second_modifier_cd',
                        'chg_per_svc_binn',
                        'psps_denied_services_cnt'], inplace=True, axis=1)
    return results_sample


results_sample = del_cols(results_sample)


def split_data(df,n):
    #y = create_target(results_sample)
    #del_cols()

    X_train, X_test, y_train, y_test = train_test_split(results_sample, y, stratify=y,test_size=n, random_state=123)

    return  X_train,X_test,y_train, y_test

X_train,X_test,y_train, y_test = split_data(results_sample,.2)



def get_classifiers():

    classifiers = [#('LogReg', LogisticRegression(n_jobs=-1)),
                   #('LinSVC', LinearSVC(random_state=123)),
                   #('KNN', KNeighborsClassifier(n_jobs=-1)),
                   #('SGD', SGDClassifier(random_state=123,n_jobs=-1)),
                   #('Gauss', GaussianNB()),
                   #('svc', SVC(random_state=123)),
                   #('RFC', RandomForestClassifier(random_state=123,n_jobs=-1)),
                   #('ADABoost', AdaBoostClassifier()),
                   #('GradBoost', GradientBoostingClassifier()),
                   ('RFC', RandomForestClassifier(random_state=123,n_jobs=-1))]
    return classifiers

get_classifiers()

def get_encoders():
    encoders = [#('TargetEncoder', TargetEncoder()),
                #('CatBoostEncoder', CatBoostEncoder()),
                ('WOEEncoder', WOEEncoder())]
    return encoders

get_encoders()

def plot_clfs(df):
    fig, axes = plt.subplots(2, 1)
    df.sort_values(by='accuracy_score',ascending=False,inplace = True)
    sns.lineplot(data=df, x='classifier', y='accuracy_score', hue='encoder',ax = axes[0])
    sns.lineplot(data=df, x='classifier', y='F1_score', hue='encoder',ax = axes[1])
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)
    axes[1].set_facecolor("aliceblue")
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
    axes[0].set_facecolor("aliceblue")
    plt.tight_layout()
    plt.show()

def evaluate_models():

    enc_name = []
    clf_name = []
    acc_score = []
    F1_score = []

    encoders = get_encoders()
    classifiers = get_classifiers()
    for index, encoder in enumerate(encoders):
        for index, classifier in enumerate(classifiers):
            pipe = Pipeline(steps=[('encoder', encoder[1]),
                                   ('scaler', MinMaxScaler()),
                                   ('classifier', classifier[1])])
            pipe.fit(X_train, y_train)
            pipe_pred = pipe.predict(X_test)
            #print(encoder[0],classifier[0],accuracy_score(y_test,pipe_pred),f1_score(y_test, pipe_pred))
            enc_name.append(encoder[0])
            clf_name.append(classifier[0])
            acc_score.append(accuracy_score(y_test, pipe_pred))
            F1_score.append(f1_score(y_test, pipe_pred))
            print('RF baselineAccuracy :', accuracy_score(y_test, pipe_pred))
            print('RF baseline F1 :', f1_score(y_test, pipe_pred))
            print(confusion_matrix(y_test, pipe_pred))
            print(classification_report(y_test, pipe_pred))

    zippedList =  list(zip(enc_name, clf_name, acc_score,F1_score))
    eval_results = pd.DataFrame(zippedList,columns = ['encoder' , 'classifier', 'accuracy_score','F1_score'])
    eval_results_top = eval_results.iloc[eval_results.groupby(['classifier']).apply(lambda x: x['accuracy_score'].idxmax())]
    #plot_clfs(eval_results)
    print(eval_results_top.sort_values(by = 'accuracy_score',ascending=False))

#evaluate_models()


def rfc_rndm_srchCV():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import MinMaxScaler

    WOE_encoder = WOEEncoder()
    X_train_enc = WOE_encoder.fit_transform(X_train, y_train)
    X_test_enc = WOE_encoder.transform(X_test)

    scaler = MinMaxScaler()
    X_train_enc_scaled = pd.DataFrame(scaler.fit_transform(X_train_enc, y_train))
    X_test_enc_scaled = pd.DataFrame(scaler.transform(X_test_enc))


    CV_grid = {'n_estimators': [1000,2000],
               'max_features': ['auto'],
               'max_depth': [175],
               'min_samples_split': [9],
               'min_samples_leaf': [1],
               'bootstrap': [True],
               'class_weight': [None, 'balanced']
               }
    RFC = RandomForestClassifier()
    #RFC_CV = GridSearchCV(estimator=RFC, param_grid=CV_grid, cv=5, verbose=1)
    RFC_RANDOM = RandomizedSearchCV(estimator=RFC, param_distributions=CV_grid, n_iter=35, cv=3, verbose=1,
                                    random_state=42, n_jobs=-1, refit=True, scoring='accuracy')
    RFC_RANDOM.fit(X_train_enc_scaled, y_train)

    print(RFC_RANDOM.best_estimator_)
    print(RFC_RANDOM.best_params_)
    print(RFC_RANDOM.best_score_)

#rfc_rndm_srchCV()

def svc_rndm_srchCV():

    CAT_encoder = CatBoostEncoder()
    X_train_enc = CAT_encoder.fit_transform(X_train, y_train)
    X_test_enc = CAT_encoder.transform(X_test)

    scaler = MinMaxScaler()
    X_train_enc_scaled = pd.DataFrame(scaler.fit_transform(X_train_enc, y_train))
    X_test_enc_scaled = pd.DataFrame(scaler.transform(X_test_enc))


    CV_grid = {'C': [1,5,10,50],
                  'gamma': [1,0.5,0.1],
                  'kernel': ['rbf', 'poly']
                  }
    svc = SVC()
    #svc_CV = GridSearchCV(estimator=svc, param_grid=CV_grid, cv=5, verbose=1)
    svc_RANDOM = RandomizedSearchCV(estimator=svc, param_distributions=CV_grid, n_iter=20, cv=3, verbose=1,
                                    random_state=42, n_jobs=-1, refit=True, scoring='accuracy')
    svc_RANDOM.fit(X_train_enc_scaled, y_train)

    print(svc_RANDOM.best_estimator_)
    print(svc_RANDOM.best_params_)
    print(svc_RANDOM.best_score_)

#svc_rndm_srchCV()

def gradboost_rndm_srchCV():

    CAT_encoder = CatBoostEncoder()
    X_train_enc = CAT_encoder.fit_transform(X_train, y_train)
    X_test_enc = CAT_encoder.transform(X_test)

    scaler = MinMaxScaler()
    X_train_enc_scaled = pd.DataFrame(scaler.fit_transform(X_train_enc, y_train))
    X_test_enc_scaled = pd.DataFrame(scaler.transform(X_test_enc))

    CV_grid = {
        "n_estimators": [5, 50, 250, 500],
        "max_depth": [1, 3, 5, 7, 9],
        "learning_rate": [0.01, 0.1, 1, 10, 100]
    }

    GradBoost = GradientBoostingClassifier()
    #GradBoost_CV = GridSearchCV(estimator=GradBoost, param_grid=CV_grid, cv=5, verbose=1)
    GradBoost_RANDOM = RandomizedSearchCV(estimator=GradBoost, param_distributions=CV_grid, n_iter=75, cv=3, verbose=1,
                                    random_state=42, n_jobs=-1, refit=True, scoring='accuracy')
    GradBoost_RANDOM.fit(X_train_enc_scaled, y_train)

    print(GradBoost_RANDOM.best_estimator_)
    print(GradBoost_RANDOM.best_params_)
    print(GradBoost_RANDOM.best_score_)

#gradboost_rndm_srchCV()


def rfc_CV_grid():

    WOE_encoder = WOEEncoder()
    X_train_enc = WOE_encoder.fit_transform(X_train, y_train)
    X_test_enc = WOE_encoder.transform(X_test)

    scaler = MinMaxScaler()
    X_train_enc_scaled = pd.DataFrame(scaler.fit_transform(X_train_enc, y_train))
    X_test_enc_scaled = pd.DataFrame(scaler.transform(X_test_enc))

    CV_grid = {'n_estimators': [1000],
                'max_features': ['auto'],
                'max_depth': [175],
                'min_samples_split': [9],
                'min_samples_leaf': [1],
                'bootstrap': [True],
                'class_weight':['balanced']
                }


    RFC = RandomForestClassifier()
    RFC_CV = GridSearchCV(RFC, param_grid=CV_grid, cv=2)
    RRC_CV = RFC_CV.fit(X_train_enc_scaled, y_train)

    best_RFC_est = RRC_CV.best_estimator_
    best_RFC_est.fit(X_train_enc_scaled, y_train)

    RFC_pred = best_RFC_est.predict(X_test_enc_scaled)
    print('CV RFC accuracy Score:  ',accuracy_score(y_test, RFC_pred))
    print('CV RFC F1 Score:  ', f1_score(y_test, RFC_pred))

    print(RFC_CV.best_estimator_)
    print(RFC_CV.best_params_)

    print(confusion_matrix(y_test, RFC_pred))
    print(classification_report(y_test, RFC_pred))


#rfc_CV_grid()
def rfc_best_est():

    import time

    WOE_encoder = WOEEncoder()
    X_train_enc = WOE_encoder.fit_transform(X_train, y_train)
    X_test_enc = WOE_encoder.transform(X_test)

    scaler = MinMaxScaler()
    X_train_enc_scaled = pd.DataFrame(scaler.fit_transform(X_train_enc, y_train))
    X_test_enc_scaled = pd.DataFrame(scaler.transform(X_test_enc))

    RF_pipeline = Pipeline(steps=[('target_encoder',TargetEncoder()),
                                  ('scaler', MinMaxScaler()),
                                  ('RF_clf', RandomForestClassifier(
                                             n_estimators=100,
                                             min_samples_split=9,
                                             min_samples_leaf=1,
                                             max_features='auto',
                                             max_depth=175,
                                             class_weight='balanced',
                                             bootstrap=True,
                                             random_state=123))])

    start0 = time.time()
    RF_pipeline.fit(X_train_enc_scaled, y_train)
    stop0 = time.time()
    print(f"Train time: {stop0 - start0}s")
    print(X_train.info())
    importance = RF_pipeline.steps[2][1].feature_importances_
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))
    # plot feature importance
    plt.bar([x for x in range(len(importance))], importance)
    plt.show()

    start1 = time.time()
    RF_pred = RF_pipeline.predict(X_test_enc_scaled)
    stop1 = time.time()
    print(f"Predict time: {stop1 - start1}s")
    print('RF Accuracy :', accuracy_score(y_test, RF_pred))
    print('RF F1 :', f1_score(y_test, RF_pred))
    print(confusion_matrix(y_test, RF_pred))
    print(classification_report(y_test, RF_pred))

#rfc_best_est()

def stack_ensemble():

    WOE_encoder = WOEEncoder()
    X_train_enc = WOE_encoder.fit_transform(X_train, y_train)
    X_test_enc = WOE_encoder.transform(X_test)

    scaler = MinMaxScaler()
    X_train_enc_scaled = pd.DataFrame(scaler.fit_transform(X_train_enc, y_train))
    X_test_enc_scaled = pd.DataFrame(scaler.transform(X_test_enc))

    clfs = list()
    clfs.append(('lr', LogisticRegression()))
    clfs.append(('knn', KNeighborsClassifier()))
    clfs.append(('svm', SVC()))
    clfs.append(('bayes', GaussianNB()))
    # define meta learner model
    meta_clf = LogisticRegression()
    # define the stacking ensemble
    stk_model = StackingClassifier(estimators=clfs, final_estimator = meta_clf, cv=5)
    # fit the model on all available data
    stk_model.fit(X_train_enc, y_train)

    stk_pred = model.predict(X_test_enc_scaled)
    print('Stack Accuracy :', accuracy_score(y_test, stk_pred))
    print('stack F1 :', f1_score(y_test, stk_pred))
    print(confusion_matrix(y_test, stk_pred))
    print(classification_report(y_test, stk_pred))

#stack_ensemble()