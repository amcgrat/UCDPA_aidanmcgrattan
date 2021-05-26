import numpy as np
from numpy import mean,std
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
from sklearn.model_selection import train_test_split,cross_validate,GridSearchCV,RandomizedSearchCV,RepeatedStratifiedKFold,cross_val_score
from sklearn.pipeline import Pipeline
from time import time
from mlxtend.plotting import plot_confusion_matrix
import re
import warnings

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
    Connect to API and download datafile and attachments
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

    results_sample = pd.DataFrame.from_records(results)
    #download attachments
    client.download_attachments("efgi-jnkv", download_dir='C:/Users/amcgrat/Desktop/UCD PROGRAM/Project/HPCPS')
    client.close()
    return results_sample

def create_sample(df,n):
    '''
    Create random sample of initial downloaded dataframe
    Parameters:
        n(int): a decimal integer between 0 and 1
        DataFrame: Pandas DataFrame
    Returns:
        DataFrame: Random sample of dataframe
    '''
    results_sample =  df.sample(frac = n,random_state=123)
    return results_sample

def get_API_data():
    '''
    Runs API and random_sample sunctions
    Parameters:
        N/A
    Returns:
        DataFrame: Pandas Dataframe with downloaded
        and randomly sampled data from API
    '''
    #downloads data to dataframe from function call
    results_sample = api_connection(3400)
    #creates random sample from function call
    results_sample = create_sample(results_sample,0.2)
    return results_sample

#results_sample = get_API_data()

def import_data_file():
    '''
    Import datafile
    Parameters:
        None: N/A
    Returns:
        DataFrame: Pandas Dataframe with imported data from API
    '''
    #Modify path and filename as necessary
    results_sample = pd.read_csv('C:/Users/amcgrat/Desktop/UCD PROGRAM/Project/HPCPS/CMS_DME_CLM_2.csv')
    return results_sample

results_sample = import_data_file()

def categorical_counts():
    '''
    Create dataframe with counts of categorical values in data sample
    Parameters:
        N/A
    Returns:
        DataFrame: Pandas Dataframe with categorical values count by field
    Outputs:
        attachment: categorical values .csv file
    '''
    uniqueValues = pd.DataFrame(results_sample.nunique())
    uniqueValues['feature names'] = uniqueValues.index
    uniqueValues.drop(['psps_submitted_charge_amt','psps_submitted_service_cnt','psps_denied_services_cnt'],0,inplace=True)
    uniqueValues.reset_index(drop=True, inplace=True)
    uniqueValues.rename(columns = {0:'value counts'}, inplace = True)
    uniqueValues1 = uniqueValues[['feature names','value counts']]
    uniqueValues1.to_csv('C:/Users/amcgrat/Desktop/UCD PROGRAM/Project/HPCPS/GRAPHS_OUPUTS_FINAL/VALUE_COUNTS.csv',index=False)
    return uniqueValues1

unique_counts = categorical_counts()

print(unique_counts)


def merge_df():
    '''
    Merge data file with dataframe created from categorical code dataframe
    Parameters:
        N/A
    Returns:
        DataFrame: Merged dataframe
    Outputs:
        N/A
    '''
    #pad dataframe column to avoid mismatching to .csv code values.
    results_sample['hcpcs_cd']=results_sample.hcpcs_cd.str.pad(5,side='left',fillchar='0')
    # Import cd codes csv file
    hcpcs_cs_cat = pd.read_csv('C:/Users/amcgrat/Desktop/UCD PROGRAM/Project/HPCPS/HCPCS_CODES_ALL_1.csv')
    # join dataframe to hcpcs code categories
    results_df_sample = pd.merge(results_sample, hcpcs_cs_cat, on=['hcpcs_cd'], how='left')
    return results_df_sample

results_sample = merge_df()

def remove_nulls(df):
    '''
    Remove null values from selected columns and replace with values
    Parameters:
        DataFrame: Pandas DataFrame
    Returns:
        DataFrame: Pandas DataFrame
    '''
    df['psps_denied_services_cnt'].replace(np.NaN,'0',inplace=True)
    df['hcpcs_initial_modifier_cd'].replace(np.NaN,'NA',inplace=True)
    df['hcpcs_second_modifier_cd'].replace(np.NaN,'NA',inplace=True)
    df['hcpcs_betos_cd'].replace(np.NaN,'NA',inplace=True)
    df['pricing_locality_cd'].replace(np.NaN,'NA',inplace=True)
    df['cd_categories'].replace(np.NaN, 'NA', inplace=True)

    return df

results_sample=remove_nulls(results_sample)

def change_data_types(df):
    '''
    Change datatypes of selected columns
    Parameters:
        DataFrame: Pandas DataFrame
    Returns:
        DataFrame: Pandas DataFrame
    '''
    #change datatypes

    #to facilitate calculations
    results_chg = results_sample.astype({'psps_denied_services_cnt': np.float64,
                                         'psps_submitted_charge_amt': np.float64,
                                         'psps_submitted_service_cnt': np.float64})
    return results_chg

results_sample = change_data_types(results_sample)

def pre_pend_cd(df):
    '''
    Change datatypes of selected columns
    Parameters:
        DataFrame: Pandas DataFrame
    Returns:
        DataFrame: Pandas DataFrame
        '''
    #prevent data columns being interpreted as numerical
    results_sample['carrier_num'] = 'cn_' + results_sample['carrier_num'].astype(str)
    results_sample['place_of_service_cd'] = 'pos_' + results_sample['place_of_service_cd'].astype(str)

    return results_sample

results_sample = pre_pend_cd(results_sample)

def add_new_col(df,n):
    '''
    Add new columns
    Parameters:
        DataFrame: Pandas DataFrame,n(number of bins)
    Returns:
        DataFrame: Pandas DataFrame
    '''
    #create lower cardinality hcpcs code groups
    results_sample['hcpcs_groups'] = results_sample.hcpcs_cd.apply(lambda x: (x[-1:] + x[:2]) if re.match(r'\d{4}[a-zA-Z]{1}', x) else x[:3])

    #create binned columns for psps_submitted_charge_amt and psps_submitted_service_cnt
    results_sample['sub_chg_amt_binn'] = pd.qcut(results_sample['psps_submitted_charge_amt'], q=n, labels=False,duplicates='drop')
    results_sample['sub_svc_cnt_binn'] = pd.qcut(results_sample['psps_submitted_service_cnt'], q=n, labels=False,duplicates='drop')

    #create denied column and convert to int.  This will function as the traget/label feature
    results_sample['accepted'] = results_sample['psps_denied_services_cnt'] < 1
    results_sample['accepted'] = results_sample["accepted"].astype(int)
    return results_sample

results_sample = add_new_col(results_sample,60)

def plot_sample_class_dist():
    '''
    Plot binary class distribution (accepted/denied)
    Parameters:
        N/A
    Returns:
        N/A
    Outputs:
        sns countplot
    '''
    dist_df =  results_sample[['accepted']]
    dist_df = dist_df.replace({'accepted': {1: 'accepted', 0: 'denied'}})
    sns.countplot(x="accepted", data=dist_df)
    plt.title('Accepted / Denied Distribution')
    plt.xlabel('')
    plt.show()

plot_sample_class_dist()

def creat_den_accpt(df):
    '''
    Create denied and accepted dataframes for kde plot function
    Parameters:
        DataFrame
    Returns:
        N/A
    '''
    df_denied = df[results_sample['accepted'] ==1]
    df_accepted = df[results_sample['accepted'] !=1]
    return df_denied,df_accepted


def creat_kde_df(df):
    '''
    Create kde dtaframe for kde plots
    Parameters:
        DataFrame
    Returns:
        DataFrame
    '''
    kde_df = df[['psps_submitted_charge_amt','psps_submitted_service_cnt','accepted']]
    kde_df = kde_df.replace({'accepted': {1: 'accepted', 0: 'denied'}})
    return kde_df

def cat_df(code,n):
    '''
    Create category value frequency dataframe for each categorical variabe
    Parameters:
        code (feature name)
        n (int: top n codes to include in dataframe)
    Returns:
        DataFrame
    '''
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
    '''
    Create categorical value frequency plots and kde plots for continuous features
    Parameters:
        code (feature name)
    Returns:
        N/A
    Outputs:
        sns lineplots : cateorical features
        kde plots : continuous features
    '''
    #call cat_df function for each ctegorical feature
    hcpcs_cd = cat_df('hcpcs_cd', 25)
    carrier_num = cat_df('carrier_num', 25)
    pricing_locality_cd = cat_df('pricing_locality_cd', 25)
    type_of_service_cd = cat_df('type_of_service_cd', 25)
    place_of_service_cd = cat_df('place_of_service_cd', 25)
    provider_spec_cd = cat_df('provider_spec_cd', 25)
    hcpcs_betos_cd = cat_df('hcpcs_betos_cd', 25)
    hcpcs_initial_modifier_cd = cat_df('hcpcs_initial_modifier_cd', 25)
    hcpcs_second_modifier_cd = cat_df('hcpcs_second_modifier_cd', 25)
    cd_categories = cat_df('cd_categories', 25)

    #create lineplots figure and subplots
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
    sns.lineplot(data = cd_categories,x = 'cd_categories',y = 'accepted',hue = 'class',ax = axes[2,1],legend = False)
    #call create_kde_d function kde subplots.
    kde_df = creat_kde_df(results_sample)
    palette = {'accepted': 'dodgerblue','denied': 'orangered'}

    # create kde subplots
    sns.kdeplot(data=kde_df, x="psps_submitted_charge_amt", hue='accepted', log_scale=True, fill=True,bw_adjust=.75, ax=axes[2,2],legend=False,palette=palette)
    sns.kdeplot(data=kde_df, x="psps_submitted_service_cnt", hue='accepted', log_scale=True, fill=True,bw_adjust=.75, ax=axes[2,3], legend=False,palette=palette)
    #loop to set formats
    for i in range (0,3):
        for j in range(0,4):
            axes[i,j].set_ylabel("value frequency")
            axes[i,j].set_xticklabels("")
            axes[i,j].set_yticklabels("")
            axes[i,j].set_yticks([])
            axes[i,j].set_xticks([])
            axes[i,j].set_facecolor("ivory")

    #Additional formatting for kde plots
    axes[2,2].set_ylabel("density")
    axes[2,3].set_ylabel("density")
    #Figure title
    plt.suptitle("Accepted/Denied Frequencies",fontsize =16)
    # Figure label
    fig.legend(title='', bbox_to_anchor=(0.1,0.875),  labels=['Accepted', 'Denied'])
    plt.show()

show_plts()

def create_target(df):
    '''
    Create taraget feature
    Parameters:
        DataFrame
    Returns:
        pandas series
    '''
    y = results_sample['accepted']

    return y

y = create_target(results_sample)

def del_cols(df):
    '''
    Drop features from dataframe
    Parameters:
        DataFrame
    Returns:
        DataFrame
    '''
    #Drops selected columns and the target column.
    results_sample.drop(['psps_submitted_charge_amt',
                         'psps_submitted_service_cnt',
                         'hcpcs_groups',
                         'cd_categories',
                         'accepted',
                         'psps_denied_services_cnt'], inplace=True, axis=1)
    return results_sample

results_sample = del_cols(results_sample)

def split_data(df,n):
    '''
    Create rain test split
    Parameters:
        DataFrame,n(int between 0-1)
    Returns:
        DataFrame :  X_train
        DataFrame :  X_test
        pandas series :  y_train
        pandas series :  y_test
    '''
    X_train, X_test, y_train, y_test = train_test_split(results_sample, y, stratify=y,test_size=n, random_state=123)
    return  X_train,X_test,y_train, y_test

X_train,X_test,y_train, y_test = split_data(results_sample,.2)

def get_classifiers():
    '''
    Create list of tuples (classifier name and classifier)
    Parameters:
        N/A
    Returns:
        list
    '''
    classifiers = [('LogReg', LogisticRegression(max_iter=1000,n_jobs=-1)),
                   ('LinSVC', LinearSVC(random_state=123)),
                   ('KNN', KNeighborsClassifier(n_jobs=-1)),
                   ('SGD', SGDClassifier(random_state=123,n_jobs=-1)),
                   ('Gauss', GaussianNB()),
                   ('ADABoost', AdaBoostClassifier()),
                   ('GradBoost', GradientBoostingClassifier()),
                   ('RFC', RandomForestClassifier(random_state=123,n_jobs=-1))]
    return classifiers

def get_encoders():
    '''
    Create list of tuples (encoder name and encoder)
    Parameters:
        N/A
    Returns:
        List
    '''
    encoders = [('TargetEncoder', TargetEncoder()),
                ('CatBoostEncoder', CatBoostEncoder()),
                ('WOEEncoder', WOEEncoder())]
    return encoders


def plot_clfs(df):
    '''
    Create lineplots of scores for classifiers and encoders. Called from evaluate_models()
    Parameters:
        dataFrame
    Returns:
        N/A
    Outputs:
        sns lineplots : cateorical features
    '''
    fig, axes = plt.subplots(2, 1)
    df.sort_values(by='accuracy_score',ascending=False,inplace = True)
    sns.lineplot(data=df, x='classifier', y='accuracy_score', hue='encoder',ax = axes[0])
    sns.lineplot(data=df, x='classifier', y='F1_score', hue='encoder',ax = axes[1])
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)
    axes[1].set_facecolor("ivory")
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
    axes[0].set_facecolor("ivory")
    plt.tight_layout()
    plt.show()

def evaluate_models():
    '''
    Evaluate all classifiers and encoder combinations
    Parameters:
        DataFrame
    Returns:
        DataFrame : (top classifier and encoder combinations grouped by classifier)
    '''
    enc_name = []
    clf_name = []
    acc_score = []
    F1_score = []

    print(X_train.info())

    encoders = get_encoders()
    classifiers = get_classifiers()
    for index, encoder in enumerate(encoders):
        for index, classifier in enumerate(classifiers):
            pipe = Pipeline(steps=[('encoder', encoder[1]),
                                   ('scaler', MinMaxScaler()),
                                   ('classifier', classifier[1])])

            pipe.fit(X_train, y_train)
            pipe_pred = pipe.predict(X_test)
            print(encoder[0],classifier[0],accuracy_score(y_test,pipe_pred),f1_score(y_test, pipe_pred))
            enc_name.append(encoder[0])
            clf_name.append(classifier[0])
            acc_score.append(accuracy_score(y_test, pipe_pred))
            F1_score.append(f1_score(y_test, pipe_pred))

    zippedList =  list(zip(enc_name, clf_name, acc_score,F1_score))
    eval_results = pd.DataFrame(zippedList,columns = ['encoder' , 'classifier', 'accuracy_score','F1_score'])
    #print(eval_results)
    eval_results_top = eval_results.iloc[eval_results.groupby(['classifier']).apply(lambda x: x['accuracy_score'].idxmax())]
    #eval_results_top.to_csv('C:/Users/amcgrat/Desktop/UCD PROGRAM/Project/HPCPS/GRAPHS_OUPUTS_FINAL/TOP_CLF_LIST_1.csv', index=False)
    plot_clfs(eval_results)
    eval_results_sorted = eval_results_top.sort_values(by = 'accuracy_score',ascending=False)
    print(eval_results_sorted)

#evaluate_models()

def rfc_CV_grid():
    '''
    Create gridSearchCV for RandomForestClassifier
    Parameters:
        N/A
    Returns:
        N/A
    Outputs:
        confusion_matrix,
        classification_report,
        scoring
    '''

    print('grid')
    WOE_encoder = WOEEncoder()
    X_train_enc = WOE_encoder.fit_transform(X_train, y_train)
    X_test_enc = WOE_encoder.transform(X_test)

    scaler = MinMaxScaler()
    X_train_enc_scaled = pd.DataFrame(scaler.fit_transform(X_train_enc, y_train))
    X_test_enc_scaled = pd.DataFrame(scaler.transform(X_test_enc))

    CV_grid = {'n_estimators': [10,50,100,500,1000,2000],
                'max_features': ['auto', 'sqrt'],
                'max_depth': [50,100,175,200],
                'min_samples_split': [5,7,9,11],
                'min_samples_leaf': [1],
                'bootstrap': [True,False],
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

def knn_CV_grid():
    '''
        Create gridSearchCV for KNeighborsClassifier
        Parameters:
            N/A
        Returns:
            N/A
        Outputs:
            confusion_matrix,
            classification_report,
            scoring
        '''
    TE_encoder = TargetEncoder()
    X_train_enc = TE_encoder.fit_transform(X_train, y_train)
    X_test_enc = TE_encoder.transform(X_test)

    scaler = MinMaxScaler()
    X_train_enc_scaled = pd.DataFrame(scaler.fit_transform(X_train_enc, y_train))
    X_test_enc_scaled = pd.DataFrame(scaler.transform(X_test_enc))

    CV_grid = {'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
                  'weights':['uniform', 'distance'],
                  'metric': ['minkowski','euclidean','manhattan'],
                  'n_neighbors': [5,6,7,8,9]
               }
    KNN = KNeighborsClassifier()
    KNN_CV = GridSearchCV(KNN, param_grid=CV_grid, cv=2)
    KNN_CV = KNN_CV.fit(X_train_enc_scaled, y_train)

    best_KNN_est = KNN_CV.best_estimator_
    best_KNN_est.fit(X_train_enc_scaled, y_train)

    KNN_pred = best_KNN_est.predict(X_test_enc_scaled)
    print('CV KNN accuracy Score:  ', accuracy_score(y_test, KNN_pred))
    print('CV KNN F1 Score:  ', f1_score(y_test, KNN_pred))

    print(KNN_CV.best_estimator_)
    print(KNN_CV.best_params_)

    print(confusion_matrix(y_test, KNN_pred))
    print(classification_report(y_test, KNN_pred))

#knn_CV_grid()

def rfc_best_model():
    '''
    Create best model for RandomForestClassifier
    Parameters:
        N/A
    Returns:
        N/A
    Outputs:
        confusion_matrix,
        classification_report,
        scoring,feature importance plot,
        confusion_matrix plot
    '''
    feature = []
    score = []

    WOE_encoder = WOEEncoder()
    X_train_enc = WOE_encoder.fit_transform(X_train, y_train)
    X_test_enc = WOE_encoder.transform(X_test)

    scaler = MinMaxScaler()
    X_train_enc_scaled = pd.DataFrame(scaler.fit_transform(X_train_enc, y_train))
    X_test_enc_scaled = pd.DataFrame(scaler.transform(X_test_enc))


    model = RandomForestClassifier(
                                   #n_estimators = 2000,
                                   #min_samples_split=9,
                                   #min_samples_leaf=1,
                                   #max_features='auto',
                                   #max_depth=175,
                                   #class_weight='balanced'
                                   #bootstrap=True,
                                   #random_state=123
                                  )

    time1 = time()
    model.fit(X_train_enc_scaled, y_train)
    time2 = time()
    time_taken = (time2-time1)
    print('train_time',time_taken)
    model_predict = model.predict(X_test_enc_scaled)
    print('RFC Accuracy :', accuracy_score(y_test, model_predict))
    print('RFC F1 :', f1_score(y_test, model_predict))
    cm = confusion_matrix(y_test, model_predict)
    print(cm)
    print(classification_report(y_test, model_predict))

    my_list = list(X_test_enc)

    importance = model.feature_importances_
    # summarize feature importance
    for i, v in enumerate(importance):
        feature.append(i)
        score.append(np.abs(v))
        #print('Feature: %0d, Score: %.5f' % (i, v))
    zippedList = list(zip(feature, my_list, score))
    eval_results = pd.DataFrame(zippedList, columns=['feature', 'feature_name', 'score'])
    eval_results_sorted = eval_results.sort_values(by='score', ascending=True)
    #print(eval_results['score'])
    print(eval_results_sorted)

    cc = ['colors'] * len(eval_results['score'])
    for n, val in enumerate(eval_results['score']):
        if val > 0:
            cc[n] = 'blue'
        elif val <= 0:
            cc[n] = 'red'

    plt.barh(eval_results_sorted['feature_name'], eval_results_sorted['score'], color=cc)
    plt.title('RandomForest Feature Importance')
    plt.show()

    #cm = confusion_matrix(y_test, model_predict)

    class_names = ['Denied', 'Accepted']
    fig, ax = plot_confusion_matrix(conf_mat=cm,
                                    colorbar=True,
                                    show_absolute=True,
                                    class_names=class_names)
    plt.title('RandomForest Confusion Matrix')
    plt.show()

rfc_best_model()

def stack_ensemble():
    '''
    Create StackingClassifier model
    Parameters:
        N/A
    Returns:
        N/A
    Outputs:
        confusion_matrix,
        classification_report,
        scoring
    '''

    WOE_encoder = WOEEncoder()
    X_train_enc = WOE_encoder.fit_transform(X_train, y_train)
    X_test_enc = WOE_encoder.transform(X_test)

    scaler = MinMaxScaler()
    X_train_enc_scaled = pd.DataFrame(scaler.fit_transform(X_train_enc, y_train))
    X_test_enc_scaled = pd.DataFrame(scaler.transform(X_test_enc))

    clfs = list()
    clfs.append(('linSVC', LinearSVC()))
    clfs.append(('bayes', GaussianNB()))
    clfs.append(('knn',KNeighborsClassifier()))
    clfs.append(('rfc', RandomForestClassifier()))
    # define meta learner model
    meta_clf = LogisticRegression()
    # define the stacking ensemble
    stk_model = StackingClassifier(estimators=clfs, final_estimator = meta_clf, cv=3)

    # fit the model on training data
    stk_model.fit(X_train_enc_scaled, y_train)
    stk_pred = stk_model.predict(X_test_enc_scaled)
    print('Stack Accuracy :', accuracy_score(y_test, stk_pred))
    print('stack F1 :', f1_score(y_test, stk_pred))
    print(confusion_matrix(y_test, stk_pred))
    print(classification_report(y_test, stk_pred))

#stack_ensemble()

def KNN_clf_opt():
    '''
    Create KNeighborsClassifier optimized model
    Parameters:
        N/A
    Returns:
        N/A
    Outputs:
        confusion_matrix,
        classification_report,
        scoring
    '''

    WOE_encoder = WOEEncoder()
    X_train_enc = WOE_encoder.fit_transform(X_train, y_train)
    X_test_enc = WOE_encoder.transform(X_test)

    scaler = MinMaxScaler()
    X_train_enc_scaled = pd.DataFrame(scaler.fit_transform(X_train_enc, y_train))
    X_test_enc_scaled = pd.DataFrame(scaler.transform(X_test_enc))

    model = KNeighborsClassifier(
                                 algorithm='auto',
                                 weights= 'distance',
                                 metric= 'minkowski',
                                 n_neighbors=9
                                 )


    model.fit(X_train_enc_scaled, y_train)
    model_predict = model.predict(X_test_enc_scaled)
    print('KNN Accuracy :', accuracy_score(y_test, model_predict))
    print('KNN F1 :', f1_score(y_test, model_predict))
    print(confusion_matrix(y_test, model_predict))
    print(classification_report(y_test, model_predict))

#KNN_clf_opt()


