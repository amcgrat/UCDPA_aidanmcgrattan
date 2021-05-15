import numpy as np
import pandas as pd
from category_encoders import *
import category_encoders as ce
import matplotlib.pyplot as plt
import seaborn as sns
from category_encoders import *
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def set_display():
    pd.set_option('display.max_rows', 20)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

set_display()

def api_connection(n):
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
                             'psps_hcpcs_asc_ind_cd,'
                             'hcpcs_betos_cd',
                     where = 'psps_submitted_charge_amt > 5',
                     limit = n)
    results_df = pd.DataFrame.from_records(results)

    client.download_attachments("efgi-jnkv", download_dir="~/Desktop")
    client.close()
    return results_df

results_df = api_connection(1500000)

#create sample dataframe
def create_sample(df,n):
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

results_sample = create_sample(results_df,0.1)

#remove nulls
def remove_nulls(df):
    #remove null values
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
def add_new_cols(df):
    #create calculated field
    results_sample['chg_per_svc'] = results_sample['psps_submitted_charge_amt'] / results_sample['psps_submitted_service_cnt']

    #create bins for continuous features
    results_sample['sub_chg_amt_binn'] = pd.qcut(results_sample['psps_submitted_charge_amt'], q=60, labels=False,duplicates='drop')
    results_sample['sub_svc_cnt_binn'] = pd.qcut(results_sample['psps_submitted_service_cnt'], q=60,labels=False, duplicates='drop')
    results_sample['chg_per_svc_binn'] = pd.qcut(results_sample['chg_per_svc'], q=60, labels=False,duplicates='drop')

    #create denied column and convert to int.  This will function as the traget/label feature
    results_sample['accepted'] = results_sample['psps_denied_services_cnt'] < 1
    results_sample['accepted'] = results_sample["accepted"].astype(int)
    return results_sample

results_sample = add_new_cols(results_sample)

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
    cd_categories = cat_df('cd_categories', 25)
    sub_chg_amt_binn = cat_df('sub_chg_amt_binn', 25)
    sub_svc_cnt_binn = cat_df('sub_svc_cnt_binn', 25)
    chg_per_svc_binn = cat_df('chg_per_svc_binn', 25)

    fig, axes = plt.subplots(4, 4)

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
    sns.lineplot(data = sub_chg_amt_binn, x='sub_chg_amt_binn', y='accepted', hue='class', sort= True,ax=axes[2, 2], legend=False)
    sns.lineplot(data = sub_svc_cnt_binn, x='sub_svc_cnt_binn', y='accepted', hue='class', sort= True,ax=axes[2, 3], legend=False)
    sns.lineplot(data = chg_per_svc_binn, x='chg_per_svc_binn', y='accepted', hue='class',sort= True, ax=axes[3, 0], legend=False)

    kde_df_table = creat_kde_df(results_sample)
    palette = {'accepted': 'dodgerblue','denied': 'orangered'}

    sns.kdeplot(data=kde_df_table, x="psps_submitted_charge_amt", hue='accepted', log_scale=True, fill=True,bw_adjust=.75, ax=axes[3,1],legend=False,palette=palette)
    sns.kdeplot(data=kde_df_table, x="psps_submitted_service_cnt", hue='accepted', log_scale=True, fill=True,bw_adjust=.75, ax=axes[3,2], legend=False,palette=palette)
    sns.kdeplot(data=kde_df_table, x="chg_per_svc", hue='accepted', log_scale=True, fill=True,bw_adjust=.75, ax=axes[3,3], legend=False,palette=palette)

    for i in range (0,4):
        for j in range(0,4):

            axes[i,j].set_ylabel("value frequency")
            axes[i,j].set_xticklabels("")
            axes[i,j].set_yticklabels("")
            axes[i,j].set_yticks([])
            axes[i,j].set_xticks([])
            axes[i,j].set_facecolor("aliceblue")

    axes[3,3].set_ylabel("density")
    axes[3,2].set_ylabel("density")
    axes[3,1].set_ylabel("density")
    plt.suptitle("Accepted/Denied Frequencies",fontsize =16)

    fig.legend(title='', bbox_to_anchor=(0.1,0.875),  labels=['Accepted', 'Denied'])

    plt.show()

show_plts()

def create_target(df):
    y = results_sample['accepted']
    return y

def del_cols():
    results_sample.drop(['psps_submitted_charge_amt',
                            'psps_submitted_service_cnt',
                            'chg_per_svc',
                            'accepted',
                            'psps_denied_services_cnt'], inplace=True, axis=1)

def split_data(df):
    from sklearn.model_selection import train_test_split
    y = create_target(results_sample)
    del_cols()

    X_train, X_test, y_train, y_test = train_test_split(results_sample, y, stratify=y,test_size=0.20, random_state=123)

    return X_train,X_test,y_train, y_test

def get_classifiers():
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.svm import SVC, LinearSVC, NuSVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier

    classifiers = [('LogReg', LogisticRegression()),
                   ('LinSVC', LinearSVC(random_state=123)),
                   ('KNN', KNeighborsClassifier()),
                   ('SGD', SGDClassifier(random_state=123)),
                   ('Gauss', GaussianNB()),
                   ('SVC', SVC(random_state=123)),
                   ('RFC', RandomForestClassifier(random_state=123)),
                   ('ADABoost', AdaBoostClassifier()),
                   ('GradBoost', GradientBoostingClassifier())]
    return classifiers

get_classifiers()

def get_encoders():
    encoders = [('WOEEncoder', WOEEncoder()),
                ('TargetEncoder', TargetEncoder()),
                ('CatBoostEncoder', CatBoostEncoder())]
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

    plt.show()

def evaluate_models():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

    X_train,X_test,y_train, y_test = split_data(results_sample)

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

    zippedList =  list(zip(enc_name, clf_name, acc_score,F1_score))
    eval_results = pd.DataFrame(zippedList,columns = ['encoder' , 'classifier', 'accuracy_score','F1_score'])
    eval_results_top = eval_results.iloc[eval_results.groupby(['classifier']).apply(lambda x: x['accuracy_score'].idxmax())]
    plot_clfs(eval_results)
    print(eval_results_top.sort_values(by = 'accuracy_score',ascending=False))

evaluate_models()


