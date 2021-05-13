import numpy as np
import pandas as pd
from category_encoders import *
import category_encoders as ce
import matplotlib.pyplot as plt
import seaborn as sns

def set_display():
    pd.set_option('display.max_rows', 20)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

set_display()

def api_connection():
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
                     limit = 3400000)
    results_df = pd.DataFrame.from_records(results)

    client.download_attachments("efgi-jnkv", download_dir="~/Desktop")
    client.close()
    return results_df

results_df = api_connection()

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

results_sample = create_sample(results_df,0.2)

#remove nulls
def remove_nulls():
    #remove null values
    results_sample['psps_denied_services_cnt'].replace(np.NaN,'0',inplace=True)
    results_sample['hcpcs_initial_modifier_cd'].replace(np.NaN,'NA',inplace=True)
    results_sample['hcpcs_second_modifier_cd'].replace(np.NaN,'NA',inplace=True)
    results_sample['hcpcs_betos_cd'].replace(np.NaN,'NA',inplace=True)
    results_sample['psps_hcpcs_asc_ind_cd'].replace(np.NaN,'NA',inplace=True)
    results_sample['pricing_locality_cd'].replace(np.NaN,'NA',inplace=True)

remove_nulls()

def change_data_types(df):
    #change datatypes
    results_chg = results_sample.astype({'psps_denied_services_cnt': np.float64,
                                         'psps_submitted_charge_amt': np.float64,
                                         'psps_submitted_service_cnt': np.float64})
    return results_chg

results_sample = change_data_types(results_sample)

#prepend_string to column values
def pre_pend():
    results_sample_prep = results_sample['carrier_num'] = 'cn_' + results_sample['carrier_num'].astype(str)
    results_sample_prep = results_sample['place_of_service_cd'] = 'pos_' + results_sample['place_of_service_cd'].astype(str)

pre_pend()

#add new columns
def add_new_cols(df):
    #create calculated field
    results_sample['chg_per_svc'] = results_sample['psps_submitted_charge_amt'] / results_sample['psps_submitted_service_cnt']

    #create bins for continuous features
    results_sample['sub_chg_amt_binn'] = pd.qcut(results_sample['psps_submitted_charge_amt'], q=60, labels=False,duplicates='drop')
    results_sample['sub_svc_cnt_binn'] = pd.qcut(results_sample['psps_submitted_service_cnt'], q=60,labels=False, duplicates='drop')
    results_sample['chg_per_svc_binn'] = pd.qcut(results_sample['chg_per_svc'], q=60, labels=False,duplicates='drop')

    # create denied column and convert to int.  This will function as the traget/label feature
    results_sample['accepted'] = results_sample['psps_denied_services_cnt'] < 1
    results_sample['accepted'] = results_sample["accepted"].astype(int)
    return results_sample

results_sample = add_new_cols(results_sample)

def creat_den_accpt(df):
    df_denied = results_sample[results_sample['accepted'] ==1]
    df_accepted = results_sample[results_sample['accepted'] !=1]
    return df_denied,df_accepted

def creat_kde_df(df):
    kde_df_table = results_sample[['psps_submitted_charge_amt','psps_submitted_service_cnt','chg_per_svc','accepted']]
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


