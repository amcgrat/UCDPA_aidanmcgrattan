import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import scipy

warnings.simplefilter(action='ignore', category=FutureWarning)


def display_settings():
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

display_settings()

#import csv files.
results_df_sample = pd.read_csv('C:/Users/amcgrat/Desktop/UCD PROGRAM/Project/HPCPS/HCPCS_CODES_ALL_SAMPLE_10.csv')
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

results_df_sample['sub_chg_log']= np.log(results_df_sample['psps_submitted_charge_amt'])
results_df_sample['sub_svc_log']= np.log(results_df_sample['psps_submitted_service_cnt'])

results_df_sample = results_df_sample.astype({'psps_denied_services_cnt': np.int32,
                                              'psps_submitted_charge_amt': np.int32,
                                              'psps_submitted_service_cnt':np.int32})


#calculate charge per service and create new column
results_df_sample['chg_per_svc'] = results_df_sample['psps_submitted_charge_amt']/results_df_sample['psps_submitted_service_cnt']

#bins = [-1, 200, 400, 600, 800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,3200,3400,3600,3800,4000,4200,4400,4600,4800,5000,10000,20000,50000,100000,500000,1000000]
#labels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]

results_df_sample['sub_chg_amt_binn'] = pd.qcut(results_df_sample['psps_submitted_charge_amt'],q = 400,labels=False,duplicates='drop')
results_df_sample['sub_svc_cnt_binn'] = pd.qcut(results_df_sample['psps_submitted_service_cnt'],q = 150,labels=False,duplicates='drop')
#results_df_sample['chg_per_svc_binn'] = pd.cut(results_df_sample['chg_per_svc'],q = 10000,labels=False,duplicates='drop')

#create denied column and convert to int.  This will function as the target/label feature
results_df_sample['accepted'] = results_df_sample['psps_denied_services_cnt']<1
results_df_sample['accepted'] = results_df_sample["accepted"].astype(int)



#for col in ['hcpcs_cd',
            #'carrier_num',
            #'pricing_locality_cd',
            #'type_of_service_cd',
            #'place_of_service_cd',
            #'provider_spec_cd',
            #'psps_hcpcs_asc_ind_cd',
            #'hcpcs_betos_cd',
            #'hcpcs_initial_modifier_cd',
            #'hcpcs_second_modifier_cd',
            #'cd_categories',
            #'sub_chg_amt_binn',
            #'sub_svc_cnt_binn',
            #'chg_per_svc_binn']:
    #results_df_sample[col] = results_df_sample[col].astype('category')

X = results_df_sample
y = results_df_sample['accepted']

df_accepted = results_df_sample[results_df_sample['accepted'] ==1]
df_denied = results_df_sample[results_df_sample['accepted'] !=1]
df_accepted =  df_accepted.sample(frac = 0.50,random_state=123)
print(df_denied.shape)
print(df_accepted.shape)
results_df_sample = df_denied.append(df_accepted)
print(results_df_sample.shape)

#results_df_sample.drop(['psps_submitted_charge_amt',
                        #'psps_submitted_service_cnt',
                        #'chg_per_svc',
                        #'denied'
                        #'psps_denied_services_cnt'], inplace=True,axis = 1)

#results_df_sample.drop(['sub_chg_amt_binn',
                        #'sub_svc_cnt_binn',
                        #'chg_per_svc_binn'], inplace=True,axis = 1)

#from category_encoders import *
#CBE_encoder = CatBoostEncoder()
#X_train_enc = CBE_encoder.fit_transform(results_df_sample, y)


def cat_df(num,name):
    code_top_n_denied = df_denied[name].value_counts(sort=True).nlargest(num)
    code_top_n_accepted = df_accepted[name].value_counts(sort=True).nlargest(num)
    code_freq_acc = pd.DataFrame()
    code_freq_den = pd.DataFrame()
    code_freq_acc['code']= code_top_n_accepted.index
    code_freq_acc['accepted_freq']= code_top_n_accepted.values
    code_freq_den['code']= code_top_n_denied.index
    code_freq_den['denied_freq']= code_top_n_denied.values
    code_top_n = pd.merge(code_freq_acc, code_freq_den, on=['code'], how='outer')
    return code_top_n

#top_code_hpcps = cat_df(200,'hcpcs_cd').dropna()
#top_code_tos = cat_df(25,'type_of_service_cd').dropna()
#top_code_pos = cat_df(42,'place_of_service_cd').dropna()
#top_code_pos = cat_df(42,'place_of_service_cd').dropna()
#top_code_spec_cd = cat_df(150,'provider_spec_cd').dropna()
#top_code_mod_1 = cat_df(100,'hcpcs_initial_modifier_cd')
#top_code_mod_1 = top_code_mod_1[top_code_mod_1['code']!= 'NA']


code_prov_spec = cat_df(200,'provider_spec_cd')
print(code_prov_spec.head(20))


#print (pd.merge(df1, df2, left_on='id', right_on='id1', how='left').drop('id1', axis=1))
from pylab import *
fig = plt.figure()
plt.suptitle('Accepted and Denied distributions')

ax = sns.kdeplot(data=results_df_sample, x="psps_submitted_service_cnt",hue='accepted',log_scale=True,bw_adjust=.75)
subplot(3,4,1)
ax = sns.kdeplot(data=results_df_sample, x="sub_chg_log",hue='accepted',log_scale=True,bw_adjust=.75)
subplot(3,4,2)
ax = sns.kdeplot(data=results_df_sample, x="sub_svc_log",hue='accepted',log_scale=True,bw_adjust=.75)
plt.xlabel("hcpcs code")
subplot(3,4,3)
ax=sns.distplot(df_denied["sub_chg_amt_binn"],hist=False,color='red',label=True)
ax=sns.distplot(df_accepted["sub_chg_amt_binn"],hist=False,color='blue')
subplot(3,4,4)
ax=sns.distplot(df_denied["sub_svc_cnt_binn"],hist=False,color='red')
ax=sns.distplot(df_accepted["sub_svc_cnt_binn"],hist=False,color ='blue' )
subplot(3,4,5)
sns.distplot(np.log(df_denied["psps_submitted_charge_amt"]),hist=False,color='red')
sns.distplot(np.log(df_accepted["psps_submitted_charge_amt"]),hist=False,color ='blue' )
subplot(3,4,6)
sns.lineplot(data=top_code_hpcps, x="code",hue = 'denied_freq', y="accepted_freq")
plt.xticks([], [])
plt.yticks([], [])
plt.xlabel("codes")
subplot(3,4,7)
sns.lineplot(data = top_code_hpcps,x = 'code',y = 'accepted_freq',color='red')
sns.lineplot(data = top_code_hpcps,x = 'code',y = 'denied_freq',color = 'blue')
plt.xticks([], [])
plt.yticks([], [])
plt.xlabel("codes")
subplot(3,4,8)
sns.lineplot(data = top_code_tos,x = 'code',y = 'accepted_freq',color='red')
sns.lineplot(data = top_code_tos,x = 'code',y = 'denied_freq',color = 'blue')
plt.xticks([], [])
plt.yticks([], [])
plt.xlabel("codes")
subplot(3,4,9)
sns.lineplot(data = top_code_pos,x = 'code',y = 'accepted_freq',color='red')
sns.lineplot(data = top_code_pos,x = 'code',y = 'denied_freq',color = 'blue')
plt.xticks([], [])
plt.yticks([], [])
plt.xlabel("codes")
subplot(3,4,10)
sns.lineplot(data = top_code_spec_cd,x = 'code',y = 'accepted_freq',color='red')
sns.lineplot(data = top_code_spec_cd,x = 'code',y = 'denied_freq',color = 'blue')
plt.xticks([], [])
plt.yticks([], [])
plt.xlabel("codes")

plt.show()


#plt.ylabel("")
#plt.xlabel("place_of_service_cd")
#plt.xticks([], [])
#plt.yticks([], [])
















