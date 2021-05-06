import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from category_encoders import *
import category_encoders as ce

pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


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

hcpcs_cs_cat = pd.read_csv('C:/Users/amcgrat/Desktop/UCD PROGRAM/Project/HPCPS/HCPCS_CODES_ALL_1.csv')

#results_df = results_df.astype({'psps_submitted_charge_amt': np.float})
#results_df = results_df.astype({'psps_denied_services_cnt': np.float})

#bins = [-1, 200, 400, 600, 800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,3200,3400,3600,3800,4000,4200,4400,4600,4800,5000,10000,20000,50000,100000,500000,1000000]
#labels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]

#results_df['binned_billed_amt'] = pd.cut(results_df['psps_submitted_charge_amt'], bins = bins,labels = labels)

#print(results_df['binned_billed_amt'].value_counts())


#bins=[]
#for i in range(1,1072510,2):
    #bins.append(i)

#create sample dataframe
results_df_sample =  results_df.sample(frac = 0.10,random_state=123)
print(results_df_sample.shape)
results_df_sample.to_csv('C:/Users/amcgrat/Desktop/UCD PROGRAM/Project/HPCPS/HCPCS_CODES_ALL_SAMPLE_function_test.csv',index=False)

#join dataframe to hcpcs code categories
results_df_sample=pd.merge(results_df_sample, hcpcs_cs_cat, on=['hcpcs_cd'], how='left')



#replace null values in denied service count column
results_df_sample['psps_denied_services_cnt'].replace(np.NaN,'0',inplace=True)

#change datatypes for submitted amount and denied service count columns
cols = ['psps_denied_services_cnt', 'psps_submitted_charge_amt','psps_submitted_service_cnt']
results_df_sample[cols] = results_df_sample[cols].apply(pd.to_numeric, errors='coerce', axis=1)

results_df_sample['chg_per_svc'] = results_df_sample['psps_submitted_charge_amt']/results_df_sample['psps_submitted_service_cnt']

#create denied column and convert to int.  This will function as the traget/label feature
results_df_sample['denied'] = results_df_sample['psps_denied_services_cnt']>0
results_df_sample['denied'] = results_df_sample["denied"].astype(int)

print(results_df_sample.info())

df_denied_chg = results_df_sample[results_df_sample['denied'] ==1]
df_accepted_chg = results_df_sample[results_df_sample['denied'] !=1]

#sns.kdeplot(data=df_denied_chg, x="psps_submitted_charge_amt",fill=True)
#sns.kdeplot(data=df_denied_chg, x="psps_submitted_charge_amt",fill=True)
sns.kdeplot(data=results_df_sample, x="psps_submitted_charge_amt",hue='denied',log_scale=True,fill=True,bw_adjust=.75)

#plt.xlabel("CHARGE PER SERVCE")
#plt.title("CHARGE PER SERVCE")
#plt.legend(['DENIED/ACCEPTED'])
plt.show()