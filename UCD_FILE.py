import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sodapy import Socrata

client = Socrata("data.cms.gov", None)
results = client.get('efgi-jnkv',
                     select ='psps_submitted_charge_amt,hcpcs_cd,psps_denied_services_cnt',
                     where = 'psps_denied_services_cnt is not null AND psps_submitted_charge_amt > 5',
                     limit = 15000000)

results_df = pd.DataFrame.from_records(results)
client.close()
results_df = results_df.astype({'psps_submitted_charge_amt': np.float})
#results_df = results_df.astype({'psps_denied_services_cnt': np.float})

#bins = [-1, 200, 400, 600, 800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,3200,3400,3600,3800,4000,4200,4400,4600,4800,5000,10000,20000,50000,100000,500000,1000000]
#labels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]

#results_df['binned_billed_amt'] = pd.cut(results_df['psps_submitted_charge_amt'], bins = bins,labels = labels)

#print(results_df['binned_billed_amt'].value_counts())


bins=[]
for i in range(1,1072510,2):
    bins.append(i)



print(results_df.info())
print(results_df.describe())
#print(results_df.head(50))
