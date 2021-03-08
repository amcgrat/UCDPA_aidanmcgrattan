import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sodapy import Socrata


client = Socrata("data.cms.gov", None)
results = client.get('efgi-jnkv',
                     select ='psps_submitted_charge_amt,hcpcs_cd,psps_denied_services_cnt',
                     where = "psps_denied_services_cnt>0",
                     limit = 1500000)
client.close()
results_df = pd.DataFrame.from_records(results)

results_df = results_df.astype({'psps_submitted_charge_amt': np.float})

results_df['binned_billed_amt'] = pd.qcut(results_df['psps_submitted_charge_amt'], q=25, precision=0,duplicates = "drop")

print(results_df.info())
print(results_df.head(25))