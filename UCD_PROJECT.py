from sodapy import Socrata
import pandas as pd

client = Socrata("data.cms.gov", None)
results = client.get('efgi-jnkv',
                     select ='hcpcs_cd,psps_denied_services_cnt',
                     where = "psps_denied_services_cnt >30 "
                             "AND (hcpcs_cd='99213' OR hcpcs_cd='99214')",
                     limit = 15000000)
client.close()
results_df = pd.DataFrame.from_records(results)
print(results_df.info())
print(results_df.describe())
print(results_df.head(20))

