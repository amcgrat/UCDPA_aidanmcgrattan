""" File to test log distribution"""
import pandas as pd
import scipy.stats as sp
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as  sns
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder


DME_1 = pd.read_csv('C:/Users/amcgrat/Desktop/UCD PROGRAM/Project/2018_Physician_Supplier_Procedure_Summary.csv')
DME_1 = DME_1[DME_1['PSPS_SUBMITTED_CHARGE_AMT']< 2500000 ]
DME_1 = DME_1[DME_1['PSPS_SUBMITTED_CHARGE_AMT'] > 5]
DME_1 = DME_1.sample(frac = 0.005)
DME_DENIED = DME_1[DME_1['PSPS_DENIED_SERVICES_CNT'].notnull()]
DME_ACCEPTED = DME_1[DME_1['PSPS_DENIED_SERVICES_CNT'].isnull()]
DME_DENIED_ASC = DME_DENIED.sort_values(by='PSPS_SUBMITTED_CHARGE_AMT')
DME_ACCEPTED_ASC = DME_ACCEPTED.sort_values(by='PSPS_SUBMITTED_CHARGE_AMT')

DME_1['LOGNORM'] = np.log(DME_1['PSPS_SUBMITTED_CHARGE_AMT'])
DME_DENIED['DENIED LOG']= np.log(DME_DENIED['PSPS_SUBMITTED_CHARGE_AMT'])
DME_ACCEPTED['ACCEPTED LOG']= np.log(DME_ACCEPTED['PSPS_SUBMITTED_CHARGE_AMT'])

x = pd.DataFrame(DME_1['LOGNORM'])
x1 = pd.DataFrame(DME_DENIED['DENIED LOG'])
x2 = pd.DataFrame(DME_ACCEPTED['ACCEPTED LOG'])

#print(len(x))
#print(x.describe())
#print(len(x1))
#print(x1.describe())
#print(len(x2))
#print(x2.describe())

#sns.set_style('darkgrid')

#sns.distplot(x,hist=False,)
#sns.distplot(x1,hist=False)
#sns.distplot(x2,hist=False)

#sns.histplot(x,x='LOGNORM',kde=True,fill=True)
#sns.histplot(x1,x='DENIED LOG',kde=True,fill=True)
#sns.histplot(x2,x='ACCEPTED LOG',kde=True,fill=True)


#sns.kdeplot(data=x, x="LOGNORM")
sns.kdeplot(data=x1, x="DENIED LOG",fill=True)
sns.kdeplot(data=x2, x="ACCEPTED LOG",fill=True)


plt.xlabel("SUBMITTED CHARGES (LOG)")
plt.title("SUBMITED CHARGE DENSITY (LOGARITHMIC)")
plt.legend(['DENIED','ACCEPTED'])
plt.show()



DME_1['DENIED'] = DME_1['PSPS_DENIED_SERVICES_CNT']>0
DME_1['DENIED'] = DME_1["DENIED"].astype(int)


encoder = ce.TargetEncoder(cols='HCPCS_CD')
DME_1['HCPCS_CD_ENCODED'] = encoder.fit_transform(DME_1['HCPCS_CD'],DME_1['DENIED'])

DME_2 = DME_1[['HCPCS_CD','HCPCS_CD_ENCODED','DENIED']]
print (DME_2.head(20))


encoder = ce.TargetEncoder(cols='PRICING_LOCALITY_CD')
DME_1['PRICING_LOCALITY_CD_ENCODED'] = encoder.fit_transform(DME_1['PRICING_LOCALITY_CD'],DME_1['DENIED'])
DME_3 = DME_1[['PRICING_LOCALITY_CD','PRICING_LOCALITY_CD_ENCODED','DENIED']]
print (DME_3.head(20))

encoder = ce.TargetEncoder(cols='PROVIDER_SPEC_CD')
DME_1['PROVIDER_SPEC_CD_ENCODED'] = encoder.fit_transform(DME_1['PROVIDER_SPEC_CD'],DME_1['DENIED'])
DME_4 = DME_1[['PROVIDER_SPEC_CD','PROVIDER_SPEC_CD_ENCODED','DENIED']]
print (DME_4.head(20))

print(DME_1.info())