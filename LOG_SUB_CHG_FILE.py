""" File to test log distribution"""
import pandas as pd
import scipy.stats as sp
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as  sns

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

x = DME_1['LOGNORM']
x1 = DME_DENIED['DENIED LOG']
x2 = DME_ACCEPTED['ACCEPTED LOG']

sns.set_style('darkgrid')

#sns.distplot(x)
sns.distplot(x1,hist=False)
sns.distplot(x2,hist=False)
plt.xlabel = ('LOG CHARGES')
plt.legend(['DENIED','ACCEPTED'])
plt.show()