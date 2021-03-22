import pandas as pd
import scipy.stats as sp
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as  sns


DME_1 = pd.read_csv('C:/Users/amcgrat/Desktop/UCD PROGRAM/Project/2018_Physician_Supplier_Procedure_Summary.csv')

#DME_DENIED = DME_1[DME_1['PSPS_DENIED_SERVICES_CNT'].isnull()]
#DME_DENIED = DME_DENIED[DME_DENIED['PSPS_SUBMITTED_CHARGE_AMT'].notnull()]
DME_1 = DME_1[DME_1['PSPS_SUBMITTED_CHARGE_AMT'] < 1500000]
DME_1 = DME_1[DME_1['PSPS_SUBMITTED_CHARGE_AMT'] > 5]

DME_1 = DME_1.sample(frac = 0.005)

DME_DENIED = DME_1[DME_1['PSPS_DENIED_SERVICES_CNT'].notnull()]

DME_ACCEPTED = DME_1[DME_1['PSPS_DENIED_SERVICES_CNT'].isnull()]

DME_DENIED_ASC = DME_DENIED.sort_values(by='PSPS_SUBMITTED_CHARGE_AMT')
DME_ACCEPTED_ASC = DME_ACCEPTED.sort_values(by='PSPS_SUBMITTED_CHARGE_AMT')

SAMPLE_EXP = pd.DataFrame(sp.expon.rvs(scale = 200000,loc=0,size=len(DME_DENIED)))


print(DME_DENIED['PSPS_SUBMITTED_CHARGE_AMT'].describe())
#print(DME_ACCEPTED['PSPS_SUBMITTED_CHARGE_AMT'].describe())
print(SAMPLE_EXP.describe())

def ecdf(data):
    """ Compute ECDF """
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n+1) / n
    return(x,y)

#_ = plt.hist(DME_DENIED_ASC['PSPS_SUBMITTED_CHARGE_AMT'])
#_ = plt.hist(DME_ACCEPTED_ASC['PSPS_SUBMITTED_CHARGE_AMT'])
#_ = plt.xlabel('Submitted Charges')
#_ = plt.ylabel('COUNT')


#sns.distplot(DME_1['PSPS_SUBMITTED_CHARGE_AMT'],bins = 40,norm_hist=True)
#sns.distplot(DME_DENIED_ASC['PSPS_SUBMITTED_CHARGE_AMT'],bins = 40,norm_hist=True)
#sns.distplot(DME_ACCEPTED_ASC['PSPS_SUBMITTED_CHARGE_AMT'],bins = 40,norm_hist=True)
#sns.distplot(SAMPLE_EXP,bins = 40,norm_hist=True)
#plt.legend(['DENIED','THEORETICAL'])
#plt.grid(True)
#plt.show()

def _check_normality(dist):
    '''
    Method to check if the samples in dist differs from a normal distribution.
    Return true if the dist is likely to be gaussian.
    '''
    _, pvalue = sp.normaltest(dist)
    if (pvalue > 0.05):
        # The samples in dist came from a normal distribution with 95% confidence pvalue >0.005
        return True
    else:
        return False


print(stats.shapiro(DME_1['PSPS_SUBMITTED_CHARGE_AMT']))
print(stats.shapiro(DME_ACCEPTED['PSPS_SUBMITTED_CHARGE_AMT']))
print(stats.shapiro(DME_DENIED['PSPS_SUBMITTED_CHARGE_AMT']))
print(stats.shapiro(SAMPLE_EXP))

