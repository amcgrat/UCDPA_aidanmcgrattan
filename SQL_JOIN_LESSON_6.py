import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


DME_1 = pd.read_csv('C:/Users/amcgrat/Desktop/UCD PROGRAM/Project/2018_Physician_Supplier_Procedure_Summary.csv')
LOC_CODE = pd.read_csv('C:/Users/amcgrat/Desktop/UCD PROGRAM/Project/sc-0304.csv')

bins = [0,50,75,100,500,1000,2500,5000,7500,
        10000,15000,25000,50000,
        100000,250000,500000,1000000,2500000,5000000,10000000,25000000,60000000]

DME_DENIED = DME_1[DME_1['PSPS_DENIED_SERVICES_CNT'].notnull()]
DME_DENIED = DME_DENIED[DME_DENIED['PSPS_SUBMITTED_CHARGE_AMT'] > 5]
DME_ACCEPTED = DME_1[DME_1['PSPS_DENIED_SERVICES_CNT'].isnull()]
DME_ACCEPTED = DME_ACCEPTED[DME_ACCEPTED['PSPS_SUBMITTED_CHARGE_AMT'] > 5]
DME_ALL = DME_1[DME_1['PSPS_SUBMITTED_CHARGE_AMT'] > 5 ]

#step = (DME_DENIED['PSPS_SUBMITTED_CHARGE_AMT'].max() - DME_DENIED['PSPS_SUBMITTED_CHARGE_AMT'].min())/21
#print(DME_DENIED['PSPS_SUBMITTED_CHARGE_AMT'].max())
#print(DME_DENIED['PSPS_SUBMITTED_CHARGE_AMT'].min())
#print(step)
#bins = np.arange(5,DME_DENIED['PSPS_SUBMITTED_CHARGE_AMT'].max()  ,step = step)
#print (len(bins))
labels = np.arange(1,22)

#print (len(labels))
DME_DENIED['DENIED_BINNED_SUBMITTED_AMT'] = pd.cut(DME_DENIED['PSPS_SUBMITTED_CHARGE_AMT'],bins = bins,labels = labels)
DME_ACCEPTED['DENIED_BINNED_SUBMITTED_AMT'] = pd.cut(DME_ACCEPTED['PSPS_SUBMITTED_CHARGE_AMT'],bins = bins,labels = labels)
DME_ALL['DENIED_BINNED_SUBMITTED_AMT'] = pd.cut(DME_ALL['PSPS_SUBMITTED_CHARGE_AMT'],bins = bins,labels = labels)
#DME_DENIED['DENIED_BINNED_SUBMITTED_AMT'] = pd.qcut(DME_DENIED['PSPS_SUBMITTED_CHARGE_AMT'],q=60,labels = range(1,61))

#print(DME_DENIED[['DENIED_BINNED_SUBMITTED_AMT','PSPS_DENIED_SERVICES_CNT','PSPS_SUBMITTED_CHARGE_AMT']])
#SAMPLE_DME_DENIED = DME_DENIED.sample(frac=0.3)

#print(DME_DENIED[['DENIED_BINNED_SUBMITTED_AMT','PSPS_DENIED_SERVICES_CNT','PSPS_SUBMITTED_CHARGE_AMT']])
#print(DME_DENIED['DENIED_BINNED_SUBMITTED_AMT'].value_counts(ascending = False))

print(DME_DENIED['PSPS_SUBMITTED_CHARGE_AMT'].max())
print(len(DME_DENIED))
print(DME_ACCEPTED['PSPS_SUBMITTED_CHARGE_AMT'].max())
print ( len(DME_ACCEPTED))
print(DME_ALL['PSPS_SUBMITTED_CHARGE_AMT'].max())
print(len(DME_ALL))

DME_DENIED_COUNTS = DME_DENIED['DENIED_BINNED_SUBMITTED_AMT'].value_counts(ascending = False)
print(DME_DENIED_COUNTS.head(25))
DME_ACCEPTED_COUNTS = DME_ACCEPTED['DENIED_BINNED_SUBMITTED_AMT'].value_counts(ascending = False)
print(DME_ACCEPTED_COUNTS.head(25))
DME_ALL_COUNTS = DME_ALL['DENIED_BINNED_SUBMITTED_AMT'].value_counts(ascending = False)
print(DME_ALL_COUNTS.head(25))

DATA_DENIED = DME_DENIED['DENIED_BINNED_SUBMITTED_AMT']
DATA_ACCEPTED = DME_ACCEPTED['DENIED_BINNED_SUBMITTED_AMT']
DATA_ALL = DME_ALL['DENIED_BINNED_SUBMITTED_AMT']

SORTED_DATA_DENIED = np.sort(DATA_DENIED)  # Or data.sort(), if data can be modified
SORTED_DATA_ACCEPTED = np.sort(DATA_ACCEPTED)
SORTED_DATA_ALL = np.sort(DATA_ALL)

#plt.step(sorted_data[::-1], np.arange(sorted_data.size))  # From the number of data points-1 to 0  This reverses the curve below.

plt.plot(SORTED_DATA_DENIED, np.arange(SORTED_DATA_DENIED.size))  # From 0 to the number of data points-1
plt.plot(SORTED_DATA_ACCEPTED, np.arange(SORTED_DATA_ACCEPTED.size))# From 0 to the number of data points-1
plt.plot(SORTED_DATA_ALL, np.arange(SORTED_DATA_ALL.size))# From 0 to the number of data points-1

plt.xlabel('submitted charge category')
plt.ylabel('number of claims')

plt.legend(["Denied", "Accepted",'All'])


plt.show()



