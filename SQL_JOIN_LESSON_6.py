import pandas as pd

DME_1 = pd.read_csv('C:/Users/amcgrat/Desktop/UCD PROGRAM/Project/DME_1.csv')
LOC_CODE = pd.read_csv('C:/Users/amcgrat/Desktop/UCD PROGRAM/Project/sc-0304.csv')

DME_ACCEPTED = DME_1[DME_1['PSPS_DENIED_SERVICES_CNT'].isnull()]
DME_DENIED = DME_1[DME_1['PSPS_DENIED_SERVICES_CNT'].notnull()]

ACC_DEN = pd.concat([DME_ACCEPTED, DME_DENIED])

ACC_DEN_JOIN = DME_ACCEPTED.join(DME_DENIED, lsuffix='HCPCS_CD', rsuffix='HCPCS_CD')

print(LOC_CODE.info())
print(DME_1.info())

DME_LOC_JOIN = DME_1.join(LOC_CODE, lsuffix='HPRICING_LOCALITY_CD', rsuffix='Social_Security_State_county_code')

print(DME_LOC_JOIN.info())


