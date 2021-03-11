import pandas as pd

DME_1 = pd.read_csv('C:/Users/amcgrat/Desktop/UCD PROGRAM/Project/DME_1.csv')
LOC_CODE = pd.read_csv('C:/Users/amcgrat/Desktop/UCD PROGRAM/Project/sc-0304.csv')

DME_ACCEPTED = DME_1[DME_1['PSPS_DENIED_SERVICES_CNT'].isnull()]
DME_DENIED = DME_1[DME_1['PSPS_DENIED_SERVICES_CNT'].notnull()]

ACC_DEN = pd.concat([DME_ACCEPTED, DME_DENIED])

ACC_DEN_JOIN = DME_ACCEPTED.join(DME_DENIED, lsuffix='HCPCS_CD', rsuffix='HCPCS_CD')

DME_LOC_JOIN = DME_1.join(LOC_CODE, lsuffix='HPRICING_LOCALITY_CD', rsuffix='Social_Security_State_county_code')

print(DME_LOC_JOIN.info())

ACC_DEN_MERGE = pd.merge(DME_ACCEPTED, DME_DENIED [['HCPCS_CD','TYPE_OF_SERVICE_CD','PLACE_OF_SERVICE_CD','PSPS_NCH_PAYMENT_AMT']],on = "HCPCS_CD",how = "right")

print(ACC_DEN_MERGE.info())