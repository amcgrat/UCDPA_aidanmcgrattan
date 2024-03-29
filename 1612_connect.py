import pyodbc
import pandas as pd
import warnings

def set_display():
    '''
    Sets Display Options including warnings
    Parameters:
        N/A
    Returns:
        N/A
    '''
    pd.set_option('display.max_rows', 20)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    #Suppress 'FutureWarning' messages
    warnings.simplefilter(action='ignore', category=FutureWarning)

set_display()

def dbsep_connect():
    c = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=DBSEP1612;DATABASE =UHN_Reporting; Trusted_Connection=yes;')
    return c

conn = dbsep_connect()

def run_sql(conn):
    SQL_Query = pd.read_sql_query('''SELECT DISTINCT MPIN_LOCATION.TaxID, 
    PROVIDER.ProvStatus 
    FROM MPIN_LOCATION 
    LEFT JOIN PROVIDER ON MPIN_LOCATION.MPIN = PROVIDER.MPIN 
    GROUP BY MPIN_LOCATION.TaxID, PROVIDER.ProvStatus''', conn)
    df = pd.DataFrame(SQL_Query)
    return df

df = run_sql(conn)

cross_tb = pd.crosstab(df.TaxID, df.ProvStatus)
par_only = cross_tb[(cross_tb['P']==1) & (cross_tb['N']==0)]
non_par = cross_tb[(cross_tb['P']==0) & (cross_tb['N']==1)]
both_par = cross_tb[(cross_tb['P']==1)]
print(par_only.head(25))
print(non_par.head(25))
print(both_par.head(25))
#par.to_csv('C:/Users/amcgrat/Desktop/par.csv')







