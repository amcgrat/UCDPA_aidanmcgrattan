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
print(df.head())

