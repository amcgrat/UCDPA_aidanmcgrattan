import pandas as pd
import xlrd as xl

Top_10_Combinations_Test = pd.read_excel('C:/Users/amcgrat/Desktop/UCD PROGRAM/Project/HPCPS/clf_enc_eval_5.xls',sheet_name = ['Top_10_Combinations'])
print(Top_10_Combinations_Test)
clf_Combinations_Test = pd.read_excel('C:/Users/amcgrat/Desktop/UCD PROGRAM/Project/HPCPS/clf_enc_eval_5.xls',sheet_name = ['clf_enc_eval_5'])
print(clf_Combinations_Test)