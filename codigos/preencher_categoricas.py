import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

df1 = pd.read_csv(r'C:\Users\guilh\OneDrive\Documentos\AI\Desafio Hackaton - Big Data\lojas_atuais.csv')

#Completando a coluna 'feature_01'
df = df1.iloc[:,3:13]
df = df.drop(['feature_08'],axis=1)
df = pd.get_dummies(df,drop_first=True,columns=['feature_02','feature_03','feature_05'])
df = df.replace(to_replace=['FEATURE_01_VALUE_10','FEATURE_01_VALUE_08','FEATURE_01_VALUE_03', 'FEATURE_01_VALUE_06'],value=[1,2,3,4])

sc_X = StandardScaler()
column_names_to_normalize = ['feature_04', 'feature_06', 'feature_07', 'feature_09','feature_10']
x = df[column_names_to_normalize].values
x_scaled = sc_X.fit_transform(x)
df_temp = pd.DataFrame(x_scaled, columns=column_names_to_normalize, index = df.index)
df[column_names_to_normalize] = df_temp

imputer = KNNImputer()
imputer.fit(df)
dfTrans = imputer.transform(df)

colnames = df.columns
dfTrans = pd.DataFrame(dfTrans,columns=colnames)
dfTrans['feature_01'] = dfTrans['feature_01'].round(0)

dfTrans = dfTrans.replace(value=['FEATURE_01_VALUE_10','FEATURE_01_VALUE_08','FEATURE_01_VALUE_03', 'FEATURE_01_VALUE_06'],to_replace=[1,2,3,4])
dfTrans['feature_01'].unique()
df['feature_01'] = dfTrans['feature_01'].values

df1['feature_01'] = df['feature_01'].values

#Completando a coluna 'feature_11
df = df1.iloc[:,4:14]
df = df.drop(['feature_08'],axis=1)
df = pd.get_dummies(df,drop_first=True,columns=['feature_02','feature_03','feature_05'])
df = df.replace(to_replace=['FEATURE_11_VALUE_01','FEATURE_11_VALUE_02','FEATURE_11_VALUE_03','FEATURE_11_VALUE_04','FEATURE_11_VALUE_05','FEATURE_11_VALUE_06','FEATURE_11_VALUE_07'],value=[1,2,3,4,5,6,7])

sc_X = StandardScaler()
column_names_to_normalize = ['feature_04', 'feature_06', 'feature_07', 'feature_09','feature_10']
x = df[column_names_to_normalize].values
x_scaled = sc_X.fit_transform(x)
df_temp = pd.DataFrame(x_scaled, columns=column_names_to_normalize, index = df.index)
df[column_names_to_normalize] = df_temp

imputer = KNNImputer()
imputer.fit(df)
dfTrans = imputer.transform(df)

colnames = df.columns
dfTrans = pd.DataFrame(dfTrans,columns=colnames)
dfTrans['feature_11'] = dfTrans['feature_11'].round(0)

dfTrans = dfTrans.replace(value=['FEATURE_11_VALUE_01','FEATURE_11_VALUE_02','FEATURE_11_VALUE_03','FEATURE_11_VALUE_04','FEATURE_11_VALUE_05','FEATURE_11_VALUE_06','FEATURE_11_VALUE_07'],to_replace=[1,2,3,4,5,6,7])
dfTrans['feature_11'].unique()
df['feature_11'] = dfTrans['feature_11'].values

df1['feature_11'] = df['feature_11'].values

#Completando a coluna 'feature_12'
df = df1.iloc[:,4:15]
df = df.drop(['feature_08','feature_11'],axis=1)
df = pd.get_dummies(df,drop_first=True,columns=['feature_02','feature_03','feature_05'])
df = df.replace(to_replace=['FEATURE_12_VALUE_01','FEATURE_12_VALUE_02'],value=[1,2])

sc_X = StandardScaler()
column_names_to_normalize = ['feature_04', 'feature_06', 'feature_07', 'feature_09','feature_10']
x = df[column_names_to_normalize].values
x_scaled = sc_X.fit_transform(x)
df_temp = pd.DataFrame(x_scaled, columns=column_names_to_normalize, index = df.index)
df[column_names_to_normalize] = df_temp

imputer = KNNImputer()
imputer.fit(df)
dfTrans = imputer.transform(df)

colnames = df.columns
dfTrans = pd.DataFrame(dfTrans,columns=colnames)
dfTrans['feature_12'] = dfTrans['feature_12'].round(0)

dfTrans = dfTrans.replace(value=['FEATURE_12_VALUE_01','FEATURE_12_VALUE_02'],to_replace=[1,2])
dfTrans['feature_12'].unique()
df['feature_12'] = dfTrans['feature_12'].values

df1['feature_12'] = df['feature_12'].values

df1.to_csv(r'C:\Users\guilh\OneDrive\Documentos\AI\Desafio Hackaton - Big Data\categoricas.csv')
