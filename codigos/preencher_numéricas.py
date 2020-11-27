from fancyimpute import KNN
KNN_imputer = KNN()

num_features = ['cod_municipio', 'feature_16', 'feature_17', 'feature_13', 'feature_14', 'feature_15', 'feature_18','feature_04', 'feature_06', 'feature_07', 'feature_09', 'feature_10' ]
df_knn = cenarios.copy()
df_knn = KNN_imputer.fit_transform(df_knn[num_features])
df_knn