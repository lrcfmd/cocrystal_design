from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

def feature_engineering():
    dataset1 = pd.read_csv('../data/coformers1.csv')
    dataset2 = pd.read_csv('../data/coformers2.csv')

    # Remove benzene and toluene solvents for having a more balanced dataset when selecting the features
    out1 = np.append(dataset2.Identifier[dataset2.NAME == 'Cc1ccccc1'].values, dataset2.Identifier[dataset2.NAME == 'c1ccccc1'].values, axis=0)
    out=np.append(out1, dataset2.Identifier[dataset2.NAME == 'Cc1ccc(C)cc1'].values , axis=0)
    dataset1= dataset1[~dataset1['Identifier'].isin(out)]
    dataset2 = dataset2[~dataset2['Identifier'].isin(out)]
    # The two datasets are concateanted for identifing the highly correlated descriptors and remove them
    df1_w=dataset1.iloc[:,2:]
    df2_w=dataset2.iloc[:,2:]
    data = pd.concat([df1_w, df2_w])
    data_ = data.drop_duplicates(keep='first')

    # Drop the highly linearly correlated features among the datasets
    # Create correlation matrix
    corr_matrix1 = data_.corr().abs()

    # Select upper triangle of correlation matrix
    upper1 = corr_matrix1.where(np.triu(np.ones(corr_matrix1.shape), k=1).astype(np.bool))

    # Find index of feature columns with Pearson correlation greater than 0.92
    to_drop1 = [column for column in upper1.columns if any(upper1[column] > 0.92)]

    # Drop the descriptos will low variance, below 0.4
    drop = data_.std()[data_.std() < 0.4].index.values
    to_drop2= [x for x in drop if x not in to_drop1]
    drop_final= to_drop1 + to_drop2

    # Remove the selected features from the datasets
    df1=df1_w.drop(columns=drop_final)
    df1=df1.fillna(df1.mean())
    df2=df2_w.drop(columns=drop_final)
    df2=df2.fillna(df2.mean())
    # Calculate the Spearman correlations of the datasets
    cor = df1.corrwith(df2, axis=0, drop=False, method='spearman').abs()
    corr=cor.sort_values(ascending=False)
    # Construct a vector w which is used to keep only the descriptors that are correlated higher than 0.30 using Spearman correlation
    # In this vector, 1 is on the positions of the descriptors that have correlation coeficcient > 0.3,  and 0 otherwise
    w = np.array(cor)
    np.nan_to_num(w,0)
    w[w<0.4] =0
    w[w=='NaN']=0
    w[w>=0.4] =1
    # Multiply the two datasets with the vector w, such that the descriptors with lower correlation will become zero and removed 
    df1 = df1*w
    df2 = df2*w
    df1_1 = df1.loc[:, (df1 != 0).any(axis=0)]
    df2_2 = df2.loc[:, (df2 != 0).any(axis=0)]
    # Calculate the Pearson correlations of the datasets 
    cort= df1_1.corrwith(df2_2, axis=0, drop=False, method='pearson').abs()
    corrt=cort.sort_values(ascending=False)
    # Keep only descriptors with Pearson correlations > 0.2
    w = np.array(cort)
    np.nan_to_num(w,0)
    w[w<0.4]=0
    w[w>=0.4]=1
    df1_1 = df1_1*w
    df2_2 = df2_2*w
    df1_1 = df1_1.loc[:, (df1_1 != 0).any(axis=0)]
    df2_2 = df2_2.loc[:, (df2_2 != 0).any(axis=0)]

    # Import the whole dataset and keep only the selected descriptors from the feature engineering part
    dataset1 = pd.read_csv('/content/drive/My Drive/cocrystal_design-master/data/coformers1.csv')
    dataset2 = pd.read_csv('/content/drive/My Drive/cocrystal_design-master/data/coformers2.csv')
    df1 = dataset1.iloc[:, 2:]  
    df2 = dataset2.iloc[:, 2:] 
    df1 = dataset1[df1_1.columns.values[:]]
    df2 = dataset2[df1_1.columns.values[:]]

    # Standarize the dataset
    X_scaler = MinMaxScaler()   
    df_concat = pd.concat([df1, df2])
    df_concat = df_concat.drop_duplicates(keep='first')
    numerical_cols = df_concat.columns[:]
    df_scaled = pd.DataFrame(X_scaler.fit(df_concat), columns=numerical_cols, index=df_concat.index)
    numerical_cols = df2.columns[:]
    df1_scaled =  pd.DataFrame(X_scaler.transform(df1[numerical_cols]), columns=numerical_cols, index=df1.index)
    df2_scaled = pd.DataFrame(X_scaler.transform(df2[numerical_cols]), columns=numerical_cols, index=df2.index)
    df =pd.concat([pd.concat([dataset1['Identifier'], df1_scaled, df2_scaled], axis=1),
     pd.concat([dataset1['Identifier'].apply(lambda x: f"{x}_"),df2_scaled, df1_scaled], axis=1) ], axis=0)
    return df