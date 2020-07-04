# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# Import the main libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler


# %%
# Construct the labelled dataset by contatenating the feature set of each coformer in both orders
def concat_bidirectional(dataset11, dataset22):

  return pd.concat([pd.concat([dataset1['Identifier'], dataset11, dataset22], axis=1), pd.concat([dataset1['Identifier'].apply(lambda x: f"{x}_"),dataset22, dataset11], axis=1) ])


# %%
# Standarize the dataset
def standarize_labelled(df1, df2):
  X_scaler = MinMaxScaler()
  df1=dataset1.iloc[:,2:]
  df1 = df1.fillna(0)
  df2=dataset2.iloc[:,2:]
  df2 = df2.fillna(0)
  df1=df1.dropna(axis=1)
  df2=df2[df1.columns.values]
  df_concat = pd.concat([df1, df2])
  df_concat = df_concat.drop_duplicates(keep='first')
  numerical_cols = df_concat.columns[:]
  df_scaled = pd.DataFrame(X_scaler.fit(df_concat), columns=numerical_cols, index=df_concat.index)

  numerical_cols = df2.columns[:]
  df1_scaled =  pd.DataFrame(X_scaler.transform(df1[numerical_cols]), columns=numerical_cols, index=df1.index)
  df2_scaled = pd.DataFrame(X_scaler.transform(df2[numerical_cols]), columns=numerical_cols, index=df2.index)

  # Final bidirectional concatenated dataset, after feature selection and scaling 
  df = concat_bidirectional(df1_scaled,df2_scaled)
  return df


# %%
# Generate the unknown dataset (unlabelled)
# Read the Zinc dataset of purcasable molecules with their dragon descriptors

#unlabeled = pd.read_csv('/content/drive/My Drive/cocrystal_design/data/unlabelled_dragon_descriptors.csv')
def standarize_unlabelled(unlabeled):
  # Take all the possible combinations between these molecules
  val = unlabeled['NAME'].values
  length = len(val)
  unlabeled=unlabeled.loc[:, (unlabeled != 0).any(axis=0)]
  pairs = [[val[i],val[j]] for i in range(length) for j in range(length) if i!=j ]

  # Remove the duplicate structures
  no_dups = []
  for pair in pairs:
    if not any(all(i in p for i in pair) for p in no_dups):
      no_dups.append(pair)

  pairs = pd.DataFrame(no_dups)

  # Assign to each molecule its descriptor vector
  keys = unlabeled['NAME'].values
  values = unlabeled.iloc[:, 1:].values
  d = {key:value for key, value in zip(keys, values)}

  # Construct two datasets of the co-former pairs extracted from ZINC15
  mol1_data= list()
  for mol1 in pairs[0]:       
      mol1_data.append(d[mol1])    
  mol1_data = pd.DataFrame(mol1_data, columns = unlabeled.iloc[:, 1:].columns.values)   
  mol2_data= list()
  for mol2 in pairs[1]:   
      mol2_data.append(d[mol2])
  mol2_data = pd.DataFrame(mol2_data, columns= unlabeled.iloc[:, 1:].columns.values) 

  final_1= mol1_data.iloc[:,1:][dataset1.columns.values[2:]]
  final_2= mol2_data.iloc[:,1:][dataset1.columns.values[2:]]

  # Standarize the unlabeled data based on the labelled
  final_1_scaled = pd.DataFrame(X_scaler.transform(final_1))
  final_2_scaled = pd.DataFrame(X_scaler.transform(final_2))
  uf=pd.concat([final_1_scaled, final_2_scaled], axis =1)

  # Construct the final dataset by adding the combination id and the scaled descroiptors
  uf_final = pd.concat([uf_tot['comb'], pd.DataFrame(uf)], axis=1)
  uf_final = pd.DataFrame(uf_final.values , columns=df.columns.values )
  return  uf_final

