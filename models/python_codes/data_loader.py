import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

def dict1():
  '''Dictionary assinign to each combination of the unlabelled dataset
      the smiles of the molecular pairs
  '''
  dictionary = pd.read_csv('../data/dictionary.csv')
  k = dictionary.comb
  v= dictionary.iloc[:, 1:3].values
  dict1= {key:value for key, value in zip(k, v)}
  return dict1


def zinc_dict():
  '''
  Dictionary where each smiles is associated with its ZINC Identifier
  '''
  zinc_smiles=pd.read_csv('../data/zinc_smiles.csv')
  k=zinc_smiles['smiles'].values
  v= zinc_smiles.Identifier.str.strip()
  zinc_dict= {key:value for key, value in zip(k, v)}
  return zinc_dict


def labelled_dataset():
  '''
  Loading the labelled dataset
  '''
  dataset1=pd.read_csv('../data/coformers1.csv') 
  dataset2=pd.read_csv('../data/coformers2.csv')
  df1=dataset1.iloc[:,2:]
  df1 = df1.fillna(df1.mean())
  df2=dataset2.iloc[:,2:]
  df2 = df2.fillna(df2.mean())
  df1=df1.dropna(axis=1)
  df2=df2[df1.columns.values]
  df_concat = pd.concat([df1, df2])
  df_concat = df_concat.drop_duplicates(keep='first')
  numerical_cols = df_concat.columns[:]
  X_scaler = MinMaxScaler()
  df_scaled = pd.DataFrame(X_scaler.fit(df_concat), columns=numerical_cols, index=df_concat.index)
  numerical_cols = df2.columns[:]
  df1_scaled =  pd.DataFrame(X_scaler.transform(df1[numerical_cols]), columns=numerical_cols, index=df1.index)
  df2_scaled = pd.DataFrame(X_scaler.transform(df2[numerical_cols]), columns=numerical_cols, index=df2.index)
  # Final bidirectional concatenated dataset, after feature selection and scaling 
  df = concat_bidirectional(df1_scaled,df2_scaled)
  return df, dataset1, dataset2, X_scaler


def concat_bidirectional(dataset11, dataset22):
  '''
  Concatenate the labelled dataset in a bidirectional way
  '''
  dataset1=pd.read_csv('../data/coformers1.csv')
  return pd.concat([pd.concat([dataset1['Identifier'], dataset11, dataset22], axis=1), 
   pd.concat([dataset1['Identifier'].apply(lambda x: f"{x}_"),dataset22, dataset11], axis=1) ])


def unlabelled_dataset():
  '''
  Load the descriptors for ZINC15 dataset
  '''
  unlabeled = pd.read_csv('../data/unlabelled_dragon_git.csv')
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
  keys = unlabeled['NAME'].values
  values = unlabeled.iloc[:, 1:].values
  d = {key:value for key, value in zip(keys, values)}
  mol1_data= list()

  unlabeled, pairs = load_zinc_dataset()
  mol1_data= list()
  for mol1 in pairs[0]:       
      mol1_data.append(d[mol1])
  mol1_data = pd.DataFrame(mol1_data, columns = unlabeled.iloc[:, 1:].columns.values)   
  mol2_data= list()
  mol2_data= list()
  for mol2 in pairs[1]:   
      mol2_data.append(d[mol2])
  mol2_data = pd.DataFrame(mol2_data, columns= unlabeled.iloc[:, 1:].columns.values) 
  final_1 = pd.concat([pairs[0],mol1_data],axis=1)
  final_1 = final_1.fillna(0)#df1.mean())
  final_2 = pd.concat([pairs[1],mol2_data],axis=1)
  final_2 = final_2.fillna(0)
  unlab=pd.concat([pairs[0], pairs[1]], axis=1)
  final_1 = final_1.replace({'#NUM!': 0})
  final_2 = final_2.replace({'#NUM!': 0})
  _,dataset1,_, X_scaler = labelled_dataset()
  final_11=final_1[dataset1.iloc[:,2:].columns.values]
  final_22=final_2[dataset1.iloc[:,2:].columns.values]
  final_1_scaled = pd.DataFrame(X_scaler.transform(final_11))
  final_2_scaled = pd.DataFrame(X_scaler.transform(final_22))
  uf=pd.concat([final_1_scaled, final_2_scaled], axis =1)
  uf1=pd.concat([final_1, final_2], axis =1)
  comb=[]
  for i in range(1,final_1.shape[0]+1):
    comb.append('comb%s' % i)
  uf_final=pd.concat([pd.DataFrame(comb, columns=['comb']),final_1, final_2 ], axis=1)
  return uf_final


