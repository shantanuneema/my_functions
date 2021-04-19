
import pandas as pd
import numpy as np

def outlier_treatment_on_dataframe(df, 
                                   feature,
                                   data_treatment="winsorize",                                  
                                   consider_lbound=True,                                  
                                   only_positive=True):    
    
    '''
    function to replace/drop outliers by either winsorizing or removing    
        - based on Q1/3 -/+ 1.5(IQR)    
        - returns can take only positive values if only_positive = True        
        - returns 0 for negative values if data_treatment = "winsorize"        
        - drop negative values if data_treatment == "remove_outliers"    
    '''
    
    assert feature in df.columns, "Key Error: {}".feature
    assert isinstance(data_treatment, str), "incorrect data type, data_treatment must be a string"
    assert {data_treatment}.issubset(["winsorize", "remove_outliers", None]), "incorrect outlier treatment"
    
    q_1, q_3 = np.percentile(df[feature], [25, 75])
    ubound = q_3 + (q_3-q_1)*1.5    
    lbound = q_1 - (q_3-q_1)*1.5
    
    if data_treatment == "winsorize":  
        df.loc[df[feature] >= ubound, feature] = ubound        
        if consider_lbound:
            df.loc[df[feature] <= lbound, feature] = lbound
        else:            
            if only_positive:
                df.loc[df[feature] <= lbound, feature] = 0.
    
    elif data_treatment == "remove_outliers":
        df = df[df[feature] <= ubound]
        if only_positive:
            df = df[df[feature] >= 0] if only_positive else df[df[feature] >= lbound]
    
    else: print("Outliers present in {}".format(feature))
    
    return df


def get_key(my_dict, val):
    
    '''
    To get key of a dictionary by providing a value    
    '''
    
    for key, value in my_dict.items():
        if val == value:
            return key
        else:
            try:
                f = interp1d(list(my_dict.values()), list(my_dict.keys()))
                return f(val)
            except:
                return np.mean([key for key, value in my_dict.items()])
            
            
def get_high_corrs(df, 
                   col_list, 
                   min_corr=0.90,
                   max_corr=1.0,
                   include_max=False):
    
    '''
    To get highly correlated features:
    - applicable to pandas dataframe only
    - result in correlation between a given range of values (max/min)
    - option to include max correlation (default=1.0)
    - results in pandas series 
    '''
                   
    c = df[col_list].corr().abs()
    s = c.unstack()
    so = s.sort_values(kind="quicksort", ascending=False)
    
    if include_max:
        answer = so[(so<=max_corr) & (so>=min_corr)]
    else:
        answer = so[(so<max_corr) & (so>=min_corr)]
    
    return answer
                   
                   
# Read parquet file from a folder

import os
import pyarrow.parquet as pq

def parquets_to_df(parquet_folder, col_names):
    
    parquet_files = []
    files = os.listdir(parquet_folder)
    for file in files:
        if ".parquet" in file:
            table = pq.read_table(parquet_folder + file, columns = col_names).to_pandas()
            parquet_files.append(table)
    df = pd.concat(parquet_files).reset_index(drop = True)
    
    return df
                   
                   
def return_quasi_const_categories(df, thresh = 0.998):
  
    '''
    to remove quasi and constant features in large amount of feature space (univariate)
    '''

    quasi_cat_const = []

    for fea in df.columns:
        predominant_cat = (df[fea].value_counts()/len(df)).values[0]
        if predominant_cat > thresh: quasi_cat_const.append(fea)
            
    return quasi_cat_const
                   
def get_duplicated_value_pairs(df):
  
    '''
    To return a dictionary with duplicated features by data values
    - keys of the returning feature dictionary can be selected as features and values may be dropped for further analysis
    '''
    
    duplicated_fea_dict = {}
    duplicated_features = []
    
    for i in range(len(df.columns)):
        fea_1 = df.columns[i]
        if fea_1 not in duplicated_features:
            duplicated_fea_dict[fea_1] = []
            for fea_2 in df.columns[i + 1: ]:
                if df[fea_1].equals(df[fea_2]):
                    duplicated_fea_dict[fea_1].append(fea_2)
                    duplicated_features.append(fea_2)
               
    return duplicated_fea_dict
                  
def correlation(df, thresh = 0.8):
  
    '''
    extract a set of correlated features in given dataframe (including y variable)
    - Take only numerical features
    '''
    
    corr_features = set()
    corr_matrix = df.corr()
      
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > thresh:
                 col = corr_matrix.columns[i]
                 corr_features.add(col)
                   
    return corr_features
