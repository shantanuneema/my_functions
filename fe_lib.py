
import pandas as pd
import numpy as np

def outlier_treatment_on_dataframe(df, 
                                   feature,
                                   data_treatment = "winsorize",                                  
                                   consider_lbound = False,                                  
                                   only_positive = True, 
                                   iqr_percentile_range = [25, 75]):    
    
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
    
    if any(df[feature].isnull()):
        print(f'nulls present in column: {feature}')
        q_1, q_3 = np.percentile(df[feature].dropna(), iqr_percentile_range)
    else:
        q_1, q_3 = np.percentile(df[feature], iqr_percentile_range)  

    ubound = q_3 + (q_3 - q_1) * 1.5    
    lbound = q_1 - (q_3 - q_1) * 1.5

    if data_treatment == "winsorize":  
        df.loc[df[feature] >= ubound, feature] = ubound        
        if only_positive:
            df.loc[df[feature] <= lbound, feature] = 0.
        else: 
            df.loc[df[feature] <= lbound, feature] = lbound 

    elif data_treatment == "remove_outliers":
        df = df[df[feature] <= ubound]
        if only_positive:
            df = df[df[feature] >= 0] if only_positive else df[df[feature] >= lbound]
        if consider_lbound:
            df.loc[df[feature] <= lbound, feature] = lbound

    else: print(f'Outliers exist in column: {feature}')
    
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

def get_corrleated_df_groups(X_train):
    
    """
    Function to determine a list of dataframes with correlated feature groups
    """
    
    corr_matrix = X_train.corr()
    corr_matrix = corr_matrix.abs().unstack()
    corr_matrix = corr_matrix[(corr_matrix >= 0.8) & (corr_matrix < 1)]
    df_cormat = pd.DataFrame(corr_matrix).reset_index(drop = False)
    df_cormat.columns = ["Feature1", "Feature2", "abs(R)"]
    
    grouped_features = []
    correlated_grps = []
    
    for feature in df_cormat["Feature1"].unique():
        if feature not in grouped_features: 
            df_corr = df_cormat[df_cormat["Feature1"] == feature]
            grouped_features = grouped_features + list(df_corr["Feature2"].unique()) + [feature]

            correlated_grps.append(df_corr)
    
    return correlated_grps
  
# Function to build monthly iterator with first day of the month and number of days in the same month
def month_iterator(from_date, read_date):
    # generator to run monthly calendarized dates between any given dates
    start = 12 * from_date.year + from_date.month - 1
    end = 12 * read_date.year + read_date.month
    for year_month in range(start, end):
        year, month = divmod(year_month, 12)
        yield dt.datetime(year, month + 1, 1).date(),\
              calendar.monthrange(year, month + 1)[1]
        
# Example use
for key, value in month_iterator(pd.to_datetime('2015-01-01'), pd.to_datetime('2017-02-01')):
    print(key, value)
    
# Function to reduce cardinality in categorical features
def cardinality_reduction(df, categorical_cols, min_thresh = 0.03, combined_cat_name = 'Other'):
    binned_level_dict = {}
    for col in categorical_cols:
        value_series = df[col].value_counts() / len(df) < min_thresh
        levels_to_bin = list(value_series[value_series].index)
        df[col] = df[col].replace(dict(zip(levels_to_bin, len(levels_to_bin) * [combined_cat_name])))
        binned_level_dict[col] = pd.Series(levels_to_bin)
    return df, binned_level_dict
  
def calculate_vif_num(X, thresh):
    X = X.assign(const=1)  # add constant as expected
    cols = X.columns
    variables = np.arange(X.shape[1])
    dropped = True
    while dropped:
        dropped = False
        c = X[cols[variables]].values
        vif = [variance_inflation_factor(c, ix) for ix in np.arange(c.shape[1])]
        vif = vif[:-1]  # all but the last element (constant removal)
        maxloc = vif.index(max(vif))

        if max(vif) > thresh:
            # print('dropping \'' + X[cols[variables]].columns[maxloc] + '\' with VIF: ' + str(round(max(vif),1)))
            variables = np.delete(variables, maxloc)
            dropped = True

    variables = np.delete(variables, len(variables) - 1)
    # print('Remaining variables:')
    # print(X.columns[variables])
    return X[cols[variables]]
