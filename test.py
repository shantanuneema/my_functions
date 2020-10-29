

def extract_target_paths(df, rt_model, 
                         target_feature, 
                         target_thresh, 
                         path_above_thresh=True):

    '''
    Funtion to extract decision paths from a decision tree sklearn model
    - for continuous target (regression models only)
    - default for higher than a given target feature's threshold
    - can be used fto extract paths in reverse direction as well by changing path_above_thresh to "False"
    
    Arguments:
        df - pandas dataframe with both dependent (y) and independent variable (X)
        rt_model - decision tree model using features in X, for target y
        target_feature - name of the column (e.g "value", "target" etc.)
        target_thresh - threshold for target (y); to find all paths above this value
        path_above_thresh - "True" to find paths leading to values above target_thresh, and 
                            "False" for values below target_thresh
    '''
    
    # Filter the dataframe based on the target threshold
    if path_above_thresh:
        df_thresh = df[df[target_feature] >= target_thresh]
    else: df_thresh = df[df[target_feature] <= target_thresh]
    
    # Divide dataframe in X, and y
    X_reduced = df_thresh.drop(target_feature, axis=1)
    y_reduced = df_thresh[target_feature]
    
    node_indicators = rt_model.decision_path(X_reduced)
    
    string_data = tree.export_graphviz(rt_model, out_file=None,
                                       feature_names=X_reduced.columns,
                                       class_names=y_reduced.name)
 
    list_out = string_data.split(";")
    node_out = [item.strip().split("\\") for item in list_out if "label" in item and "nsamples" in item]
    
    children_left = rt_model.tree_.children_left
    children_right = rt_model.tree_.children_right
    
    last_nodes = [i for i in range(len(node_out)) if len(node_out[i])==3]
    last_node_targets = [float("".join(filter(lambda d: str.isdigit(d) or d == '.', node_out[i][2]))) \
                         for i in last_nodes]
    
    if path_above_thresh:
        target_thresh_ind = [i for i in range(len(last_nodes)) if last_node_targets[i] >= target_thresh]
    else:
        target_thresh_ind = [i for i in range(len(last_nodes)) if last_node_targets[i] <= target_thresh]
    
    target_nodes = [last_nodes[i] for i in target_thresh_ind]
    
    decision_paths = []
    for i in range(len(node_indicators.indptr)-1):
        str_ind = node_indicators.indptr[i]
        end_ind = node_indicators.indptr[i+1]
        decision_paths.append(tuple(node_indicators.indices[str_ind:end_ind]))
        
    target_paths = []
    for path in list(set(decision_paths)):
        path_size = len(path)
        if path[path_size-1] in target_nodes:
            target_paths.append(path)
            
    extracted_paths = []
    for k in range(len(target_paths)):
        path_size = len(target_paths[k])
        temp_path = ["target -> "+node_out[target_paths[k][path_size-1]][2].replace("\"]", "").replace(" ","").split("=")[1]] 
        signs = ["<=" if i in children_left else ">" for i in target_paths[k][1:]]
        decisions = [node_out[i][0].split("label=")[1].replace("\"", "") for i in target_paths[k][:-1]]
        for x, decision in enumerate(decisions):
            temp_path.append(decision if signs[x] == "<=" else decision.replace("<=", ">"))
        extracted_paths.append(temp_path)
        
    return True