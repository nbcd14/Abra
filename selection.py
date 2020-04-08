import pandas
import random
import scipy as sc
from scipy.special import beta
import numpy as np
import pandas as pd
import functools

def get_correlated_columns(data: pd.DataFrame, correlation_threshold=1):
    list_of_correlated_columns = []
    corr = data.corr()
    
    for col_name in list(corr):
        correlated_columns = sorted(list(corr[col_name][corr[col_name]>=correlation_threshold].index))
        if len(correlated_columns) > 1:
            str_correlated_columns = str(correlated_columns)
            list_of_correlated_columns.append(str_correlated_columns)
            
    return [list(filter(lambda t: t not in ('[', ', ', ']'), x.split("'"))) for x in set(list_of_correlated_columns)]

def get_non_numeric_columns(data: pd.DataFrame, level_threshold=20):
    non_numeric_columns = []
        
    for col_name in list(data):
        if data[col_name].dtype == 'object' or len(data[col_name].unique()) < level_threshold:
            non_numeric_columns.append(col_name)
                
    return non_numeric_columns

def get_unary_columns(data: pd.DataFrame):
    unary_columns = []
    
    for col_name in list(data):
        if len(data[col_name].unique()) == 1:
            unary_columns.append(col_name)
            
    return unary_columns
            
            
def uniform(m):
    return 1

def get_binomial(n, p):
    def binomial(k):
        return (p**k) * ((1 - p)**(n - k))
    return binomial
    
def get_beta_binomial(n, a, b):
    def beta_binomial(k):
        return beta(k+a, n-k+b) / beta(a,b)
    return beta_binomial

def get_fast_ols_inputs(X, y, weights=None):
    if weights is None:
        weights = np.ones(len(X))

    XtX = np.matmul(np.multiply(X.T, weights), X)
    Xty = np.matmul(X.T, np.multiply(weights, y))
    yty = np.matmul(y, np.multiply(weights, y))
    
    return {'XtX':XtX, 'Xty':Xty, 'yty':yty}
    

def fast_ols_beta_hat(XtX, Xty, var_idx=None):
    if var_idx is None:
        var_idx = list(range(len(XtX)))
        
    return np.matmul(np.linalg.pinv(XtX[var_idx,:][:,var_idx]), Xty[var_idx])

def fast_ols(XtX, Xty, yty, var_idx=None):
    if var_idx is None:
        var_idx = list(range(len(XtX)))
        
    beta_hat = fast_ols_beta_hat(XtX, Xty, var_idx)
    rss = yty - np.matmul(Xty[var_idx].T, beta_hat)
    return beta_hat, rss

def ols_log_likelihood(rss, n, p):
    return -0.5 * (n * np.log(2 * np.pi) + n * np.log(rss/n) + n)

def ols_bic(rss, n, p):
    return np.log(n) * p - 2 * ols_log_likelihood(rss, n, p)

def ols_aic(rss, n, p):
    return 2 * p - 2 * ols_log_likelihood(rss, n, p)

def ols_adj_r_squared(rss, tss, n, p):
    return 1- (rss/tss)*(n-1)/(n-p-1)

def remove_from_list(list_of_items, items_to_be_removed):
    return [item for item in list_of_items if item not in items_to_be_removed]

def get_drop_group(col, group_matrix):
    return [drop_col for drop_col in range(len(group_matrix)) if (group_matrix.T[col,:]>0)[drop_col]]

def get_add_group(col, group_matrix):
    return [add_col for add_col in range(len(group_matrix)) if (group_matrix[col,:]>0)[add_col]]
    
#def get_candidate_model_cols(curr_cols, group_matrix):
#    candidate_cols = curr_cols.copy()
#    all_cols = list(range(len(group_matrix)))
#    
#    if np.random.rand() < 0.5:
#        col = random.choice(all_cols)
#        if col in candidate_cols:
#            candidate_cols = remove_from_list(candidate_cols, get_drop_group(col, group_matrix))
#        else: 
#            candidate_cols = candidate_cols + get_add_group(col, group_matrix)      
#    else:
#        non_curr_cols = list(set(all_cols) - set(curr_cols))
#        if len(non_curr_cols) > 0:
#            drop_group = get_drop_group(random.choice(candidate_cols), group_matrix)
#            add_group = get_add_group(random.choice(non_curr_cols), group_matrix)
#            
#            candidate_cols = remove_from_list(candidate_cols, drop_group)
#            candidate_cols = candidate_cols + add_group
#    
#    return candidate_cols


def get_candidate_model_cols(curr_cols, all_cols, group_matrix=None):
    candidate_cols = curr_cols.copy()
    if np.random.rand() < 0.5:
        col = random.choice(all_cols)
        candidate_cols.remove(col) if col in candidate_cols else candidate_cols.append(col)           
    else:
        non_curr_cols = list(set(all_cols) - set(curr_cols))
        if len(non_curr_cols) > 0:
            candidate_cols.remove(random.choice(candidate_cols))
            candidate_cols.append(random.choice(non_curr_cols))
    return candidate_cols

def fast_bms(data: pd.DataFrame, 
             target: pd.Series, 
             method: str ='median', 
             weights=None, 
             iterations: int=100000, 
             burn: int=10000, 
             prior=uniform, 
             group_matrix=None):
    
    fast_ols_inputs = get_fast_ols_inputs(data.values, target.values, weights=None)
    n, p = data.shape
    
    if group_matrix is None:
        group_matrix = np.eye(p)
    
    all_cols = list(range(p))
    curr_cols = all_cols
    coeff = np.zeros((burn+iterations, p))
    log_ml = []
    
    for iteration in range(burn+iterations):

        candidate_cols = get_candidate_model_cols(curr_cols, all_cols, group_matrix)
        
        candidate_beta_hat, candidate_rss = fast_ols(**fast_ols_inputs, var_idx=candidate_cols)
        candidate_bic = ols_bic(candidate_rss, n, p)
        candidate_log_ml = -0.5 * candidate_bic + np.log(prior(len(candidate_cols)))
        
        if iteration==0 or min(np.exp(candidate_log_ml - curr_log_ml), 1) > np.random.rand():

            curr_log_ml = candidate_log_ml
            curr_cols = candidate_cols
            curr_beta_hat = candidate_beta_hat
        
        log_ml.append(curr_log_ml)
        coeff[iteration, curr_cols] = curr_beta_hat
        
    pip = (coeff[burn:] != 0).mean(axis=0)
    
    if method == 'average':
        optimal_model = coeff[burn:].mean(axis=0)
    elif method == 'median':  
        optimal_model = [list(data)[i] for i in range(p) if pip[i] > 0.5]
    elif method == 'max':
        max_model_cols = (coeff[burn:][np.argmax(log_ml[burn:]), :] != 0)
        optimal_model = [list(data)[i] for i in range(p) if max_model_cols[i]]
    else:
        print('method must be one of "average", "median" or "max"')
        
    return (
        optimal_model, 
        pd.DataFrame({'Variables':list(data), 'PIP':pip}).sort_values(by='PIP', ascending=False), 
        np.array(log_ml[burn:]), 
        coeff[burn:]
    )
        
        
def backward_selection(data: pd.DataFrame, 
                       target: pd.Series, 
                       weights=None, 
                       criteria='bic', 
                       group_matrix=None):
    '''
    Performs backwards selection for a linear regression
    
    Parameters
    ----------
    data: pd.DataFrame
        A pandas DataFrame containing the candidate set of predictors/exogenous variables
    target: pd.Series
        A pandas series containing the targets/endogenous variable
    weights
        A 
    '''
    
    fast_ols_inputs = get_fast_ols_inputs(data.values, target.values, weights)
    tss = np.sum((target.values - target.values.mean())**2)
    n, p = data.shape
    
    if group_matrix is None:
        group_matrix = np.eye(p)
    
    all_cols = list(range(p))
    curr_cols = all_cols
    
    _, min_rss = fast_ols(**fast_ols_inputs, var_idx=curr_cols)
    rss_list = [min_rss]
    curr_cols_list = [curr_cols]
    drop_groups = ['']
    
    while len(curr_cols) > 1:
        first_candidate = True
        for col in curr_cols:
            candidate_drop_group = [drop_col for drop_col in range(p) if (group_matrix.T[col,:]>0)[drop_col]]
            candidate_cols = [col for col in curr_cols if col not in candidate_drop_group]
            if len(candidate_cols) > 0:
                _, candidate_rss = fast_ols(**fast_ols_inputs, var_idx=candidate_cols)
            
            if first_candidate or candidate_rss < min_rss:
                min_rss = candidate_rss
                drop_group = candidate_drop_group
                first_candidate = False
                
        curr_cols = [col for col in curr_cols if col not in drop_group] 
        curr_cols_list.append(curr_cols)
        drop_groups.append(drop_group)
        rss_list.append(min_rss)
    
    bic = [ols_bic(rss_list[i], n, len(curr_cols_list[i])) for i in range(len(rss_list))]
    aic = [ols_aic(rss_list[i], n, len(curr_cols_list[i])) for i in range(len(rss_list))]
    adj_r_squared = [ols_adj_r_squared(rss_list[i], tss, n, len(curr_cols_list[i])) for i in range(len(rss_list))]
    
    if criteria == 'bic':     
        optimal_model = curr_cols_list[np.argmin(bic)]
    elif criteria == 'aic':
        optimal_model = curr_cols_list[np.argmin(aic)]
    elif criteria == 'adj. R-squared': 
        optimal_model = curr_cols_list[np.argmin(adj_r_squared)]
    
    return (
        optimal_model,
        pd.DataFrame({'BIC':bic,
                      'AIC':aic,
                      'Adj. R-squared':adj_r_squared,
                      'Dropped Variables':drop_groups,
                      'RSS':rss_list
                     })
    )
    
    
        
def fast_backward_selection(data, 
                            target, 
                            model, 
                            model_kwargs={},
                            refit_freq=2,
                            criteria='bic', 
                            group_matrix=None, 
                            p_value_threshold=0.001):
    
    X = data.values
    y = target.values
    n, p = X.shape
    
    if group_matrix is None:
        group_matrix = np.eye(p)
     
    all_cols = list(range(p))
    curr_cols = all_cols
    prev_cols = all_cols
    
    bic = []
    aic = []
    max_pvalue = []
    drop_groups = []
    
    while len(curr_cols) > 1:
        
        # fit current model 
        fit_model = model(y, X[:, curr_cols], **model_kwargs).fit()
        
        # get p-values for all variables in the current mdoel
        pvalues = []
        for col in curr_cols:
            group = group_matrix[col, curr_cols]
            pvalues.append(fit_model.wald_test(np.diag(group)[group>0, :]).pvalue.item(0))
            
        # evaluate model and stop if performance is worse
        if criteria == 'bic':
            criterion_val = fit_model.bic
        elif criteria == 'aic':
            criterion_val = fit_model.aic
        elif criteria == 'p-value':
            if max(pvalues) < p_value_threshold:
                break
                
        bic.append(fit_model.bic)
        aic.append(fit_model.aic)
        max_pvalue.append(max(pvalues))
        drop_groups.append(list(set(prev_cols) - set(curr_cols)))
        print(len(curr_cols))
        
        if curr_cols == all_cols: prev_criterion_val = criterion_val        
        if prev_criterion_val < criterion_val: break
            
        #remove variables with the highest p-values
        prev_cols = curr_cols
        prev_criterion_val = criterion_val
        
        for refit in range(refit_freq):
            drop_col = curr_cols[np.argmax(pvalues)]
            drop_group = [col for col in range(p) if (group_matrix.T[drop_col,:]>0)[col]]

            pvalues = [pvalues[i] for i in range(len(curr_cols)) if curr_cols[i] not in drop_group]
            curr_cols = [curr_cols[i] for i in range(len(curr_cols)) if curr_cols[i] not in drop_group] 
        
    return (
        [list(data)[col] for col in range(p) if col in prev_cols],
        pd.DataFrame({'Dropped Variables':drop_groups,
                      'BIC':bic,
                      'AIC':aic,
                      'Max p-value':max_pvalue
                     })
    )
    
def logistic_neg_loglikelihood(beta_hat, x, y, weights, group_matrix, C):
    y_hat = 1/(1 + np.exp(-np.matmul(x, beta_hat)))
    loglikelihood = np.dot(y * np.log(y_hat) + (1-y) * np.log(1-y_hat), weights)
    penalty = np.sum(np.matmul(group_matrix, beta_hat**2)**0.5)
    return -loglikelihood * C + penalty

def logistic_neg_gradient(beta_hat, x, y, weights, group_matrix, C):
    y_hat = 1/(1 + np.exp(-np.matmul(x, beta_hat)))
    logistic_gradient = np.dot(x.T, np.multiply(y-y_hat, weights))
    penalty_gradient = beta_hat * (np.matmul(group_matrix, beta_hat**2)**-0.5)
    return -logistic_gradient * C + penalty_gradient

from scipy.optimize import minimize

def group_lasso_selection(data, target, weights=None, group_matrix=None, C=1):
    X = data.values
    y = target.values
    
    if weights is None:
        weights = np.ones(len(X))
    
    if group_matrix is None:
        group_matrix = np.eye(X.shape[1])
    
    beta_hat, rss = fast_ols(**get_fast_ols_inputs(X, y, weights))
    beta_init = beta_hat/(rss/len(X))
    result = minimize(logistic_neg_loglikelihood, 
                      beta_init, 
                      method='BFGS', 
                      jac=logistic_neg_gradient, 
                      args=(X, y, weights, group_matrix, C),
                      options={'disp': True}
                     )
    return result.x