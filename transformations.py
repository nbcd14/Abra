import pandas as pd
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm
import sys
from functools import reduce
import patsy



def ihs(x, ihs_factor=1, center=0):
    '''
    The inverse hyperbolic sine transformation. The transform is typically used for
    variables for positive right skewed variables, in place of a log transform.
    '''
    return np.arcsinh((x - center) / ihs_factor)

def get_ihs_factor(x):
    '''
    Returns the IHS factor for a variable (x), that results in the variable being as
    normally distributed as possible post IHS transformation.
    '''
    def fun(params, x, y):
        return params[0] + params[1] * ihs(x, params[2]) - y
    
    params_init = np.array([0, 1, 1])
    y = norm.ppf((x.rank() - 0.5) / len(x))
    res_lsq = least_squares(fun, params_init, args = (x, y))
    return res_lsq.x[2].item()


def angular(x, lower=0, upper=1):
    '''
    The arcsine square root transformation is typically used for proportions. 
    '''
    ratio = (x - lower) / (upper - lower)
    output = ratio.copy()
    output[ratio>1] = np.pi / 2 + np.sqrt(ratio[ratio>1] - 1)
    output[(ratio>=0)&(ratio<=1)] = np.arcsin(np.sqrt(ratio[(ratio>=0)&(ratio<=1)]))
    output[ratio<0] = -np.sqrt(-ratio[ratio<0])
    return output

def square(x, center=0, flat_side=None):
    '''
    Returns the square of the series provided. 
    
    Parameter
    ---------
    x: pd.Series
        The series to be squared
    center: float
        Specifies the vertex of returned quadratic term
    flat_side: str or None
        Either None, "left", or "right". If "left" all values less than the center are set to
        zero. If "right" all values greater than the center are set to zero.
    '''
    if flat_side == 'left':
        return floor(x - center, 0)**2
    elif flat_side == 'right':
        return cap(x - center, 0)**2
    return (x - center)**2
    
def interaction(x, y, x_center=0, y_center=0):
    
    return (x - x_center) * (y - y_center)


def get_numeric_columns(data, level_threshold=20):
    '''
    Takes a dataset and returns a list of column names for the numeric columns of the dataset.
    Columns that do not have an object dtype or a less the level_threshold uniques values are considered
    numeric.
    '''
    numeric_columns = []
    for col_name in list(data):
        if data[col_name].dtype != 'object' and len(data[col_name].unique()) > level_threshold:
            numeric_columns.append(col_name)
    return numeric_columns
    
def b_spline(x, knots, degree, upper_bound, lower_bound, include_intercept):
    '''
    A wrapper for the patsy bs formula, used to create basis vectors for input series x.
    '''
    patsy_formula = "bs(x, knots={}, degree={}, upper_bound={}, lower_bound={}, include_intercept={})-1"

    design_matrix = patsy.dmatrix(
                        patsy_formula.format(
                            knots,
                            degree,
                            upper_bound,
                            lower_bound,
                            include_intercept
                        ), 
                        {"x": floor(cap(x, upper_bound), lower_bound)}
                    )
    return np.asarray(design_matrix)
    
def get_square_transforms_for_dataset(data, include_cols=[]):
    '''
    Return a list of dicts used by the apply_tranforms to square variables.  
    '''
    if not include_cols:
        include_cols = get_numeric_columns(data)
        
    square_transforms = []    
    for col_name in include_cols:
        if data[col_name].skew() < 0:
            center = data[col_name].max()
            flat_side = 'right'
        else:
            center = data[col_name].min()
            flat_side = 'left'
            
        square_transforms.append(
            get_transform(
                col_name+'_sqr', 
                col_name, 
                'square', 
                params= {
                    'center':center,
                    'flat_side':flat_side
                }
            )
        )
    return square_transforms
    
    
def get_pareto_outlier_cap_cutoff(x, tail_pct=0.01, alpha=0.01, use_mle=True, p1=0.1, p2=0.9):
    '''
    Determines the cap value for a variable, by fitting a pareto distribution to the right tail
    of the variable. Values that are unlikely to occur under the fitted pareto distribution (i.e. in
    the tail of the fitted pareto) are considered outliers and fall above the returned cap value.
    Function is based on Sornette, 2009.
    
    Parameters
    ----------
    x: pd.Series
        The variable that requires capping
    tail_pct: float
        Specifies what percentage of the largest values of x should be used to fit the pareto distribution
    alpha: float
        The likelihood that values above the cap value are not outliers, assuming the fitted pareto
        distribution accurately describes the tail of x (i.e. the cumulative mass of the pareto above the cap
        value equals alpha)
    use_mle: boolean
        If true, the parameters of the pareto distribution are estimated using maximum likelihood estimation.
        Note that the presence of outliers may bias the parameter estimates.
    p1: float
        If use_mle is false, the parameters of the pareto are fit using the method of quantiles (i.e. the
        parameters lam and beta are estimated by solving p1 = 1 - (beta/q1)^lam and p2 = 1 - (beta/q2)^lam 
        where q1 and q2 are samples from the tail corresponding to the p1 and p2 percentiles).
    p2: float
        See p1 descripton
    '''
    tail = x[x > x.quantile(1 - tail_pct)]
    if use_mle:
        beta = np.min(tail)
        lam = 1/(np.log(tail/beta).mean())
        return beta * alpha**(-1 / lam)
    else:
        q1 = tail.quantile(p1)
        q2 = tail.quantile(p2)
        if len(tail) > 100 and q2 > q1 and q1 > 0:
            lam = np.log((1 - p1) / (1 - p2)) / np.log(q2 / q1)
            beta = q1 * (1 - p1)**(1 / lam)
            return (beta * alpha**(-1 / lam)).item()
            
    

def get_pareto_outlier_floor_cutoff(x, tail_pct=0.01, alpha=0.01, use_mle=True, p1=0.1, p2=0.9):
    '''
    Determines the floor value for a variable, by fitting a pareto distribution to the left tail
    of the variable. Values that are unlikely to occur under the fitted pareto distribution (i.e. in
    the tail of the fitted pareto) are considered outliers and fall below the returned floor value.
    
    Parameters
    ----------
    x: pd.Series
        The variable that requires flooring
    tail_pct: float
        Specifies what percentage of the smallest values of x should be used to fit the pareto distribution
    alpha: float
        The likelihood that values below the floor value are not outliers, assuming the fitted pareto
        distribution accurately describes the tail of x (i.e. the cumulative mass of the pareto below the floor
        value equals alpha)
    use_mle: boolean
        If true, the parameters of the pareto distribution are estimated using maximum likelihood estimation.
        Note that the presence of outliers may bias the parameter estimates.
    p1: float
        If use_mle is false, the parameters of the pareto are fit using the method of quantiles (i.e. the
        parameters lam and beta are estimated by solving p1 = 1 - (beta/q1)^lam and p2 = 1 - (beta/q2)^lam 
        where q1 and q2 are samples from the tail corresponding to the p1 and p2 percentiles).
    p2: float
        See p1 descripton
    '''
    cutoff = get_pareto_outlier_cap_cutoff(-x, tail_pct, alpha, use_mle, p1, p2)
    if cutoff is not None:
        return -cutoff

def cap(x, cutoff):
    return np.where(x > cutoff, cutoff, x)

def floor(x, cutoff):
    return np.where(x < cutoff, cutoff, x)

def impute_val(x, value, imputation):
    return np.where(x == value, imputation, x)

def impute_null(x, imputation):
    return np.where(x.isnull(), imputation, x)

def flag(x, value):
    return np.where(x == value, 1, 0)

def null_flag(x):
    return np.where(x.isnull(), 1, 0)

def apply_transforms(dataset, transforms):
    '''
    Applies a list of tranforms to the columns of the dataset provided and returns a new
    dataset with the transformed columns
    
    Parameters
    ----------
    data: pd.DataFrame
        A dataFrame with columns to be tranformed
    transforms: list of dicts
        A list of dicts describing the tranforms to be applied. Each dict contains the name
        of the column(s) to be transformed (input), the function to be applied (func), additional
        parameters for the function (params), the name(s) for the new transformed column(s)
        (output), and whether to drop the original column(s) (drop_input)

    '''
    data = dataset.copy()
    for transform in transforms:
        func = getattr(sys.modules[__name__], transform['func'])
        if isinstance(transform['output'], list):
            outputs = func(data[transform['input']], **transform['params'])
            for i in range(len(transform['output'])):
                data[transform['output'][i]] = outputs[:, i]
        else:
            data[transform['output']] = func(data[transform['input']], **transform['params'])
        if transform['drop_input'] and transform['output'] != transform['input']:
            data = data.drop(transform['input'], axis=1)
    return data

def get_transform(output, inp, func, params={}, drop_input=False):
    '''
    Creates a dict describing a tranformation that can be applied using the function 
    apply_tranforms 
    
    Parameters
    ----------
    output: str or list(str)
        The name(s) for the transformed column(s) 
    inp: str or list(str)
        The name of the column(s) to be transformed
    func: str
        The name of the function/transformation to be applied to the inp column(s)
    params: dict
        Additional parameters for func
    drop_input: boolean
        Whether to drop the original inp column(s) after adding the transformed output 
        column(s)
    '''
    return {
        'output': output,
        'input': inp,
        'func': func,
        'params': params,
        'drop_input': drop_input
    }
    
def get_onehot_transform(x: pd.Series, col_name):
    '''
    Takes a categorical variable and returns a list of dicts representing the transforms
    used by apply_transforms to one-hot encode the variable. Note flags are created for each
    minority category in the column.
    '''
    transforms = []
    for val in list(x.unique()):
        if val != x.mode()[0]:
            transforms.append(
                get_transform(
                    col_name + '_eq_' + str(val),
                    col_name,
                    'flag',
                    {'value': val.item() if hasattr(val, 'item') else val}
                )
            )
    return transforms
    
def get_impute_null_transforms_for_dataset(data, add_flag=True):
    '''
    Returns a list of dicts describing imputations for the columns of the dataset provided.
    Categorical variables recieve a mode imputaton, while numerical columns recieve a
    median imputation.
    
    Parameters
    ----------
    data: pd.DataFrame
        The training dataset used to determine the imputations used for each column
    add_flag: boolean
        If true, a missing flag transform is added for each variable in addition to
        a imputation transform
    '''
    
    transforms = []
    for col_name in list(data):

        if data[col_name].dtype == 'object':
            imputation = data[col_name].mode()[0]
        else:
            imputation = data[col_name].median()
        
        if add_flag:
            transforms.append(
                get_transform(
                    col_name+'_missing',
                    col_name,
                    'null_flag',
                    {}
                )
            )
    
        transforms.append(
            get_transform(
                col_name,
                col_name,
                'impute_null',
                {'imputation':imputation}
            )
        )
    
    return transforms
    
def get_cap_floor_transforms_for_dataset(data, tail_pct=0.01, alpha=0.01, use_mle=False, p1=0.1, p2=0.9):
    '''
    Returns a list of dicts describing capping and flooring transforms for the numerics 
    columns of the dataset provided. Cap and floor values are determined by fitting a pareto
    distribution to the tails of each column using get_pareto_outlier_cap_cutoff and
    get_pareto_outlier_floor_cutoff. 
    '''
    transforms = []
    for col_name in get_numeric_columns(data):

        cap_cutoff = get_pareto_outlier_cap_cutoff(data[col_name], tail_pct, alpha, use_mle, p1, p2)
        floor_cutoff = get_pareto_outlier_floor_cutoff(data[col_name], tail_pct, alpha, use_mle, p1, p2)
    
        if cap_cutoff is not None:
            transforms.append(
                get_transform(
                    col_name,
                    col_name,
                    'cap',
                    {'cutoff':cap_cutoff}
                )
            )
    
        if floor_cutoff is not None:
            transforms.append(
                get_transform(
                    col_name,
                    col_name,
                    'floor',
                    {'cutoff':floor_cutoff}
                )
            )
    
    return transforms

def get_mode_flag_transforms_for_dataset(data, mode_pct_threshold):
    '''
    Returns a list of dicts used by apply_transforms to add flags indicating the mode of the
    variables of the dataset provided, if they constitute a significant portion of each variable
    (i.e. above the mode_pct_threshold). This assumes that the mode carries some additional
    significance (e.g. zero inflated variables).
    '''
    transforms = []
    for col_name in get_numeric_columns(data):
        
        mode = data[col_name].mode()[0]
        mode_pct = (data[col_name] == mode).sum()/len(data[col_name])
        print(mode_pct)
        if mode_pct > mode_pct_threshold:
            transforms.append(
                get_transform(
                    col_name+'_eq_'+str(mode),
                    col_name,
                    'flag',
                    {'imputation':mode}
                )
            )
    
    return transforms
    
def get_onehot_tranforms_for_dataset(data):
    '''
    Applies the function get_onehot_tranforms to each categorical column of the dataset 
    provided and returns as list of dicts used by apply_tranforms to one hot encode all the
    categorical columns in the dataset.
    '''
    transforms = []
    numeric_columns = get_numeric_columns(data)
    for col_name in list(data):
        if col_name not in numeric_columns and len(data[col_name].unique()) > 2:
            transforms = transforms + get_onehot_transform(data[col_name], col_name)
    return transforms
    
def get_ihs_transforms_for_dataset(data, drop_input=False, skew_threshold=1, kurtosis_threshold=2):
    '''
    Returns a list of dicts used by apply_tranforms to apply the IHS (inverse hyperbolic sine) transform
    to numeric columns with high kurtosis and right skew in the dataset provided. The IHS factor for
    each column is determined using get_ihs_factor.
    
    Parameters
    ----------
    data: pd.DataFrame
        The dataset used to determine which columns require an IHS transform and the IHS factor for
        each column
    drop_input: boolean
        Whether to drop the original untransformed columns after the IHS transformed columns are added
    skew_threshold: float
        Numeric columns with skew less than this value will not recieve an IHS transform
    kurtosis_thereshold: float
        Numeric columns with kurtosis less than this value will not recieve an IHS transform
    '''
    transforms = []
    for col_name in get_numeric_columns(data):
        if data[col_name].skew() > skew_threshold and data[col_name].kurtosis() > kurtosis_threshold:
            ihs_factor = get_ihs_factor(data[col_name])
            transforms.append(
                get_transform(
                    col_name+'_ihs',
                    col_name,
                    'ihs',
                    {'ihs_factor':ihs_factor},
                    drop_input
                )
            )
    
    return transforms

def get_bspline_transform(x, col_name, df=4, degree=3, include_intercept=False):
    '''
    Creates a dict used by apply_transforms to add b-spline basis vectors for the provided column.
    The function takes the same arguments as the patsy bs formula and stores the parameter
    required by the patsy bs function to create basis vectors for new data, similar to the 
    patsy stateful transforms (but without storing the parameters in an object). 
    
    Parameters
    ----------
    x: pd.Series
        The series for which to create a b-spline basis for
    col_name: str
        The column name of the series (x) provided, used to create the column names for the basis vectors
    df: int
        The number of basis vectors to create (passed to patsy bs)
    degree: int
        The degree of the spine (passed to patsy bs)
    include_intercept: boolean
        Whether to include an intercept term in the basis (passed to patsy bs)  
    '''
    n_knots = df - degree - 1
    if not include_intercept:
        n_knots += 1
    
    output = ['{}_s{}'.format(col_name, i) for i in range(df)]
    knots = [np.percentile(x, 100 * prob) for prob in np.linspace(0, 1, n_knots + 2)[1:-1]]
    transform = get_transform(
                    output,
                    col_name,
                    'b_spline',
                    params= {
                        'knots':knots,
                        'degree':degree,
                        'upper_bound':x.max(),
                        'lower_bound':x.min(),
                        'include_intercept':include_intercept
                    },
                    drop_input=False
                )
    return transform

def get_bspline_transforms_for_dataset(data, df=4, degree=3, include_cols=[], exclude_cols=[]):
    '''
    Applies the function get_bspline_transform to each numeric column of the dataset provided,
    and returns a list of dicts used by apply_transforms to create spline basis vectors. Note function
    assumes none of the splines include an intercept.
    
    Parameters
    ----------
    data: pd.DataFrame
        The data set with columns for which to create splines for 
    df: int
        The number of basis vectors to create for each spline (passed to patsy bs)
    degree: int
        The degree of each spine (passed to patsy bs)
    include_cols: list of str
        The column names of the variables in data that the function should create spline tranforms for.
        If not included spline transforms for all numeric columns are created.
    exclude_cols: list of str
        The column names for the variables for which b-spline transforms should not be created.
    '''
    transforms = []
    
    if not include_cols:
        include_cols = get_numeric_columns(data)
                    
    for col_name in include_cols:
        if col_name not in exclude_cols:
            transforms.append(
                get_bspline_transform(
                    data[col_name],
                    col_name,
                    df,
                    degree,
                    False
                )
            )
    return transforms


def get_standard_transforms_for_dataset(dataset):
    '''
    Returns a list of dicts representing imputation, cap, floor, onehot, and ihs transforms. The
    list of dicts is used by apply_transforms to apply each transform to a dataset.
    '''
    
    data = dataset.copy()
    
    null_transforms = get_impute_null_transforms_for_dataset(data)
    data = apply_transforms(data, null_transforms)

    cap_floor_transforms = get_cap_floor_transforms_for_dataset(data, use_mle=False)
    data = apply_transforms(data, cap_floor_transforms)

    onehot_transforms = get_onehot_tranforms_for_dataset(data)
    
    return null_transforms+cap_floor_transforms+onehot_transforms

def get_group_dict(transforms):
    '''
    Takes a list of dict describing data transforms and returns a dict, where keys are column names and
    their corresponding values are a list of columns that must be included with the key in any model.
    The output is used by get_group_matrix that creates a group matrix that is used in group/hierarchical
    selection procedures. In particular the function requires that:
        1. The b-spline basis vectors of a variable must be included together
        2. Models with interactions or squared terms must include the corresponding main effects
        3. TBD: missing flags must be included for any variable in the model
    '''
    group_dict = {}
    for transform in transforms:
        if transform['func'] == 'b_spline':
            for col_name in transform['output']:
                group_dict[col_name] = transform['output']
        elif transform['func'] == 'square':
            group_dict[transform['output']] = [transform['input']]
        elif transform['func'] == 'interaction':
            group_dict[transform['output']] = transform['input']
            
    return group_dict

def get_group_matrix(data, group_dict):
    '''
    Takes a group dict (created by get_group_dict) describing columns that must be included together in
    a model by any selection procedures and creates a group matrix, where the ij-th entry is 1 if variable
    j must included if variable i is included and 0 otherwise.
    
    Parameters
    ----------
    data: pd.DataFrame
        The data set for which to create a group matrix for
    group_dict: dict
        A dict where keys are column and their corresponding values are a list of columns that must be 
        included with the key column in a model
    '''
    n = data.shape[1]
    idx_map = dict(zip(list(data), list(range(n))))
    group_matrix = np.eye(n)
    
    for col_name in group_dict.keys():
        group = [idx_map[col] for col in group_dict[col_name]]
        group_matrix[idx_map[col_name], group] = 1
    return group_matrix
        

def safe_make_list(item):
    '''Helper function for prune_transforms.'''
    if isinstance(item, list):
        return item
    else: 
        return [item]

def prune_transforms(transforms, required_cols):
    '''
    Removes transforms from the transforms provided that not needed to create
    the required_cols. The function is used after variable selection to remove
    transforms used to create variables in the candidate set that did not make
    it into the final model.
    
    Returns
    --------
    required_transforms: list of dicts
        The transforms required to create the required_cols
    parentless: list of str
        The names of the input columns required to make required_cols
    '''

    children = required_cols.copy()
    required_transforms = []
    parentless = []
    
    while True:
        parents = []
        for child in children:
            parent_found = False
            for transform in transforms:
                if child in safe_make_list(transform['output']):
                    required_transforms.append(transform)
                    for input_col in safe_make_list(transform['input']):
                        if child != input_col:
                            parents.append(input_col)
                            parent_found = True
            if not parent_found:
                parentless.append(child)
        if len(parents) == 0:
            break
        children = parents
    
    return reversed(required_transforms), parentless
                       
    
        
         