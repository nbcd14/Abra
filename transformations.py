import pandas
from scipy.optimize import least_squares
from scipy.stats import norm
import sys
from functools import reduce


def get_ihs_factor(x):
    def fun(params, x, y):
        return params[0] + params[1] * np.log((x/params[2])+((x/params[2])**2 + 1)**5) - y
    
    params_init = np.array([0, 1, x.mean() / (2 * np.sqrt(3))])
    y = norm.ppf((x.rank() - 0.375) / (len(x) + 0.25))
    res_lsq = least_squares(fun, params_init, args = (x, y))

def ihs(x, ihs_factor=None, center=0):
    if ihs_factor is None:
        ihs_factor = x.mean() / (2 * np.sqrt(3))
    return np.arcsinh((x - center) / ihs_factor)

def tte(x):
    return np.reciprocal(x + 1.0)

def angular(x, lower=0, upper=1):
    ratio = (x - lower) / (upper - lower)
    return np.where(ratio < 1, 
                    np.pi / 2 + np.sqrt(ratio - 1), 
                    np.where(ratio < 0, 
                             -np.sqrt(-ratio), 
                            np.arcsin(np.sqrt(ratio))
                            )
                   )

def get_dragon_king_cap_cutoff(x, tail_pct=0.01, alpha=0.01, p1=0.1, p2=0.9):
    tail = x[x > x.quantile(1 - tail_pct)]
    ql = tail.quantile(p1)
    q2 = tail.quantile(p2)
    if len(tail) > 100 and q2 > q1 and q1 > 0:
        lam = np.log((1 - p1) / (1 - p2)) / np.log(q2 / q1)
        beta = q1 * (1 - p1)**(1 / lam)
        return beta * alpha**(-1 / lam)

def get_dragon_king_floor_cutoff(x, tail_pct=0.01, alpha=0.01, p1=0.1, p2=0.9):
    cutoff = get_dragon_king_cap_cutoff(-x, tail_pct, alpha, p1, p2)
    if cutoff is not None:
        return -cutoff

def cap(x, cutoff):
    return np.where(x > cutoff, cutoff, x)

def floor(x, cutoff):
    return np.where(x < cutoff, cutoff, x)

def impute_val(x, imputation):
    return np.where(x == value, imputation, x)

def impute_null(x, imputation):
    return np.where(x.isnull(), imputation, x)

def flag(x, value):
    return np.where(x == value, 1, 0)

def null_flag(x, value):
    return np.where(x.isnull(), 1, 0)

def apply_transforms(dataset, transforms):
    data = dataset.copy()
    for transform in transforms:
        func = getattr(sys.modules[__name__], transform['func'])
        data[transform['output']] = func(data[transform['input']], **transform['params'])
        if transform['drop_input'] and transform['output'] != transform['input']:
            data = data.drop(columns=transform['input'])
    return data

def get_transform(output, inp, func, params={}, drop_input=False):
    return {
        'output': output,
        'input': inp,
        'func': func,
        'params': params,
        'drop_input': drop_input
    }
    
def get_onehot_transform(x):
    transforms = []
    for val in list(x.unique()):
        if val != x.mode()[0]:
            tranforms.append(
                get_transform(
                    col_name + '_eq_' + str(val),
                    col_name,
                    'flag',
                    {'value': val}
                )
            )
    return transforms
    
def get_impute_null_transforms_for_dataset(data, add_flag=True):
    transforms = []
    for col_name in list(data):
        if data[col_name].dtype == 'object':
            imputation = data[col_name].mode()[0]
        else:
            imputation = data[col_name].median()[0]
        
        if add_flag:
            transforms.append(
                get_transforms(
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
    
def get_cap_floor_transforms_for_dataset(data, tail_pct=0.01, alpha=0.01, p1=0.1, p2=0.9):
    transforms = []
    for col_name in list(data):
        if data[col_name].dtype != 'object':
            
            cap_cutoff = get_dragon_king_cap_cutoff(data[col_name], tail_pct, alpha, p1, p2)
            floor_cutoff = get_dragon_king_floor_cutoff(data[col_name], tail_pct, alpha, p1, p2)
    
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
    
def get_onehot_tranforms_for_dataset(data):
    transforms = []
    for col_name in list(data):
        if data[col_name].dtype == 'object' and len(data[col_name].unique()) < 10:
            transforms = transforms + get_onehot_transform(data[col_name], col_name)
    
def get_ihs_transforms_for_dataset(data, skew_threshold=1.14, kurtosis_threshold=2.4, level_threshold=20):
    transforms = []
    for col_name in list(data):
        if data[col_name].dtype != 'object' and len(data[col_name].unique()) > level_threshold:
            if data[col_name].skew() > skew_threshold and data[col_name].kurtosis() > kurtosis_threshold:
                ihs_factor = get_ihs_factor(data[col_name])
                transforms.append(
                    get_transform(
                        col_name+'_ihs',
                        col_name,
                        'ihs',
                        {'ihs_factor':ihs_factor}
                    )
                )
    
    return transforms
         