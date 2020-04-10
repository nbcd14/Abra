from selection import *
import statsmodels.api as sm
from sklearn import datasets

def test_get_correlated_columns():
    d = pd.DataFrame(np.random.random((10000,20)))
    d.columns = ['s{}'.format(col) for col in d.columns]
    d['s20'] = d['s19']*0.5
    d['s21'] = d['s19']*2
    d['s22'] = d['s18']
    d['s24'] = d['s17'] + np.random.normal(1,1,10000)/1000
    
    assert sorted(get_correlated_columns(d)) == sorted([['s19', 's20', 's21'], ['s18', 's22']])
    assert sorted(get_correlated_columns(d, 0.999)) == sorted([['s19', 's20', 's21'], ['s18', 's22'], ['s17', 's24']])
    
def test_get_non_numeric_columns():
    d = pd.DataFrame(np.random.random((10000,20)))
    d.columns = ['s{}'.format(col) for col in d.columns]
    rand = np.random.rand(10000)
    d['s20'] = np.where(rand>0.25, np.where(rand<0.5, 1, 2), 0)
    d['s21'] = np.array([i%19 for i in range(10000)])
    assert get_non_numeric_columns(d) == ['s20', 's21']
    assert get_non_numeric_columns(d, 19) == ['s20']

def test_get_unary_columns():
    d = pd.DataFrame(np.random.random((10000,20)))
    d.columns = ['s{}'.format(col) for col in d.columns]
    d['s20'] = 2
    d['s21'] = 3
    d.loc[1000,'s21']=1
    assert get_unary_columns(d) == ['s20']
    
    
def test_ols():
    X, y, coeff = sklearn.datasets.make_regression(100000, 40, 20,  noise=100, coef=True)
    n, p = X.shape
    weights = np.random.rand(n)
    fit_model = sm.OLS(y, X).fit()
    tol = 0.0001

    fast_ols_inputs = get_fast_ols_inputs(X, y)
    beta_hat_1 = fast_ols_beta_hat(fast_ols_inputs['XtX'], fast_ols_inputs['Xty'], var_idx=None)
    beta_hat_2, rss = fast_ols(**fast_ols_inputs, var_idx=None)
    log_likelihood = ols_log_likelihood(rss, n, p)
    bic = ols_bic(rss, n, p)
    aic = ols_aic(rss, n, p)
    adj_r_squared = ols_adj_r_squared(rss, np.sum((y-y.mean())**2), n, p)

    assert (np.abs(fit_model.params - beta_hat_1) < tol).all()
    assert (np.abs(fit_model.params - beta_hat_2) < tol).all()
    assert np.abs(np.sum((y-np.matmul(X, fit_model.params))**2) - rss) < tol
    assert np.abs(fit_model.llf - log_likelihood) < tol
    assert np.abs(fit_model.bic - bic) < tol
    assert np.abs(fit_model.aic - aic) < tol
    assert np.abs(fit_model.rsquared_adj - adj_r_squared) < tol
    
def test_wls():    
    X, y, coeff = sklearn.datasets.make_regression(100000, 40, 20,  noise=100, coef=True)
    n, p = X.shape
    weights = np.random.rand(n)
    fit_model = sm.WLS(y, X, weights).fit()
    tol = 0.0001

    fast_ols_inputs = get_fast_ols_inputs(X, y, weights)
    beta_hat_1 = fast_ols_beta_hat(fast_ols_inputs['XtX'], fast_ols_inputs['Xty'], var_idx=None)
    beta_hat_2, rss = fast_ols(**fast_ols_inputs, var_idx=None)
    
    assert (np.abs(fit_model.params - beta_hat_1) < tol).all()
    assert (np.abs(fit_model.params - beta_hat_2) < tol).all()
    assert np.abs(np.matmul((y-np.matmul(X, fit_model.params))**2, weights) - rss) < tol
    
def test_fast_bms():
    n_features = 100
    n_informative = 20
    X, y, coeff = sklearn.datasets.make_regression(100000, n_features, n_informative, noise=500, coef=True)

    results = fast_bms(pd.DataFrame(X),pd.Series(y), iterations=5000, 
                       burn=1000, method='median', prior=get_binomial(100,0.2))
    assert (coeff[results[0]] > 0).sum() > n_informative*0.8
    assert (coeff[list(set(range(100))- set(results[0]))] == 0).sum() > (n_features-n_informative)*0.8

    results = fast_bms(pd.DataFrame(X),pd.Series(y), iterations=5000, 
                       burn=1000, method='max', prior=get_binomial(100,0.2))
    assert (coeff[results[0]] > 0).sum() > n_informative*0.8
    assert (coeff[list(set(range(100))- set(results[0]))] == 0).sum() > (n_features-n_informative)*0.8
    
def test_backward_selection():
    n_features = 100
    n_informative = 20
    X, y, coeff = sklearn.datasets.make_regression(100000, n_features, n_informative,  noise=500, coef=True)

    results = backward_selection(pd.DataFrame(X),pd.Series(y), criteria='bic')
    assert (coeff[results[0]] > 0).sum() > n_informative*0.8
    assert (coeff[list(set(range(100))- set(results[0]))] == 0).sum() > (n_features-n_informative)*0.8

    results = backward_selection(pd.DataFrame(X),pd.Series(y), criteria='aic')
    assert (coeff[results[0]] > 0).sum() > n_informative*0.8
    assert (coeff[list(set(range(100))- set(results[0]))] == 0).sum() > (n_features-n_informative)*0.8




    