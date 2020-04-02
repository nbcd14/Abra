from selection import *
import statsmodels.api as sm
from sklearn import datasets

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
    