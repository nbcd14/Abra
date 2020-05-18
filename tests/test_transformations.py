from transformations import *
import numpy as np
import patsy


def test_get_ihs_factor():
    t_1 = pd.Series(1*np.sinh(np.random.normal(0,1,1000)))
    t_100 = pd.Series(100*np.sinh(np.random.normal(2,1,1000)))
    t_10000 = pd.Series(10000*np.sinh(np.random.normal(0,2,1000)))
    
    assert np.abs(get_ihs_factor(t_1)-1)/1 < 0.2
    assert np.abs(get_ihs_factor(t_100)-100)/100 < 0.2
    assert np.abs(get_ihs_factor(t_10000)-10000)/10000 < 0.2
    
def test_get_pareto_outlier_cap_cutoff():
    x = pd.Series(np.random.pareto(1,100000)+1)
    result = get_pareto_outlier_cap_cutoff(x, tail_pct=0.9, alpha=0.05, use_mle=True)
    assert np.abs((result/x.quantile(0.95)) - 1) < 0.2

    result = get_pareto_outlier_cap_cutoff(x, tail_pct=0.9, alpha=0.01, use_mle=False)
    assert np.abs((result/x.quantile(0.99)) - 1) < 0.2

    x = pd.Series(np.random.pareto(10,100000)+1)
    result = get_pareto_outlier_cap_cutoff(x, tail_pct=0.9, alpha=0.01, use_mle=True)
    assert np.abs((result/x.quantile(0.99)) - 1) < 0.2

    result = get_pareto_outlier_cap_cutoff(x, tail_pct=0.9, alpha=0.01, use_mle=False, p1=0.2, p2=0.95)
    assert np.abs((result/x.quantile(0.99)) - 1) < 0.2
    
    result = get_pareto_outlier_cap_cutoff(x, tail_pct=0.9, alpha=0.01, use_mle=False, p1=0.5, p2=0.5)
    assert result is None
    
    result = get_pareto_outlier_cap_cutoff(x, tail_pct=0.001, alpha=0.01, use_mle=False, p1=0.5, p2=0.6)
    assert result is None
    
def test_get_pareto_outlier_floor_cutoff():
    x = -pd.Series(np.random.pareto(1,100000)+1)
    result = get_pareto_outlier_floor_cutoff(x, tail_pct=0.9, alpha=0.05, use_mle=True)
    assert np.abs((result/x.quantile(0.05)) - 1) < 0.2

    result = get_pareto_outlier_floor_cutoff(x, tail_pct=0.9, alpha=0.01, use_mle=False)
    assert np.abs((result/x.quantile(0.01)) - 1) < 0.2

    x = -pd.Series(np.random.pareto(10,100000)+1)
    result = get_pareto_outlier_floor_cutoff(x, tail_pct=0.9, alpha=0.01, use_mle=True)
    assert np.abs((result/x.quantile(0.01)) - 1) < 0.2

    result = get_pareto_outlier_floor_cutoff(x, tail_pct=0.9, alpha=0.01, use_mle=False, p1=0.2, p2=0.95)
    assert np.abs((result/x.quantile(0.01)) - 1) < 0.2
    
    result = get_pareto_outlier_cap_cutoff(x, tail_pct=0.9, alpha=0.01, use_mle=False, p1=0.5, p2=0.5)
    assert result is None
    
    result = get_pareto_outlier_cap_cutoff(x, tail_pct=0.001, alpha=0.01, use_mle=False, p1=0.5, p2=0.6)
    assert result is None
    
def test_square():
    x = pd.Series(np.arange(10))

    result = square(x, center=5, flat_side='right')
    assert (result == np.array([25, 16,  9,  4,  1,  0,  0,  0,  0, 0])).all()

    result = square(x, center=5, flat_side='left')
    assert (result == np.array([0, 0,  0,  0,  0,  0,  1,  4,  9, 16])).all()
    
def test_ihs():
    assert np.isclose(
        ihs(np.array([-10, 0, np.e, 10])), 
        np.array([-2.99822295, 0, 1.72538256, 2.99822295])
    ).all()
    
def test_cap():
    x = pd.Series(np.arange(5))
    assert (cap(x, 6) == x).all()
    assert (cap(x, 3) == np.array([0, 1, 2, 3, 3])).all()
    assert (cap(x, -1) == np.array([-1,-1,-1,-1,-1])).all()
    
def test_floor():
    x = pd.Series(np.arange(5))
    assert (floor(x, 6) == np.array([6,6,6,6,6])).all()
    assert (floor(x, 3) == np.array([3, 3, 3, 3, 4])).all()
    assert (floor(x, -1) == x).all()
    
def test_impute_val():
    x = pd.Series([1, 3, 2])
    assert (impute_val(x, 3, 2) == pd.Series([1, 2, 2])).all()
    
def test_impute_null():
    x = pd.Series([1, np.nan, 2])
    y = pd.Series(['a', np.nan, 'b'])
    assert (impute_null(x, 2) == pd.Series([1, 2, 2])).all()
    assert (impute_null(y, 2) == pd.Series(['a', 2, 'b'])).all()

def test_flag():
    x = pd.Series([1, 1, 2])
    assert (flag(x, 1) == pd.Series([1, 1, 0])).all()
    
def test_impute_null():
    x = pd.Series([1, np.nan, 2])
    assert (null_flag(x) == np.array([0, 1, 0])).all()
    
def test_get_onehot_transform():
    cat_data = pd.DataFrame({
        'cat1':['a', 'b', 'b', 'c', 'a'], 
        'cat2':[1, 2, 2, 3, 1], 
        'cat3':[1, 'a', 'a', 3, 1]
    })

    onehot_transforms = get_onehot_transform(cat_data['cat1'], 'cat1') + \
                        get_onehot_transform(cat_data['cat2'], 'cat2') + \
                        get_onehot_transform(cat_data['cat3'], 'cat3')

    onehot_data = apply_transforms(cat_data, onehot_transforms)
    assert (onehot_data['cat1_eq_b'].values == np.array([0, 1, 1, 0, 0])).all()
    assert (onehot_data['cat1_eq_c'].values == np.array([0, 0, 0, 1, 0])).all()
    assert (onehot_data['cat2_eq_2'].values == np.array([0, 1, 1, 0, 0])).all()
    assert (onehot_data['cat2_eq_3'].values == np.array([0, 0, 0, 1, 0])).all()

def test_b_spline():
    data = pd.DataFrame({'test1':np.random.normal(0,1,1000), 
                         'test2':np.random.normal(0,1,1000), 
                         'test3':np.random.normal(0,1,1000)})

    test1_transform = get_bspline_transform(data['test1'], 'test1', df=3, degree=0, include_intercept=False)
    test2_transform = get_bspline_transform(data['test2'], 'test2', df=3, degree=2, include_intercept=True)
    test3_transform = get_bspline_transform(data['test3'], 'test3', df=3, degree=3, include_intercept=False)

    data2 = apply_transforms(data, [test1_transform, test2_transform, test3_transform])

    res1 = np.asarray(patsy.dmatrix("bs(x, df=3, degree=0, include_intercept=False) - 1", {"x": data['test1']}))
    assert np.isclose(res1, data2[['test1_s0', 'test1_s1', 'test1_s2']].values).all()

    res2 = np.asarray(patsy.dmatrix("bs(x, df=3, degree=2, include_intercept=True) - 1", {"x": data['test2']}))
    assert np.isclose(res2, data2[['test2_s0', 'test2_s1', 'test2_s2']].values).all()

    res3 = np.asarray(patsy.dmatrix("bs(x, df=3, degree=3, include_intercept=False) - 1", {"x": data['test3']}))
    assert np.isclose(res3, data2[['test3_s0', 'test3_s1', 'test3_s2']].values).all()