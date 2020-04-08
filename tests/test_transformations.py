from transformations import *
import numpy as np


def test_get_ihs_factor():
    t_1 = pd.Series(1*np.sinh(np.random.normal(0,1,1000)))
    t_100 = pd.Series(100*np.sinh(np.random.normal(2,1,1000)))
    t_10000 = pd.Series(10000*np.sinh(np.random.normal(0,2,1000)))
    
    assert np.abs(get_ihs_factor(t_1)-1)/1 < 0.2
    assert np.abs(get_ihs_factor(t_100)-100)/100 < 0.2
    assert np.abs(get_ihs_factor(t_10000)-10000)/10000 < 0.2