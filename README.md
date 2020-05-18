# Abra

This package implements some the functionality found in R GLM variable reduction packages (e.g. fastbw, bms) in python. In particular it contains the following functions:

1. fast_bms: implements bayesian model selection for linear regression with comparable speeds to the R bms function (where run time is independent of the number of samples). It supports the use of custom priors and weighted regression.
2. backward_selection: implements backwards selection for linear regression. The function supports weighted regression and group selection. Similar to fast_bms the function only supports linear regression, but run time is independent of the number of samples. 
3. fast_backward_selection: implements backwards selections for any statsmodel model. The function uses the wald test to determine which variables to drop at each iteration. The function supports weighted regression and group selection.
4. group_lasso_selection: implements group lasso for logistic regression. Note, the package [group-lasso](https://pypi.org/project/group-lasso/) has a better implementation of this algorithm.
    
Note, use of the original R packages is preferred. These should only be used if use of python is required.

The package also implements simple data cleaning functions useful for financial data. These include:

1. Adding median and mode imputations for nulls
2. Capping and flooring outliers
3. One hot encoding categorical variables
4. Applying inverse hyperbolic sine transforms to skewed variables
5. Angular transforms for proportions
6. Adding squared and interaction terms
7. Adding b-splines (using patsy b-splines)
    
Note unlike other packages, the cleaning pipeline is functional and stores transformation fitting parameters in human readable json, rather than objects. An example of how to use the pipeline can be found under examples.

Note the package is not complete. The following remains:

1. Add exception handling
2. Full unit test coverage
3. Clustering function for categorical variable with high cardinality
    
