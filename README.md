# Barra risk model

## utils.py
1. `ewa()`: exponential weighted average
2. `cov_ewa()`: covariance matrix with each squared range has an exponential weight
3. `num_eigvals_explain()`: the number of eigenvalues it takes to explain a percentage of total variance
4. `draw_eigvals_edf()`: to draw the empirical distribution function (EDF) of eigenvalues of a covariance matrix
   
## bias_statistics.py
`BiasStatsCalculator` with 2 versions of bias statistics calculation:
1. single window
2. rolling window

## factor_covariance_adjustment.py
`FactorCovAdjuster` with 3 factor covariance matrix adjustment methods in Barra risk model:
1. Newey-West adjustment
2. eigenfactor risk adjustment
3. volatility regime adjustment
