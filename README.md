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

## Reference

[1] Briner, Beat, Rachael Smith, and Paul Ward. 2009. “The Barra European Equity Model (EUE3).” Research Notes.

[2] Jose Menchero , D.J. Orr and Jun Wang. 2011. “The Barra US Equity Model (USE4).” Methodology Notes.

[3] Menchero, Jose, Jun Wang, and D.J. Orr. 2011. “Eigen-Adjusted Covariance Matrices.” MSCI Research Insight.

[4] Menchero, Jose, and Andrei Morozov. "Improving Risk Forecasts through Cross-Sectional Observations." The Journal of Portfolio Management 41.3 (2015): 84-96.
