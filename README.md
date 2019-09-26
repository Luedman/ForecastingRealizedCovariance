## Forecasting Realized Covariance Matrices with LSTM and Echo State Networks 
Master Thesis (Work in Progress)
Lukas Schreiner

### 1.1 Models Univariate Forecasting of Realized Variance

* LSTM
    - 32 LSTM Cells, 1 Layer
    - Adam Optimizer with learning rate decay
    - Gradient Clipping at 0.5
    - Dropout = 0.01
    - Regularization = 0.001
    - Activation Functions: Relu
    - 5000 training epochs with early stopping
* Echo State Network
    - internalNodes : 100 
    - spectralRadius': 0.17
    - regressionLambda': 1.0
    - connectivity': 0.011
    - leakingRate': 0.08
* Univariate HAR

### 1.2 Models Univariate Forecasting of Realized Variance

tbd

### 2.1 Results Univariate 

#### RMSE Error Measure
<div align='center'>
  <img src='Pictures/Figure1a.png'>
</div>

#### QLIK Error Measure
<div align='center'>
  <img src='Pictures/Figure1b.png'>
</div>

### L1 Norm Error Measure
<div align='center'>
  <img src='Pictures/Figure1c.png'>
</div>

### 2.2 Results Multivariate

tbd

### 3. Data
Oxford-Man Institute of Quantitative Finance
Dataset can be found [here](https://realized.oxford-man.ox.ac.uk)