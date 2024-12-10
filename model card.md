Model Card: House Price Prediction

Basic Information
Author: Chipo Jaya, chipo.jaya@gwu.edu
Model Date: December 10, 2024
Model Version: 1.0
License: MIT
Model Implementation Code: House_Price_Prediction_Model

Intended Use
Primary Use: Predict house sale prices based on property features for real estate analysis and market trend predictions.
Intended Users: Real estate professionals, data analysts, and researchers.
Out-of-Scope Uses: This model is not suitable for applications requiring legal or financial guarantees or for use outside the distribution of the training dataset.

Training Data
Source: Kaggle's "House Prices: Advanced Regression Techniques" competition dataset
Number of Rows: 1,460 in training data
Key Features: OverallQual, GrLivArea, TotalBsmtSF, GarageCars, FullBath, YearBuilt, LotArea

Test Data
Source: Kaggle's "House Prices: Advanced Regression Techniques" test dataset
Number of Rows: 1,459
Differences: The test dataset lacks the SalePrice column, which is the target variable in the training dataset.

Model Details
Input Features: OverallQual, GrLivArea, TotalBsmtSF, GarageCars, FullBath, YearBuilt, LotArea
Target Variable: SalePrice (continuous numerical variable)
Type of Model: XGBoost Regressor
Software Used: Python 3.8, scikit-learn 0.24.2, XGBoost 1.4.2
Hyperparameters:
n_estimators: 200
max_depth: 3
learning_rate: 0.1

Quantitative Analysis
Metrics Used: Root Mean Squared Error (RMSE), R-squared (R²)
Final Values:
Best Model Score (MSE): 751,474,682.02
Cross-Validation RMSE: $31,178.57 (± $7,242.93)

Potential Negative Impacts
Mathematical or Software Problems:
Potential overfitting to training data
Bias towards features more common in the training dataset
Real-world Risks:
May perpetuate existing biases in housing markets
Could be misused to justify discriminatory pricing practices

Uncertainties
Mathematical or Software 
Problems:
Model performance on data significantly different from the training set
Handling of outliers or unusual property characteristics
Real-world Risks:
Changes in market conditions not reflected in the training data
Regional differences in housing markets not captured by the model

Unexpected Results
The model showed exceptionally high performance on the training data, with an R² score of 0.9998, which suggests potential overfitting. This unexpected result warrants further investigation and validation on unseen data.

Link to Jupyter Notebook house_prices_model.ipynb 
