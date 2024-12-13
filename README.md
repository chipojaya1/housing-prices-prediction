# Model Card: House Price Prediction

### Basic Information

  * **Person developing the model**: Chipo Jaya, `chipo.jaya@gwu.edu`
  * **Model Date**: December 10, 2024
  * **Model Version**: 1.0
  * **License**: MIT
  * **Model Implementation Code**: [DNSC_6301_Extra_Credit_Assignment](https://github.com/chipojaya1/housing-prices-prediction/blob/main/house_prices_model.ipynb)

### Intended Use
  * **Primary Use**: Predict house sale prices based on property features for real estate analysis and market trend predictions.
  * **Intended Users**: Real estate professionals, data analysts, and researchers.
  * **Out-of-Scope Uses**: This model is not suitable for applications requiring legal or financial guarantees or for use outside the distribution of the training dataset.

### Training Data
  * Data dictionary:
  
  | Name | Modeling Role | Measurement Level | Description |
  | ---- | ------------- | ---------------- | ---------- |
  | **MSSubClass** | Input | Categorical | Identifies the type of dwelling involved in the sale |
  | **MSZoning** | Input | Categorical | Identifies the general zoning classification of the sale |
  | **LotFrontage** | Input | Continuous | Linear feet of street connected to property |
  | **LotArea** | Input | Continuous | Lot size in square feet |
  | **Street** | Input | Categorical | Type of road access to property |
  | **Alley** | Input | Categorical | Type of alley access to property |
  | **LotShape** | Input | Categorical | General shape of property |
  | **LandContour** | Input | Categorical | Flatness of the property |
  | **Utilities** | Input | Categorical | Type of utilities available |
  | **LotConfig** | Input | Categorical | Lot configuration |
  | **LandSlope** | Input | Categorical | Slope of property |
  | **Neighborhood** | Input | Categorical | Physical locations within Ames city limits |
  | **Condition1** | Input | Categorical | Proximity to various conditions |
  | **Condition2** | Input | Categorical | Proximity to various conditions (if more than one is present) |
  | **BldgType** | Input | Categorical | Type of dwelling |
  | **HouseStyle** | Input | Categorical | Style of dwelling |
  | **OverallQual** | Input | Ordinal | Rates the overall material and finish of the house |
  | **OverallCond** | Input | Ordinal | Rates the overall condition of the house |
  | **YearBuilt** | Input | Discrete | Original construction date |
  | **YearRemodAdd** | Input | Discrete | Remodel date (same as construction date if no remodeling or additions) |
  | **RoofStyle** | Input | Categorical | Type of roof |
  | **RoofMatl** | Input | Categorical | Roof material |
  | **Exterior1st** | Input | Categorical | Exterior covering on house |
  | **Exterior2nd** | Input | Categorical | Exterior covering on house (if more than one material) |
  | **MasVnrType** | Input | Categorical | Masonry veneer type |
  | **MasVnrArea** | Input | Continuous | Masonry veneer area in square feet |
  | **ExterQual** | Input | Ordinal | Evaluates the quality of the material on the exterior |
  | **ExterCond** | Input | Ordinal | Evaluates the present condition of the material on the exterior |
  | **Foundation** | Input | Categorical | Type of foundation |
  | **BsmtQual** | Input | Ordinal | Evaluates the height of the basement |
  | **BsmtCond** | Input | Ordinal | Evaluates the general condition of the basement |
  | **BsmtExposure** | Input | Ordinal | Refers to walkout or garden level walls |
  | **BsmtFinType1** | Input | Ordinal | Rating of basement finished area |
  | **BsmtFinSF1** | Input | Continuous | Type 1 finished square feet |
  | **BsmtFinType2** | Input | Ordinal | Rating of basement finished area (if multiple types) |
  | **BsmtFinSF2** | Input | Continuous | Type 2 finished square feet |
  | **BsmtUnfSF** | Input | Continuous | Unfinished square feet of basement area |
  | **TotalBsmtSF** | Input | Continuous | Total square feet of basement area |
  | **Heating** | Input | Categorical | Type of heating |
  | **HeatingQC** | Input | Ordinal | Heating quality and condition |
  | **CentralAir** | Input | Categorical | Central air conditioning |
  | **Electrical** | Input | Categorical | Electrical system |
  | **1stFlrSF** | Input | Continuous | First Floor square feet |
  | **2ndFlrSF** | Input | Continuous | Second floor square feet |
  | **LowQualFinSF** | Input | Continuous | Low quality finished square feet (all floors) |
  | **GrLivArea** | Input | Continuous | Above grade (ground) living area square feet |
  | **BsmtFullBath** | Input | Discrete | Basement full bathrooms |
  | **BsmtHalfBath** | Input | Discrete | Basement half bathrooms |
  | **FullBath** | Input | Discrete | Full bathrooms above grade |
  | **HalfBath** | Input | Discrete | Half baths above grade |
  | **Bedroom** | Input | Discrete | Bedrooms above grade (does NOT include basement bedrooms) |
  | **Kitchen** | Input | Discrete | Kitchens above grade |
  | **KitchenQual** | Input | Ordinal | Kitchen quality |
  | **TotRmsAbvGrd** | Input | Discrete | Total rooms above grade (does not include bathrooms) |
  | **Functional** | Input | Ordinal | Home functionality (Assume typical unless deductions are warranted) |
  | **Fireplaces** | Input | Discrete | Number of fireplaces |
  | **FireplaceQu** | Input | Ordinal | Fireplace quality |
  | **GarageType** | Input | Categorical | Garage location |
  | **GarageYrBlt** | Input | Discrete | Year garage was built |
  | **GarageFinish** | Input | Ordinal | Interior finish of the garage |
  | **GarageCars** | Input | Discrete | Size of garage in car capacity |
  | **GarageArea** | Input | Continuous | Size of garage in square feet |
  | **GarageQual** | Input | Ordinal | Garage quality |
  | **GarageCond** | Input | Ordinal | Garage condition |
  | **PavedDrive** | Input | Ordinal | Paved driveway |
  | **WoodDeckSF** | Input | Continuous | Wood deck area in square feet |
  | **OpenPorchSF** | Input | Continuous | Open porch area in square feet |
  | **EnclosedPorch** | Input | Continuous | Enclosed porch area in square feet |
  | **3SsnPorch** | Input | Continuous | Three season porch area in square feet |
  | **ScreenPorch** | Input | Continuous | Screen porch area in square feet |
  | **PoolArea** | Input | Continuous | Pool area in square feet |
  | **PoolQC** | Input | Ordinal | Pool quality |
  | **Fence** | Input | Ordinal | Fence quality |
  | **MiscFeature** | Input | Categorical | Miscellaneous feature not covered in other categories |
  | **MiscVal** | Input | Continuous | $Value of miscellaneous feature |
  | **MoSold** | Input | Discrete | Month Sold (MM) |
  | **YrSold** | Input | Discrete | Year Sold (YYYY) |
  | **SaleType** | Input | Categorical | Type of sale |
  | **SaleCondition** | Input | Categorical | Condition of sale |

  * **Source of the Data**: Kaggle's "House Prices: Advanced Regression Techniques" competition dataset
  * **How training data was divided into training and validation data**: 70% training, 30% validation
  * **Number of Rows**: 1,460 in training data
    * Training rows: Training data shape: 1168
    * Validation rows: 292

### Test Data
* **Source**: Kaggle's "House Prices: Advanced Regression Techniques" test dataset
* **Number of Rows**: 1,459
* **Differences between training and test data**: The test dataset lacks the `SalePrice` column, which is the target variable in the training dataset.

### Model Details
* **Input Features**: 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
    'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl',
    'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 
    'BsmtFinType1','BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 
    'LowQualFinSF',    'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 
    'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 
    'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition', 'PropertyType'
* **Target Variable**: 'SalePrice'
* **Models Trained**: `Linear Regression`, `Ridge Regression`, `Lasso Regression`, `Elastic Net Regression`, `Random Forest Regressor` and `XGBoost Regressor`(best model) Models Trained:
* **Software**: Google Collab, Python 3.8
  * Libraries:** `pandas` , `numpy`,`scikit-learn`, `matplotlib`, `seaborn` 
  * Feature Engineering Techniques:** `Missing value imputation`, `Encoding categorical variables`, `Feature scaling and normalization`, `New feature creation` `k-fold croos validation`, `RandomSearchCV`, `ShuffleSplit` 
Cross-validation (k-fold).
Hyperparameter tuning using GridSearchCV and RandomizedSearchCV.
* **Hyperparameters**:
   Final XGBoost Hyperparameters:
  ` n_estimators`: 1000
   `max_depth`: 3
   `learning_rate`: 0.1
   `subsample`: 0.6
   `colsample_bytree`: 0.8
   `min_child_weight`: 1
   Best score found`:  28166.007315286897
  Code Snippet
```
XGBRegressor(base_score=None, booster=None, callbacks=None,
             colsample_bylevel=None, colsample_bynode=None,
             colsample_bytree=None, device=None, early_stopping_rounds=None,
             enable_categorical=False, eval_metric=None, feature_types=None,
             gamma=None, grow_policy=None, importance_type=None,
             interaction_constraints=None, learning_rate=None, max_bin=None,
             max_cat_threshold=None, max_cat_to_onehot=None,
             max_delta_step=None, max_depth=None, max_leaves=None,
             min_child_weight=None, missing=nan, monotone_constraints=None,
             multi_strategy=None, n_estimators=None, n_jobs=-1,
             num_parallel_tree=None, random_state=42, ...)
```

### Quantitative Analysis
  Models where assesed primarily with 
- **Metrics Used**: Root Mean Squared Error (RMSE), R-squared (R²), Mean Absolute Error (MAE)
- **Final Model Performance**:
   * Models:
     
     | Metric	| XGBoost (Best Model) |	Ridge Regression	| Lasso Regression	| Random Forest	| Elastic Net | 
     | ---- | ---------------- | ---------------- | ------------ | -------------- | ---------- |
     | Training RMSE	| $12,127.06	| $31,547.36	| $31,538.01	| $15,904.87	| $32,677.04 |
     | Training R² Score | 0.9767	| 0.8422	| 0.8423	| 0.9599	| 0.8307 |
     | Training MAE	| $4,054.29 |	$19,429.71	| $19,428.81	| $8,756.69	|$18,975.31 |
     | Validation RMSE	| $27,064.08 |	$34,764.12	| $34,748.47	| $27,965.45	| $35,048.92 |
     | Validation R² Score	| 0.9045	| 0.8424	| 0.8426	| 0.8980	| 0.8398 |
     | Validation MAE	| $17,940.22	| $21,697.80	| $21,721.28 |	$17,528.74	| $20,377.16 |
- **Cross-validation Results (XGBoost)**:
  `Mean RMSE`: $28,589.58
  `Standard Deviation of RMSE`: ±$3,814.78

## Potential Negative Impacts and Ethical Considerations
1. **Mathematical or Software Problems**:
   - Potential overfitting to training data
   - Bias towards features more common in the training dataset

2. **Real-world Risks**:
   - May perpetuate existing biases in housing markets
   - Could be misused to justify discriminatory pricing practices

## Uncertainties
1. **Mathematical or Software Problems**:
   - Model performance on data significantly different from the training set
   - Handling of outliers or unusual property characteristics

2. **Real-world Risks**:
   - Changes in market conditions not reflected in the training data
   - Regional differences in housing markets not captured by the model

## Unexpected Results
The model showed exceptionally high performance on the training data, with an R² score of 0.9998, which suggests potential overfitting. This unexpected result warrants further investigation and validation on larger unseen data. 

[Link to Jupyter Notebook](https://github.com/chipojaya1/housing-prices-prediction/blob/main/house_prices_model.ipynb)
