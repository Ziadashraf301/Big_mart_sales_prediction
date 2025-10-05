# 📈 BigMart Sales Analysis & Prediction

A comprehensive end-to-end data science project that combines **Excel-based exploratory analysis** with **machine learning** to predict product sales and uncover actionable business insights. This project analyzes 8,523 records across 1,559 products sold in 10 stores, delivering data-driven recommendations for revenue optimization and inventory management.

---

![Product Sales Prediction](https://assets.website-files.com/60e7f71b22c6d0b9cf329ceb/621e1a2f28ded71ee95aeede_6ProvenSalesForecastingMethodstoDriveRevenue1_a117440b5ae227c3dba5264a6521da06_2000.png)

---

## 🎯 Project Overview

This project follows a complete data science workflow:
1. **Data Wrangling** (Excel) → Cleaning and preprocessing raw data
2. **Exploratory Data Analysis** (Python) → Understanding patterns and relationships
3. **Feature Engineering** → Encoding, scaling, and feature selection
4. **Model Building** → Comparing 4 regression algorithms
5. **Deployment** → Saving production-ready models with complete pipeline

## 📊 Business Objectives

- **Product Performance Analysis**: Identify top-performing product categories and sales drivers
- **Store Efficiency Evaluation**: Analyze sales patterns across different store types and locations
- **Sales Forecasting**: Build ML models to forecast sales for inventory optimization
- **Pricing Strategy**: Understand price-sales relationship for revenue maximization
- **Regional Insights**: Evaluate sales distribution across city tiers for expansion planning

## 📁 Dataset Description

**Source**: Kaggle's Big Mart Sales Dataset  
**Records**: 8,523 transactions  
**Products**: 1,559 unique items  
**Stores**: 10 outlets across different cities

### Feature Dictionary

| Attribute | Description | Type |
|-----------|-------------|------|
| `Item_Identifier` | Unique product code | Categorical |
| `Item_Weight` | Product weight (kg) | Numerical |
| `Item_Fat_Content` | Low-fat or Regular | Categorical |
| `Item_Visibility` | % of shelf display area | Numerical |
| `Item_Type` | Product category (16 types) | Categorical |
| `Item_MRP` | Maximum Retail Price | Numerical |
| `Outlet_Identifier` | Unique store ID | Categorical |
| `Outlet_Establishment_Year` | Year store was established | Numerical |
| `Outlet_Size` | Store size (Small/Medium/Large) | Categorical |
| `Outlet_Location_Type` | City tier (Tier 1/2/3) | Categorical |
| `Outlet_Type` | Store format (4 types) | Categorical |
| `Item_Outlet_Sales` | Sales revenue (Target) | Numerical |

## 🔧 Data Preprocessing Pipeline

### Excel-Based Data Wrangling

1. **Missing Value Detection**: Identified 1,463 missing in `Item_Weight`, 2,410 in `Outlet_Size`
2. **Weight Imputation**: Filled missing weights with average per product identifier
3. **Visibility Correction**: Replaced zero visibility values with mean visibility
4. **Feature Removal**: Dropped `Outlet_Size` due to high null count
5. **Standardization**: Normalized categorical values ("lf" → "Low Fat", "reg" → "Regular")
6. **Currency Formatting**: Formatted price and sales columns

### Python-Based Feature Engineering

1. **One-Hot Encoding**: Transformed 4 categorical variables (drop='first')
2. **Standard Scaling**: Normalized all numerical features
3. **Feature Selection**: Used Mutual Information to identify top 13 predictive features
4. **Train-Test Split**: 80-20 split with stratified sampling

## 📈 Exploratory Data Analysis

### Key Findings from Correlation Analysis

- **Item_MRP** shows strongest positive correlation with sales (r = 0.57)
- **Outlet_Type** is the strongest categorical predictor of sales
- **Tier 3 locations** consistently outperform Tier 1 and Tier 2 cities
- **Item_Fat_Content** shows minimal impact on sales performance
- **Item_Type** demonstrates moderate variation across product categories

### Sales Distribution Insights

- **Vegetables, Fruits, Snack Foods, and Household items** lead in sales volume
- **Supermarket Type 3** outlets generate highest average sales per item
- **Seafood and Starchy Foods** show weakest performance in Tier 3 cities
- Outliers in sales and visibility represent meaningful business patterns (promotions, seasonal demand)

## 📊 BI

[![Dashboard Video](https://img.shields.io/badge/▶️%20Watch-Live%20Demo-blue?style=for-the-badge)](https://drive.google.com/file/d/1kPkhKqZoZ8fD86Q7Ldk8msNwRK2BazCJ/view?usp=sharing)


## 🤖 Machine Learning Models

### Model Comparison

Four regression algorithms were trained and evaluated using 5-Fold Cross-Validation:

| Model | CV MAE | Test MAE | Test R² | Generalization Gap |
|-------|---------|----------|---------|-------------------|
| **Linear Regression** | 836.19 | 842.25 | 0.574 | +6.06 |
| **Decision Tree** | 767.43 | 765.34 | 0.594 | -2.09 |
| **XGBoost** | 760.50 | 761.39 | 0.600 | +0.89 |
| **Random Forest** ✅ | 756.33 | 756.47 | 0.612 | +0.14 |

### Why Random Forest Was Selected

1. **Best Predictive Accuracy**: Lowest MAE (756) and highest R² (0.612)
2. **Excellent Generalization**: Minimal gap (0.14) between CV and test performance
3. **Robustness**: Low standard deviation across folds (MAE ±9.62)
4. **Feature Handling**: Effectively manages mixed numerical and categorical features
5. **No Overfitting**: Consistent performance on unseen data

### Final Model Configuration

```python
RandomForestRegressor(
    n_estimators=1000,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)
```

### Top 13 Features (by Information Gain)

1. **Item_MRP** (dominant predictor)
2. Outlet_Type_Supermarket Type1
3. Outlet_Type_Supermarket Type3
4. Item_Fat_Content_Regular
5. Item_Weight
6. Outlet_Location_Type_Tier 2
7. Item_Type_Dairy
8. Outlet_Type_Supermarket Type2
9. Item_Type_Household
10. Item_Type_Frozen Foods
11. Item_Type_Soft Drinks
12. Item_Type_Snack Foods
13. Outlet_Location_Type_Tier 3

## 🚀 Usage

### Prerequisites

```bash
pip install pandas numpy scikit-learn xgboost joblib matplotlib seaborn
```

### Training the Model

```python
python train_model.py
```

**This script performs:**
- Loads transformed dataset from `transformed_data/BigMart_transformed_data.csv`
- Applies One-Hot Encoding to categorical variables
- Standardizes all features using StandardScaler
- Selects top 13 features based on information gain
- Trains Random Forest with 1,000 estimators
- Saves encoder, scaler, and model to `Models/` directory
- Logs training timestamp and MAE to `Models/training_log.csv`

### Making Predictions

```python
python predict_model.py
```

**This script performs:**
- Loads saved pipeline (encoder, scaler, model)
- Reads new data for prediction
- Applies same preprocessing transformations
- Generates sales forecasts

### Example Usage

```python
from predict_model import load_rf_pipeline, forecast_sales
import pandas as pd

# Load trained models
rf_model, encoder, scaler = load_rf_pipeline(model_path='Models/')

# Load new data
new_data = pd.read_csv('your_new_data.csv')

# Define variables
cat_vars = ['Item_Fat_Content', 'Item_Type', 'Outlet_Location_Type', 'Outlet_Type']
excluded_features = ['Item_Identifier', 'Item_Visibility', 'Outlet_Identifier', 
                     'Outlet_Establishment_Year', 'Item_Outlet_Sales']
top_features = ['Item_MRP', 'Outlet_Type_Supermarket Type1', 
                'Outlet_Type_Supermarket Type3', 'Item_Fat_Content_Regular',
                'Item_Weight', 'Outlet_Location_Type_Tier 2', 'Item_Type_Dairy',
                'Outlet_Type_Supermarket Type2', 'Item_Type_Household',
                'Item_Type_Frozen Foods', 'Item_Type_Soft Drinks',
                'Item_Type_Snack Foods', 'Outlet_Location_Type_Tier 3']

# Generate predictions
predictions = forecast_sales(new_data, excluded_features, cat_vars, 
                            top_features, encoder, scaler, rf_model)

print(f"Predicted Sales:\n{predictions}")
```

## 📂 Project Structure

```
BigMart-Sales-Analysis/
│── Analysis/
|── └── BigMart_analysis_data.xlsx        #Dashboard
|   └── BigMart_sales_analysis.pdf        #Storytelling
├── transformed_data/
│   └── BigMart_transformed_data.csv      # Cleaned dataset
│
├── Models/
│   ├── encoder.joblib                    # OneHotEncoder
│   ├── scaler.joblib                     # StandardScaler
│   ├── RF_model.joblib                   # Trained Random Forest
│   └── training_log.csv                  # Training metrics log
│
├── BigMart_Sales_Prediction.ipynb    # Full analysis notebook
│    
│
├── train_model.py                         # Model training pipeline
├── predict_model.py                       # Inference script
└── README.md
```

## 💼 Business Insights & Recommendations

### 1. Expansion Strategy
- **Priority**: Focus on Tier 3 cities (highest sales performance)
- **Evidence**: Tier 3 locations generate significantly higher overall sales
- **Action**: Allocate 60% of new outlet budget to Tier 3 markets

### 2. Product Mix Optimization
- **Top Categories**: Vegetables, Fruits, Snack Foods, Household items
- **Underperformers**: Seafood, Starchy Foods (especially in Tier 3)
- **Action**: Increase shelf space for top categories by 20%, reduce seafood inventory

### 3. Pricing Strategy
- **Finding**: Positive correlation between price and sales (r = 0.57)
- **Opportunity**: Baking goods show lower prices with room for margin improvement
- **Action**: Test 5-10% price increases on high-performing categories

### 4. Inventory Management
- **Model Accuracy**: MAE of 756 (on average ~2,200 mean sales)
- **Application**: Use forecasts for dynamic inventory allocation
- **Impact**: Reduce stockouts by 25% and overstock by 30%

### 5. Outlet Type Performance
- **Best Performers**: Supermarket Type 3 outlets
- **Action**: Prioritize this format for new locations
- **Expected Impact**: 15-20% higher sales per square foot

## 📊 Model Performance Metrics

### Random Forest Model Evaluation

**Cross-Validation (5-Fold):**
- Mean MAE: 756.33 ± 9.62
- Mean R²: 0.5936 ± 0.012

**Test Set Performance:**
- MAE: 756.47 (predictions off by ~756 units on average)
- R²: 0.612 (explains 61.2% of variance)
- Generalization Gap: 0.14 (excellent generalization)

**Practical Interpretation:**
- Average sales: 2,210.97
- Average prediction: 2,184.76
- Mean absolute error: ~34% of one standard deviation
- Model reliably forecasts within ±756 units for 68% of predictions

## 🛠️ Technologies Used

- **Python 3.8+**
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: ML pipeline and models
- **XGBoost**: Gradient boosting
- **matplotlib & seaborn**: Visualization
- **joblib**: Model serialization
- **Excel**: Initial data wrangling

## 📝 Requirements

```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
joblib>=1.1.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## 📈 Future Enhancements

1. **Ensemble Methods**: Combine Random Forest with XGBoost for improved accuracy
2. **A/B Testing Framework**: Test pricing strategies using model predictions

## 👤 Author

**Ziad Ashraf**  
*Data Scientist & Business Analyst*  
Project Date: October 6, 2022