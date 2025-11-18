# 🏠 House Price Prediction - Regression Modelling Workflow

An end-to-end machine learning project to develop a reliable, interpretable, and production-ready regression model for predicting house prices in the Indian real estate market. This repository demonstrates a complete data science workflow, from exploratory data analysis and baselines to advanced regularization and model deployment artifacts.

---

## 🚀 Executive Summary

The primary objective was to build a robust model for predicting house prices (`Price`) using a diverse set of property features.

| Metric | Champion Model | Value |
| :--- | :--- | :--- |
| **Champion Model** | **Lasso Regression** | |
| Test RMSE | Lasso Regression | **212,586.74** |
| Test Adjusted R² | Lasso Regression | **0.6775** |
| Multicollinearity | Handled with Regularization | VIF Inf. for area features |

The **Lasso Regression** model (with `alpha=1000`) was selected as the champion due to achieving the **lowest Test RMSE** and the **highest Adjusted R²** among all evaluated models.

---

## 🎯 Project Objective

To build a complete predictive modeling pipeline for house price prediction, adhering to professional data science practices:

* **Establish Baselines**: Compare performance against simple heuristics (Mean and Median).
* **Regression Models**: Implement Simple Linear Regression (SLR), Multiple Linear Regression (MLR), Ridge, Lasso, and ElasticNet.
* **Diagnostics**: Use RMSE, MAE, R², Adjusted R², and MAPE, along with Variance Inflation Factor (VIF) and residual analysis.
* **Champion Selection**: Select and justify the best-performing model based on evaluation metrics and stability.
* **Production Readiness**: Export the model and necessary artifacts for deployment.

---

## 🛠️ Methodology & Workflow

### 1. Data Preprocessing & Feature Engineering
* **Data Source**: `HousePriceIndia.csv` (not included in the repository for privacy).
* **Features Used**: 16 selected features, including:
    * `number of bedrooms`, `number of bathrooms`, `lot area`, `number of floors`
    * `waterfront present`, `number of views`, `condition of the house`, `grade of the house`
    * `Area of the house(excluding basement)`, `Area of the basement`
    * `Number of schools nearby`, `Distance from the airport`
* **Engineered Features**:
    * `house_age` (2025 - `Built Year`)
    * `is_renovated` (Binary: 1 if `Renovation Year` > 0)
    * `total_area` (`living area` + `Area of the basement`)
* **Scaling**: `StandardScaler` was applied to all feature columns.
* **Split**: 80% Train, 20% Test (`random_state=42`).

### 2. Model Diagnostics
* **Multicollinearity**: VIF analysis showed **infinite VIF** for the area-related features (`living area`, `total_area`, `Area of the basement`, `Area of the house(excluding basement)`), confirming high **multicollinearity**. This result explicitly justified the use of regularization techniques (Ridge, Lasso, ElasticNet) to stabilize the models.
* **Residual Analysis**: The **Residual Distribution for the Ridge Model** (a very close runner-up to Lasso) showed a distribution centered near zero, indicating a well-fitted model without significant systematic bias.

### 3. Model Evaluation & Selection

Models were ranked primarily by the **lowest Test RMSE** and secondarily by the **highest Adjusted R²**. Cross-Validation (5-Fold RMSE) confirmed the stability of the linear and regularized models.

| Rank | Model | Test RMSE | Test MAE | R² | Adjusted R² | MAPE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Lasso Regression** | **212,586.74** | **137,889.13** | **0.6793** | **0.6775** | **28.47%** |
| 2 | Ridge Regression | 212,733.40 | 138,058.81 | 0.6789 | 0.6771 | 28.50% |
| 3 | Multiple Linear Regression | 212,778.42 | 138,268.83 | 0.6787 | 0.6770 | 28.56% |
| 4 | Simple Linear Regression | 259,221.01 | 172,726.52 | 0.5232 | 0.5230 | 35.40% |
| 5 | ElasticNet Regression | 371,544.37 | 234,959.91 | 0.0204 | 0.0150 | 52.60% |
| 6 | Mean Baseline | 375,435.65 | 237,863.70 | -0.0002 | -0.0002 | 53.29% |
| 7 | Median Baseline | 386,867.04 | 226,052.41 | -0.0621 | -0.0621 | 42.52% |

The **Lasso Regression** model achieved superior performance, demonstrating the effectiveness of the L1 regularization penalty in handling the highly correlated features identified by VIF.

---

## 📦 Deployment Artifacts

The following production-ready artifacts are saved in the project directory, enabling the seamless deployment of the champion model:

* `champion_model.joblib`: The serialized **Lasso Regression** model object.
* `scaler.joblib`: The fitted `StandardScaler` object, required to preprocess new input data before prediction.
* `houseprice_predictions_test.csv`: A CSV file containing the actual prices, predicted prices, and residuals for the test set.
* `test_metrics.json`: A JSON file summarizing the champion model's key performance metrics on the test set.
* `model_leaderboard.csv`: A CSV file containing the full comparison table for all models.

### Prediction Pipeline Flow

```mermaid
graph TD
    A[New Property Data] --> B{Feature Engineering};
    B --> C{Load scaler.joblib};
    C --> D[StandardScaler Transform];
    D --> E{Load champion_model.joblib};
    E --> F[Predict Price];
    F --> G[Output Predicted Price];
