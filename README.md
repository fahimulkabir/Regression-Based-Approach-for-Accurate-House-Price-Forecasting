# 🏠 Housing Price Prediction Using Machine Learning

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Status](https://img.shields.io/badge/status-Completed-success)
![Python](https://img.shields.io/badge/python-3.8%2B-green)

---

## 📄 Overview

This project focuses on building a machine learning model to predict median housing prices in California based on historical census data. The solution aims to support real estate valuation, investment decision-making, and urban planning initiatives using interpretable and scalable predictive models.

---

## 🚀 Project Design

The project follows a standard machine learning workflow:

- **Data Exploration**: Understanding distributions, relationships, and variability in features.
- **Feature Engineering**: Creating new meaningful features like `RoomsPerHousehold`, `BedroomsPerRoom`, and `PopulationPerHousehold`.
- **Data Preprocessing**: Standardization and dataset splitting (80% train / 20% test).
- **Model Selection**: Comparison between Linear Regression and Random Forest Regressor.
- **Model Tuning**: Hyperparameter tuning using GridSearchCV for Random Forest.
- **Evaluation**: Metrics like Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²).
- **Interpretation**: Analyzing feature importance and spatial trends.

---

## 📂 Dataset

- **Source**: Scikit-learn's California Housing Dataset
- **Total Entries**: 20,640
- **Features**: Median Income, House Age, Average Rooms, Average Bedrooms, Population, Average Occupancy, Latitude, Longitude, Median House Value (Target)
- **Missing Values**: None
- **Type**: All numerical (float64)

---

## ⚙️ Implementation Steps

1. Load and explore the California Housing dataset.
2. Conduct data visualization (correlation matrix, scatterplots, histograms).
3. Perform feature engineering to create new informative variables.
4. Split data into training and testing sets.
5. Scale features using `StandardScaler`.
6. Train Linear Regression and Random Forest models.
7. Apply GridSearchCV to optimize Random Forest hyperparameters.
8. Evaluate models using multiple performance metrics.
9. Visualize results and feature importances.

---

## 📊 Evaluation Results

| Model                   | MSE    | MAE    | R²     |
|--------------------------|--------|--------|--------|
| Linear Regression        | 0.4540 | 0.4874 | 0.6535 |
| Random Forest (Untuned)  | 0.2561 | 0.3299 | 0.8046 |
| Random Forest (Tuned)    | 0.2549 | 0.3284 | 0.8054 |

✅ **Tuned Random Forest achieved the best performance**, with an R² score above 0.80.

---

## 📈 Visualizations

- **Correlation Heatmap**: Strong positive correlation between Median Income and House Value.
- **Geographic Scatterplot**: High-value homes concentrated along the California coastline.
- **Feature Importance**: Income level, location (Latitude, Longitude), and average rooms significantly influenced prices.
- **Histograms**: Several features (e.g., Median Income, Population) showed skewness, influencing preprocessing choices.

---

## 💡 Key Insights

- Median Income is the strongest predictor of house prices.
- Coastal regions exhibit significantly higher house values.
- Feature engineering and standardization enhanced model performance.
- Hyperparameter tuning, while yielding modest improvement, stabilized model predictions.
- Tree-based ensemble models (Random Forest) significantly outperform linear methods in capturing housing price dynamics.

---

## 🛠️ Future Improvements

- Implement Gradient Boosting and XGBoost models.
- Incorporate geospatial feature transformations (distance to coast, clustering).
- Apply SHAP values for model interpretability.
- Explore fairness and bias issues in housing predictions.
- Deploy the final model via Streamlit or Flask web app.

---

## 📢 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/fahimulkabir/Regression-Based-Approach-for-Accurate-House-Price-Forecasting.git
cd Regression-Based-Approach-for-Accurate-House-Price-Forecasting

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Jupyter Notebook
jupyter notebook
