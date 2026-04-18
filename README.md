# 🏙️ Dubai Real Estate Valuation AI (XAI + MLOps)

![MLOps Pipeline](https://github.com/N-KATOCH/dubai_real_estate_predictor/actions/workflows/main.yml/badge.svg)

An end-to-end Machine Learning system designed to predict property prices in Dubai with high accuracy (96%+) while providing transparent, dirham-based explanations for every valuation.

## 🚀 Business Impact
- **Monetary Attribution:** Converts complex SHAP values into actual AED impacts, allowing agents to explain price drivers (e.g., "The sea view adds 200k AED").
- **Investment Signals:** Identifies undervalued deals by calculating the gap between market listing price and AI-predicted valuation.
- **NLP Insights:** Extracts value from property descriptions to quantify the ROI of features like "Upgraded Kitchen" or "Beach Access."

## 🛠️ Tech Stack
- **Modeling:** LightGBM (Gradient Boosting) with Log-transformation for skewed price data.
- **Explainability (XAI):** SHAP (Shapley Additive Explanations) vectorized for AED currency conversion.
- **MLOps:** MLflow for experiment tracking, model versioning, and lifecycle management.
- **Engineering:** Modular Python architecture (`src/` structure) for production scalability.

## 📂 Project Structure
- `src/preprocessing.py`: Modular NLP cleaning and feature engineering logic.
- `src/train.py`: Model training pipeline, metric logging, and SHAP-to-AED attribution engine.
- `requirements.txt`: Managed environment dependencies for reproducibility.

## 📊 Sample Results
The system provides a "Price Bridge" for each listing:
| Feature | Impact (AED) |
| :--- | :--- |
| **Base Price** | 2,500,000 |
| **Size Premium** | +450,000 |
| **Community (Palm Jumeirah)** | +1,200,000 |
| **Lack of Pool** | -150,000 |
| **Final AI Valuation** | **4,000,000** |
