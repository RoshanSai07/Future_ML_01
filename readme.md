## Quick Start

### 1. Clone Repository

```bash
git clone <your-repository-url>
cd retail-demand-forecasting
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Dashboard

```bash
python app.py
```

### 4. Open Browser

```text
http://127.0.0.1:5000
```

---

## Dashboard Pages

### Dashboard

View:

- Revenue Analytics
- Daily & Monthly Revenue Trends
- Top Revenue Countries
- Model Performance Comparison
- Feature Importance Analysis
- Business Insights

---

### Forecasts

View:

- Revenue Forecasts
- Forecast Charts
- Forecast Tables

---

### Manual Prediction

Enter:

- Inventory
- Revenue History
- Orders
- Customers
- Quantity Sold
- Average Order Value

Get:

- Predicted Revenue
- Expected Demand
- Restocking Recommendation

---

## Retraining Models

If you want to retrain all forecasting models:

```bash
python train_model.py
```

This will regenerate:

```text
data/model_comparison.csv
data/forecast_results.csv
data/future_forecasts.csv

models/
```

and update the dashboard automatically.

---

## Technologies Used

- Python
- Flask
- Pandas
- XGBoost
- SARIMA
- AutoARIMA
- Chart.js
- HTML/CSS/JavaScript

---

## Project Goal

Build an end-to-end retail demand forecasting system that combines:

- Data Analysis
- Time Series Forecasting
- Machine Learning
- Business Intelligence
- Inventory Planning

inside a single interactive dashboard.
