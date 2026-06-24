from flask import Flask, render_template, request, jsonify

import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# ==========================================================
# LOAD FILES
# ==========================================================

comparison_df = pd.read_csv(
    "data/model_comparison.csv"
)

forecast_df = pd.read_csv(
    "data/forecast_results.csv"
)

future_df = pd.read_csv(
    "data/future_forecasts.csv"
)

# ==========================================================
# LOAD DATASET FOR ANALYTICS
# ==========================================================

retail = pd.read_excel(
    "Online Retail.xlsx"
)

retail = retail[
    retail["Quantity"] > 0
]

retail = retail[
    retail["UnitPrice"] > 0
]

retail["Revenue"] = (
    retail["Quantity"] *
    retail["UnitPrice"]
)

retail["InvoiceDate"] = pd.to_datetime(
    retail["InvoiceDate"]
)

# ==========================================================
# KPI DATA
# ==========================================================

total_revenue = round(
    retail["Revenue"].sum(),
    2
)

total_transactions = (
    retail["InvoiceNo"]
    .nunique()
)

total_customers = (
    retail["CustomerID"]
    .nunique()
)

total_products = (
    retail["StockCode"]
    .nunique()
)

# ==========================================================
# DAILY REVENUE
# ==========================================================

daily_revenue = (
    retail.groupby(
        pd.Grouper(
            key="InvoiceDate",
            freq="D"
        )
    )["Revenue"]
    .sum()
    .reset_index()
)

# ==========================================================
# MONTHLY REVENUE
# ==========================================================

monthly_revenue = (
    retail.groupby(
        pd.Grouper(
            key="InvoiceDate",
            freq="ME"
        )
    )["Revenue"]
    .sum()
    .reset_index()
)

# ==========================================================
# COUNTRY REVENUE
# ==========================================================

country_revenue = (
    retail.groupby(
        "Country"
    )["Revenue"]
    .sum()
    .sort_values(
        ascending=False
    )
    .head(10)
    .reset_index()
)

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

feature_importance = []

if os.path.exists(
    "models/xgboost.pkl"
):

    try:

        model = joblib.load(
            "models/xgboost.pkl"
        )

        features = joblib.load(
            "models/features.pkl"
        )

        feature_importance = pd.DataFrame({

            "Feature":
            features,

            "Importance":
            model.feature_importances_

        })

        feature_importance = (
            feature_importance
            .sort_values(
                by="Importance",
                ascending=False
            )
            .head(10)
        )

        feature_importance = (
            feature_importance
            .to_dict(
                "records"
            )
        )

    except:

        feature_importance = []

# ==========================================================
# BEST MODEL
# ==========================================================

best_model = comparison_df.loc[
    comparison_df["SMAPE"].idxmin()
]

# ==========================================================
# LOAD MODEL
# ==========================================================

xgb_model = None

if os.path.exists(
    "models/xgboost.pkl"
):

    xgb_model = joblib.load(
        "models/xgboost.pkl"
    )

# ==========================================================
# DASHBOARD
# ==========================================================

@app.route("/")
def dashboard():

    return render_template(

        "index.html",

        total_revenue=
        round(total_revenue),

        total_transactions=
        total_transactions,

        total_customers=
        total_customers,

        total_products=
        total_products,

        best_model=
        best_model["Model"],

        best_smape=
        best_model["SMAPE"],

        comparison=
        comparison_df.to_dict(
            "records"
        ),

        forecast_json=
        forecast_df.to_json(
            orient="records"
        ),

        daily_json=
        daily_revenue.to_json(
            orient="records"
        ),

        monthly_json=
        monthly_revenue.to_json(
            orient="records"
        ),

        country_json=
        country_revenue.to_json(
            orient="records"
        ),

        feature_json=
        feature_importance
    )

# ==========================================================
# FORECAST PAGE
# ==========================================================

@app.route("/forecast")
def forecast():

    total_forecast = round(
        future_df[
            "Forecasted Revenue"
        ].sum(),
        2
    )

    return render_template(

        "forecast.html",

        forecasts=
        future_df.to_dict(
            "records"
        ),

        total_forecast=
        total_forecast
    )

# ==========================================================
# PREDICT PAGE
# ==========================================================

@app.route("/predict")
def predict_page():

    return render_template(
        "predict.html"
    )

# ==========================================================
# API PREDICT
# ==========================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def predict():

    if xgb_model is None:

        return jsonify({

            "success": False,

            "message":
            "Model not found"

        })

    try:

        data = request.json

        inventory = float(
            data["inventory"]
        )

        revenue = float(
            data["revenue_lag_1"]
        )

        orders = float(
            data["orders_lag_1"]
        )

        customers = float(
            data["customers_lag_1"]
        )

        quantity = float(
            data["quantity_lag_1"]
        )

        aov = float(
            data["aov_lag_1"]
        )

        prediction = revenue * 1.05

        expected_demand = int(
            prediction /
            max(aov, 1)
        )

        restock = max(

            expected_demand -
            inventory,

            0
        )

        if restock > 100:

            recommendation = (
                "Urgent Restock"
            )

        elif restock > 25:

            recommendation = (
                "Monitor Inventory"
            )

        else:

            recommendation = (
                "Inventory Healthy"
            )

        return jsonify({

            "success": True,

            "predicted_revenue":
            round(
                prediction,
                2
            ),

            "expected_demand":
            expected_demand,

            "restock_needed":
            int(restock),

            "recommendation":
            recommendation

        })

    except Exception as e:

        return jsonify({

            "success": False,

            "message":
            str(e)

        })

# ==========================================================
# API ROUTES
# ==========================================================

@app.route(
    "/api/comparison"
)
def comparison_api():

    return jsonify(
        comparison_df.to_dict(
            "records"
        )
    )

@app.route(
    "/api/forecast"
)
def forecast_api():

    return jsonify(
        future_df.to_dict(
            "records"
        )
    )

# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )