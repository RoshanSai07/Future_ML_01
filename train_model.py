import os
import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
warnings.filterwarnings("ignore")

# SETUP
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
print("\nLoading Dataset...\n")

# LOAD DATA
df = pd.read_excel("Online Retail.xlsx")
df = df[df["Quantity"] > 0]
df = df[df["UnitPrice"] > 0]
df["Revenue"] = df["Quantity"] * df["UnitPrice"]
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# DAILY REVENUE
daily = (df.groupby(pd.Grouper(key="InvoiceDate",freq="D")).agg({"Revenue": "sum","InvoiceNo": "nunique","CustomerID": "nunique","StockCode": "nunique","Quantity": "sum"}).reset_index())
daily.columns = ["Date","Revenue","Orders","Customers","Products","Quantity"]
daily["AOV"] = daily["Revenue"] / daily["Orders"]
print("Daily Records:", len(daily))

# TRAIN TEST SPLIT
split = int(len(daily) * 0.8)
train_df = daily.iloc[:split].copy()
test_df = daily.iloc[split:].copy()
train_series = train_df["Revenue"]
test_series = test_df["Revenue"]

# SARIMA
print("\nTraining SARIMA...\n")
sarima = SARIMAX(train_series,order=(1, 1, 1),seasonal_order=(1, 1, 1, 7),enforce_stationarity=False,enforce_invertibility=False)
sarima_fit = sarima.fit(disp=False)
sarima_pred = sarima_fit.forecast(steps=len(test_series))

# AUTO ARIMA
print("\nTraining AutoARIMA...\n")
auto_arima = pm.auto_arima(train_series,seasonal=True,m=7,trace=False,error_action="ignore",suppress_warnings=True)
auto_pred = auto_arima.predict(n_periods=len(test_series))

# XGBOOST FEATURE ENGINEERING
xgb_df = daily.copy()
xgb_df["DayOfWeek"] = xgb_df["Date"].dt.dayofweek
xgb_df["DayOfMonth"] = xgb_df["Date"].dt.day
xgb_df["Month"] = xgb_df["Date"].dt.month
xgb_df["WeekOfYear"] = xgb_df["Date"].dt.isocalendar().week.astype(int)
xgb_df["Weekend"] = (xgb_df["DayOfWeek"] >= 5).astype(int)
xgb_df["IsMonthStart"] = (xgb_df["Date"].dt.is_month_start).astype(int)
xgb_df["IsMonthEnd"] = (xgb_df["Date"].dt.is_month_end).astype(int)
xgb_df["Day_Sin"] = np.sin(2 * np.pi * xgb_df["DayOfWeek"] / 7)
xgb_df["Day_Cos"] = np.cos(2 * np.pi * xgb_df["DayOfWeek"] / 7)
for lag in [1, 2, 3, 7, 14, 21, 28]:
    xgb_df[f"Revenue_Lag_{lag}"] = (xgb_df["Revenue"].shift(lag))

for lag in [1, 7, 14]:
    xgb_df[f"Orders_Lag_{lag}"] = (xgb_df["Orders"].shift(lag))
    xgb_df[f"Customers_Lag_{lag}"] = (xgb_df["Customers"].shift(lag))
    xgb_df[f"Quantity_Lag_{lag}"] = (xgb_df["Quantity"].shift(lag))
    xgb_df[f"AOV_Lag_{lag}"] = (xgb_df["AOV"].shift(lag))

xgb_df["Revenue_Rolling_7"] = (xgb_df["Revenue"].shift(1).rolling(7).mean())
xgb_df["Revenue_Rolling_14"] = (xgb_df["Revenue"].shift(1).rolling(14).mean())
xgb_df["Revenue_Rolling_30"] = (xgb_df["Revenue"].shift(1).rolling(30).mean())
xgb_df["Revenue_Std_7"] = (xgb_df["Revenue"].shift(1).rolling(7).std())
xgb_df["Revenue_EMA_7"] = (xgb_df["Revenue"].shift(1).ewm(span=7).mean())
xgb_df["Revenue_EMA_14"] = (xgb_df["Revenue"].shift(1).ewm(span=14).mean())
xgb_df = xgb_df.dropna()

FEATURES = ["DayOfWeek","DayOfMonth","Month","WeekOfYear","Weekend","IsMonthStart","IsMonthEnd","Day_Sin","Day_Cos","Revenue_Lag_1","Revenue_Lag_2","Revenue_Lag_3",
    "Revenue_Lag_7","Revenue_Lag_14","Revenue_Lag_21","Revenue_Lag_28","Orders_Lag_1","Orders_Lag_7","Orders_Lag_14","Customers_Lag_1","Customers_Lag_7","Customers_Lag_14","Quantity_Lag_1",
    "Quantity_Lag_7","Quantity_Lag_14","AOV_Lag_1", "AOV_Lag_7","AOV_Lag_14","Revenue_Rolling_7","Revenue_Rolling_14","Revenue_Rolling_30","Revenue_Std_7","Revenue_EMA_7","Revenue_EMA_14"]
target = "Revenue"
split_xgb = int(len(xgb_df) * 0.8)
X_train = xgb_df[FEATURES].iloc[:split_xgb]
X_test = xgb_df[FEATURES].iloc[split_xgb:]
y_train = xgb_df[target].iloc[:split_xgb]
y_test = xgb_df[target].iloc[split_xgb:]
xgb_dates = xgb_df["Date"].iloc[split_xgb:]

# XGBOOST
print("\nTraining XGBoost...\n")
xgb_model = XGBRegressor(objective="reg:squarederror",n_estimators=1000,learning_rate=0.03,max_depth=6,min_child_weight=2,subsample=0.85,colsample_bytree=0.85,random_state=42)
xgb_model.fit(X_train,y_train)
xgb_pred = xgb_model.predict(X_test)

# ALIGN WINDOWS
common_len = min(len(test_series),len(xgb_pred))
actual = test_series.iloc[-common_len:].values
actual_dates = (test_df["Date"].iloc[-common_len:].reset_index(drop=True))
sarima_pred = np.array(sarima_pred[-common_len:])
auto_pred = np.array(auto_pred[-common_len:])
xgb_pred = np.array(xgb_pred[-common_len:])

# ENSEMBLE
ensemble_pred = (0.2 * sarima_pred +0.3 * auto_pred +0.5 * xgb_pred)

# METRICS
def smape(actual, pred):
    return (100 *np.mean(2 *np.abs(pred - actual)/(np.abs(actual) +np.abs(pred))))

results = []
models = {"SARIMA": sarima_pred,"AutoARIMA": auto_pred,"XGBoost": xgb_pred,"Ensemble": ensemble_pred}
for name, pred in models.items():
    mae = mean_absolute_error(actual,pred)
    rmse = np.sqrt(mean_squared_error(actual,pred))
    s = smape(actual,pred)
    results.append({"Model": name,"MAE": round(mae, 2),"RMSE": round(rmse, 2),"SMAPE": round(s, 2)})
results_df = pd.DataFrame(results)
print("\nMODEL COMPARISON\n")
print(results_df)

# BEST MODEL
winner = results_df.loc[results_df["SMAPE"].idxmin()]
best_model_name = winner["Model"]
print("\nBEST MODEL\n")
print(winner)

# FORECAST RESULTS
forecast_results = pd.DataFrame({"Date": actual_dates,"Actual": actual,"SARIMA": sarima_pred,"AutoARIMA": auto_pred,"XGBoost": xgb_pred,"Ensemble": ensemble_pred})
forecast_results.to_csv("data/forecast_results.csv",index=False)
results_df.to_csv("data/model_comparison.csv",index=False)

# RETRAIN ON FULL DATA
print("\nRetraining Best Model On Full Dataset...\n")
future_days = 30
future_dates = pd.date_range(start=daily["Date"].max() + pd.Timedelta(days=1),periods=future_days,freq="D")

# FUTURE FORECAST
if best_model_name == "SARIMA":
    final_model = SARIMAX(daily["Revenue"],order=(1,1,1),seasonal_order=(1,1,1,7),enforce_stationarity=False,enforce_invertibility=False)
    final_fit = final_model.fit(disp=False)
    future_values = final_fit.forecast(future_days)
    joblib.dump(final_fit,"models/sarima.pkl")
elif best_model_name == "AutoARIMA":
    final_fit = pm.auto_arima(daily["Revenue"],seasonal=True,m=7,trace=False)
    future_values = final_fit.predict(n_periods=future_days)
    joblib.dump(final_fit,"models/autoarima.pkl")
else:
    joblib.dump(xgb_model,"models/xgboost.pkl")
    future_values = []
    history = xgb_df.copy()
    for future_date in future_dates:
        row = {}
        row["Date"] = future_date
        row["DayOfWeek"] = future_date.dayofweek
        row["DayOfMonth"] = future_date.day
        row["Month"] = future_date.month
        row["WeekOfYear"] = future_date.isocalendar().week
        row["Weekend"] = int(future_date.dayofweek >= 5)
        row["IsMonthStart"] = int(future_date.is_month_start)
        row["IsMonthEnd"] = int(future_date.is_month_end)
        row["Day_Sin"] = np.sin(2 * np.pi *row["DayOfWeek"] / 7)
        row["Day_Cos"] = np.cos(2 * np.pi *row["DayOfWeek"] / 7)
        revenue_history = history["Revenue"].tolist()
        row["Revenue_Lag_1"] = revenue_history[-1]
        row["Revenue_Lag_2"] = revenue_history[-2]
        row["Revenue_Lag_3"] = revenue_history[-3]
        row["Revenue_Lag_7"] = revenue_history[-7]
        row["Revenue_Lag_14"] = revenue_history[-14]
        row["Revenue_Lag_21"] = revenue_history[-21]
        row["Revenue_Lag_28"] = revenue_history[-28]
        row["Orders_Lag_1"] = history["Orders"].iloc[-1]
        row["Orders_Lag_7"] = history["Orders"].iloc[-7]
        row["Orders_Lag_14"] = history["Orders"].iloc[-14]
        row["Customers_Lag_1"] = history["Customers"].iloc[-1]
        row["Customers_Lag_7"] = history["Customers"].iloc[-7]
        row["Customers_Lag_14"] = history["Customers"].iloc[-14]
        row["Quantity_Lag_1"] = history["Quantity"].iloc[-1]
        row["Quantity_Lag_7"] = history["Quantity"].iloc[-7]
        row["Quantity_Lag_14"] = history["Quantity"].iloc[-14]
        row["AOV_Lag_1"] = history["AOV"].iloc[-1]
        row["AOV_Lag_7"] = history["AOV"].iloc[-7]
        row["AOV_Lag_14"] = history["AOV"].iloc[-14]
        row["Revenue_Rolling_7"] = (history["Revenue"].tail(7).mean())
        row["Revenue_Rolling_14"] = (history["Revenue"].tail(14).mean())
        row["Revenue_Rolling_30"] = (history["Revenue"].tail(30).mean())
        row["Revenue_Std_7"] = (history["Revenue"].tail(7).std())
        row["Revenue_EMA_7"] = (history["Revenue"].ewm(span=7).mean().iloc[-1])
        row["Revenue_EMA_14"] = (history["Revenue"].ewm(span=14).mean().iloc[-1])
        X_future = pd.DataFrame([row])[FEATURES]
        pred = float(xgb_model.predict(X_future)[0])
        pred = max(pred, 0)
        future_values.append(pred)
        history = pd.concat([
            history,
            pd.DataFrame([{"Date": future_date,"Revenue": pred,"Orders": history["Orders"].tail(7).mean(),"Customers": history["Customers"].tail(7).mean(),"Products": history["Products"].tail(7).mean(),
                "Quantity": history["Quantity"].tail(7).mean(),"AOV": history["AOV"].tail(7).mean()}])], ignore_index=True)
    future_values = np.array(future_values)
future_forecast = pd.DataFrame({"Date": future_dates,"Forecasted Revenue": future_values})
future_forecast.to_csv("data/future_forecasts.csv",index=False)
joblib.dump(FEATURES,"models/features.pkl")
print("\nFiles Created Successfully\n")
print("data/model_comparison.csv")
print("data/forecast_results.csv")
print("data/future_forecasts.csv")
print("\nTraining Complete.")