import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

#LOAD DATA
print("\nLoading Online Retail Dataset...\n")
data = pd.read_excel("Online Retail.xlsx")
print("Dataset Shape:")
print(data.shape)
print("\nColumns:")
print(data.columns.tolist())
print("\nFirst 5 Rows:")
print(data.head())

#DATA INFORMATION
print("\nDataset Info:\n")
print(data.info())
print("\nMissing Values:\n")
print(data.isnull().sum())

#CONVERT DATE COLUMN
data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])

#REMOVE RETURNS & INVALID RECORDS
print("\nRemoving Returns and Invalid Transactions...\n")
initial_rows = len(data)
data = data[data["Quantity"] > 0]
data = data[data["UnitPrice"] > 0]
removed_rows = initial_rows - len(data)
print(f"Rows Removed: {removed_rows}")
print(f"Rows Remaining: {len(data)}")

#CREATE REVENUE FEATURE
data["Revenue"] = (data["Quantity"]*data["UnitPrice"])
print("\nRevenue Feature Created\n")
print(data[["Quantity","UnitPrice","Revenue"]].head())

#BASIC REVENUE STATISTICS
print("\nRevenue Statistics:\n")
print(data["Revenue"].describe())

#REVENUE DISTRIBUTION
plt.figure(figsize=(12,5))
plt.hist(np.log1p(data["Revenue"]),bins=100)
plt.title("Log Revenue Distribution")
plt.xlabel("log(Revenue + 1)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

#TOP COUNTRIES
country_revenue = (data.groupby("Country")["Revenue"].sum().sort_values(ascending=False).head(10))
print("\nTop 10 Countries by Revenue:\n")
print(country_revenue)
plt.figure(figsize=(12,5))
country_revenue.plot(kind="bar")
plt.title("Top 10 Countries by Revenue")
plt.ylabel("Revenue")
plt.show()

#TOP PRODUCTS
top_products = (data.groupby("Description")["Revenue"].sum().sort_values(ascending=False).head(10))
print("\nTop 10 Products:\n")
print(top_products)
plt.figure(figsize=(12,5))
top_products.plot(kind="bar")
plt.title("Top 10 Products by Revenue")
plt.ylabel("Revenue")
plt.show()

#MONTHLY REVENUE
monthly_revenue = (data.groupby(pd.Grouper(key="InvoiceDate",freq="ME"))["Revenue"].sum().reset_index())
print("\nMonthly Revenue:\n")
print(monthly_revenue.head())
print(monthly_revenue.tail())
print(f"\nTotal Monthly Observations: "f"{len(monthly_revenue)}")
plt.figure(figsize=(14,5))
plt.plot(monthly_revenue["InvoiceDate"],monthly_revenue["Revenue"],linewidth=2)
plt.title("Monthly Revenue Trend")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.grid(True)
plt.show()

#DAILY REVENUE
daily_revenue = (data.groupby(pd.Grouper(key="InvoiceDate",freq="D"))["Revenue"].sum().reset_index())
print(f"\nDaily Observations: "f"{len(daily_revenue)}")
plt.figure(figsize=(14,5))
plt.plot(daily_revenue["InvoiceDate"],daily_revenue["Revenue"],linewidth=1)
plt.title("Daily Revenue Trend")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.grid(True)
plt.show()

#WEEKLY REVENUE
data["DayOfWeek"] = (data["InvoiceDate"].dt.day_name())
weekly_sales = (data.groupby("DayOfWeek")["Revenue"].mean())
print(weekly_sales)

weekly_revenue = (data.groupby(pd.Grouper(key="InvoiceDate", freq="W"))["Revenue"].sum().reset_index())
print(f"\nWeekly Observations: {len(weekly_revenue)}")
plt.figure(figsize=(14, 5))
plt.plot(weekly_revenue["InvoiceDate"], weekly_revenue["Revenue"], linewidth=1.5, color='orange')
plt.title("Weekly Revenue Trend")
plt.xlabel("Date (Weeks)")
plt.ylabel("Revenue")
plt.grid(True)
plt.show()

#OUTLIER DETECTION
z_scores = np.abs((daily_revenue["Revenue"]-daily_revenue["Revenue"].mean())/daily_revenue["Revenue"].std())
outliers = daily_revenue[z_scores > 3]
print("\nOutliers Found:\n")
print(outliers)

#TIME SERIES DECOMPOSITION
daily_series = (daily_revenue.set_index("InvoiceDate")["Revenue"])
decomposition = seasonal_decompose(daily_series,model="additive",period=30)
decomposition.plot()
plt.show()

#ADF STATIONARITY TEST
print("\nADF Stationarity Test\n")
adf_result = adfuller(daily_series.dropna())
print("ADF Statistic:",adf_result[0])
print("P-Value:",adf_result[1])
if adf_result[1] < 0.05:
    print("\nResult: Series is Stationary")
else:
    print("\nResult: Series is NOT Stationary")

#TRAIN TEST SPLIT
split_index = int(len(daily_revenue) * 0.8)
train = daily_revenue.iloc[:split_index]
test = daily_revenue.iloc[split_index:]
plt.figure(figsize=(14,6))
plt.plot(train["InvoiceDate"],train["Revenue"],label="Training Data")
plt.plot(test["InvoiceDate"],test["Revenue"],label="Testing Data")
plt.axvline(x=test["InvoiceDate"].iloc[0],color="red",linestyle="--",label="Train/Test Split")
plt.title("Train/Test Split")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.legend()
plt.grid(True)
plt.show()
print(f"\nTraining Samples: {len(train)}")
print(f"Testing Samples: {len(test)}")

print("\nEDA COMPLETE!")