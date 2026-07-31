"""
Store Sales and Profit Analysis - Superstore Dataset
A simple, beginner-friendly Python analysis using pandas & matplotlib.
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("SampleSuperstore.csv", encoding="latin1")
df["Order Date"] = pd.to_datetime(df["Order Date"])

print("Shape of dataset:", df.shape)
print("\nColumns:", list(df.columns))
print("\nMissing values:\n", df.isnull().sum()[df.isnull().sum() > 0])

total_sales = df["Sales"].sum()
total_profit = df["Profit"].sum()
profit_margin = total_profit / total_sales * 100
total_orders = df["Order ID"].nunique()

print("\n--- OVERALL KPIs ---")
print(f"Total Sales:    ${total_sales:,.2f}")
print(f"Total Profit:   ${total_profit:,.2f}")
print(f"Profit Margin:  {profit_margin:.2f}%")
print(f"Total Orders:   {total_orders:,}")


cat = df.groupby("Category")[["Sales", "Profit"]].sum().sort_values("Sales", ascending=False)
print("\n--- SALES & PROFIT BY CATEGORY ---\n", cat)


subcat = df.groupby("Sub-Category")[["Sales", "Profit"]].sum().sort_values("Profit")
print("\n--- SUB-CATEGORIES SORTED BY PROFIT (LOWEST FIRST) ---\n", subcat)

region = df.groupby("Region")[["Sales", "Profit"]].sum().sort_values("Sales", ascending=False)
print("\n--- SALES & PROFIT BY REGION ---\n", region)


top_customers = df.groupby("Customer Name")["Sales"].sum().sort_values(ascending=False).head(10)
print("\n--- TOP 10 CUSTOMERS BY SALES ---\n", top_customers)


monthly = df.set_index("Order Date").resample("ME")[["Sales", "Profit"]].sum()


discount_profit = df.groupby("Discount")["Profit"].mean()




plt.style.use("seaborn-v0_8-whitegrid")


fig, ax = plt.subplots(figsize=(8, 5))
cat.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
ax.set_title("Sales & Profit by Category")
ax.set_ylabel("Amount ($)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("chart_category.png", dpi=150)
plt.close()


fig, ax = plt.subplots(figsize=(9, 6))
colors = ["#C44E52" if v < 0 else "#55A868" for v in subcat["Profit"]]
ax.barh(subcat.index, subcat["Profit"], color=colors)
ax.set_title("Profit by Sub-Category (Red = Loss-Making)")
ax.set_xlabel("Profit ($)")
plt.tight_layout()
plt.savefig("chart_subcategory_profit.png", dpi=150)
plt.close()


fig, ax = plt.subplots(figsize=(7, 5))
region.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
ax.set_title("Sales & Profit by Region")
ax.set_ylabel("Amount ($)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("chart_region.png", dpi=150)
plt.close()


fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(monthly.index, monthly["Sales"], label="Sales", marker="o", markersize=3)
ax.plot(monthly.index, monthly["Profit"], label="Profit", marker="o", markersize=3)
ax.set_title("Monthly Sales & Profit Trend")
ax.set_ylabel("Amount ($)")
ax.legend()
plt.tight_layout()
plt.savefig("chart_monthly_trend.png", dpi=150)
plt.close()


fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(discount_profit.index.astype(str), discount_profit.values, color="#8172B2", width=0.5)
ax.set_title("Average Profit by Discount Level")
ax.set_xlabel("Discount")
ax.set_ylabel("Average Profit ($)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("chart_discount_profit.png", dpi=150)
plt.close()

print("\nAll charts saved successfully.")