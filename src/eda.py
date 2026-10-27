"""Week 3 - EDA and visualization."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns

sns.set_theme(style="whitegrid")
df = pd.read_csv("data/smart_logistics_dataset.csv", parse_dates=["Timestamp"])
df["Logistics_Delay_Reason"] = df["Logistics_Delay_Reason"].fillna("None")
df["Month"] = df.Timestamp.dt.to_period("M").astype(str)
df["Hour"] = df.Timestamp.dt.hour
num = ["Inventory_Level", "Temperature", "Humidity", "Waiting_Time",
       "User_Transaction_Amount", "User_Purchase_Frequency", "Asset_Utilization",
       "Demand_Forecast", "Logistics_Delay"]

# EDA numbers
desc = df[num].describe().T; desc["median"] = df[num].median(); desc["skew"] = df[num].skew()
desc.round(2).to_csv("outputs/summary_stats.csv")
corr = df[num].corr(); corr.round(3).to_csv("outputs/correlations.csv")
print(desc.round(2).to_string()); print(corr["Logistics_Delay"].sort_values().round(3).to_string())
print(df.groupby("Traffic_Status").Logistics_Delay.mean().round(3).to_string())
print(df.groupby("Asset_ID").Logistics_Delay.mean().round(3).sort_values().to_string())
print(df.groupby("Logistics_Delay_Reason").Waiting_Time.mean().round(2).to_string())

def save(name): plt.tight_layout(); plt.savefig(f"outputs/{name}.png", dpi=130); plt.close()

# 1 Histogram + KDE: distribution
plt.figure(figsize=(7,4)); sns.histplot(df.Waiting_Time, bins=25, kde=True, color="#2a6f97")
plt.title("Distribution of waiting time (min)"); save("01_waiting_time_hist")
# 2 Bar: delay rate by traffic
plt.figure(figsize=(6,4)); (df.groupby("Traffic_Status").Logistics_Delay.mean()*100).plot.bar(color="#2a6f97")
plt.ylabel("Delay rate (%)"); plt.title("Delay rate by traffic status"); plt.xticks(rotation=0); save("02_delay_by_traffic")
# 3 Box: waiting by delay reason
plt.figure(figsize=(7,4)); sns.boxplot(data=df, x="Logistics_Delay_Reason", y="Waiting_Time", color="#a9d6e5")
plt.title("Waiting time by delay reason"); save("03_wait_by_reason")
# 4 Heatmap: correlations
plt.figure(figsize=(8,6.5)); sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, annot_kws={"size":7})
plt.title("Correlation matrix"); save("04_correlation_heatmap")
# 5 Line: monthly delay rate
m = df.groupby("Month").Logistics_Delay.mean()*100
plt.figure(figsize=(8,4)); m.plot(marker="o", color="#2a6f97"); plt.ylabel("Delay rate (%)")
plt.title("Monthly delay rate"); plt.xticks(rotation=45); save("05_monthly_delay_trend")
# 6 Bar: delay rate per truck
plt.figure(figsize=(8,4)); (df.groupby("Asset_ID").Logistics_Delay.mean()*100).sort_values().plot.barh(color="#2a6f97")
plt.xlabel("Delay rate (%)"); plt.title("Delay rate by truck"); save("06_delay_by_truck")
# 7 Scatter: inventory vs demand
plt.figure(figsize=(6,5)); sns.scatterplot(data=df, x="Demand_Forecast", y="Inventory_Level", hue="Logistics_Delay", alpha=.6)
plt.title("Inventory vs demand forecast"); save("07_inventory_vs_demand")
# 8 Delay reasons share
plt.figure(figsize=(6,4)); df.Logistics_Delay_Reason.value_counts().plot.bar(color="#2a6f97")
plt.ylabel("Shipments"); plt.title("Delay reasons"); plt.xticks(rotation=0); save("08_delay_reasons")
# 9 Hour of day heat
pv = df.pivot_table(index="Asset_ID", columns=pd.cut(df.Hour,[-1,5,11,17,23],labels=["0-5","6-11","12-17","18-23"]), values="Logistics_Delay", aggfunc="mean", observed=True)
plt.figure(figsize=(6,5)); sns.heatmap(pv, annot=True, fmt=".2f", cmap="YlOrRd"); plt.title("Delay rate: truck x time of day"); save("09_truck_hour_heatmap")
