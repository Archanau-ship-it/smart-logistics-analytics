"""Week 1 - Strategic planning: baseline KPIs + roadmap illustration."""
import pandas as pd

df = pd.read_csv("data/smart_logistics_dataset.csv", parse_dates=["Timestamp"])

# NOTE: "None" in Logistics_Delay_Reason is a real label (no delay), not missing.
df["Logistics_Delay_Reason"] = df["Logistics_Delay_Reason"].fillna("None")

kpis = {
    "On-time rate (%)": round((1 - df["Logistics_Delay"].mean()) * 100, 1),
    "Delay rate (%)": round(df["Logistics_Delay"].mean() * 100, 1),
    "Avg waiting time (min)": round(df["Waiting_Time"].mean(), 1),
    "Avg asset utilization (%)": round(df["Asset_Utilization"].mean(), 1),
    "Forecast vs inventory gap (units)": round((df["Inventory_Level"] - df["Demand_Forecast"]).mean(), 1),
}
print(pd.Series(kpis).to_string())
pd.Series(kpis, name="baseline").to_csv("outputs/baseline_kpis.csv")

print("\nDelay rate by traffic status (%):")
print((df.groupby("Traffic_Status")["Logistics_Delay"].mean() * 100).round(1).to_string())
print("\nDelay reasons:")
print(df["Logistics_Delay_Reason"].value_counts().to_string())


