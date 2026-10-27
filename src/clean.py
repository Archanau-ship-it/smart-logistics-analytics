"""Week 2 - Data collection, cleaning, preprocessing pipeline."""
import numpy as np, pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_csv("data/smart_logistics_dataset.csv")
log = []   # audit trail for the report

def note(step, detail): log.append({"step": step, "detail": detail}); print(f"[{step}] {detail}")

# 1. Profile
note("shape", f"{df.shape[0]} rows x {df.shape[1]} cols")
note("missing(before)", df.isna().sum()[lambda s: s > 0].to_dict())
note("duplicates", int(df.duplicated().sum()))

# 2. Missing values: 'None' string was read as NaN by pandas -> it is a valid category
df["Logistics_Delay_Reason"] = df["Logistics_Delay_Reason"].fillna("None")
note("missing(after)", int(df.isna().sum().sum()))

# 3. Types
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
for c in ["Asset_ID", "Shipment_Status", "Traffic_Status", "Logistics_Delay_Reason"]:
    df[c] = df[c].astype("category")

# 4. Consistency checks (domain rules)
note("lat range ok", bool(df.Latitude.between(-90, 90).all()))
note("lon range ok", bool(df.Longitude.between(-180, 180).all()))
note("delay flag vs reason mismatch",
     int(((df.Logistics_Delay == 0) & (df.Logistics_Delay_Reason != "None")).sum() +
         ((df.Logistics_Delay == 1) & (df.Logistics_Delay_Reason == "None")).sum()))

# 5. Outliers: IQR rule
num = ["Inventory_Level", "Temperature", "Humidity", "Waiting_Time",
       "User_Transaction_Amount", "Asset_Utilization", "Demand_Forecast"]
for c in num:
    q1, q3 = df[c].quantile([.25, .75]); iqr = q3 - q1
    n = int(((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum())
    note("outliers IQR", f"{c}: {n}")

# 6. Feature engineering
df["Hour"] = df.Timestamp.dt.hour
df["DayOfWeek"] = df.Timestamp.dt.dayofweek
df["Month"] = df.Timestamp.dt.month
df["Stock_Gap"] = df.Inventory_Level - df.Demand_Forecast

# 7. Scaling (kept as separate columns so originals stay interpretable)
mm = MinMaxScaler().fit_transform(df[num])
for i, c in enumerate(num): df[f"{c}_minmax"] = mm[:, i]
zs = StandardScaler().fit_transform(df[num])
for i, c in enumerate(num): df[f"{c}_z"] = zs[:, i]

# 8. Encoding
df_model = pd.get_dummies(df, columns=["Traffic_Status", "Shipment_Status"], drop_first=True)
df.to_csv("outputs/cleaned_logistics.csv", index=False)
df_model.to_csv("outputs/model_ready_logistics.csv", index=False)
pd.DataFrame(log).to_csv("outputs/cleaning_log.csv", index=False)
note("saved", f"cleaned {df.shape}, model-ready {df_model.shape}")
