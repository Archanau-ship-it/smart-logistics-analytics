"""Week 4 - Predictive modeling + optimization."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score, roc_auc_score
from scipy.optimize import linprog
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RS = 42
df = pd.read_csv("data/smart_logistics_dataset.csv", parse_dates=["Timestamp"])
df["Logistics_Delay_Reason"] = df["Logistics_Delay_Reason"].fillna("None")
df["Hour"] = df.Timestamp.dt.hour; df["DayOfWeek"] = df.Timestamp.dt.dayofweek
df["Stock_Gap"] = df.Inventory_Level - df.Demand_Forecast
feat = ["Inventory_Level","Temperature","Humidity","User_Transaction_Amount","User_Purchase_Frequency",
        "Asset_Utilization","Demand_Forecast","Hour","DayOfWeek","Stock_Gap"]
X = pd.get_dummies(df[feat + ["Traffic_Status","Asset_ID"]], drop_first=True)

# ---------- A. Regression: Waiting_Time ----------
y = df.Waiting_Time
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=RS)
kf = KFold(5, shuffle=True, random_state=RS)
rows = []
for name, m in {"Baseline (mean)": DummyRegressor(), "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(200, random_state=RS, n_jobs=-1)}.items():
    cv = -cross_val_score(m, Xtr, ytr, cv=kf, scoring="neg_root_mean_squared_error").mean()
    m.fit(Xtr, ytr); p = m.predict(Xte)
    rows.append({"model": name, "CV_RMSE": cv, "test_MAE": mean_absolute_error(yte, p),
                 "test_RMSE": mean_squared_error(yte, p) ** .5, "test_R2": r2_score(yte, p)})
reg = pd.DataFrame(rows).round(3); print(reg.to_string()); reg.to_csv("outputs/regression_results.csv", index=False)

gs = GridSearchCV(RandomForestRegressor(random_state=RS, n_jobs=-1),
                  {"n_estimators": [100, 300], "max_depth": [3, 5, None], "min_samples_leaf": [1, 5, 15]},
                  cv=kf, scoring="neg_root_mean_squared_error").fit(Xtr, ytr)
p = gs.predict(Xte)
print("Tuned RF", gs.best_params_, "RMSE", round(mean_squared_error(yte, p)**.5, 3), "R2", round(r2_score(yte, p), 3))
pd.Series({"best_params": str(gs.best_params_), "test_MAE": mean_absolute_error(yte, p),
           "test_RMSE": mean_squared_error(yte, p)**.5, "test_R2": r2_score(yte, p)}).to_csv("outputs/tuned_rf_regression.csv")

# ---------- B. Classification: Logistics_Delay ----------
yc = df.Logistics_Delay
Xtr, Xte, ytr, yte = train_test_split(X, yc, test_size=.2, random_state=RS, stratify=yc)
rows = []
for name, m in {"Baseline (majority)": DummyClassifier(strategy="most_frequent"),
                "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                "Random Forest": RandomForestClassifier(300, random_state=RS, n_jobs=-1)}.items():
    cv = cross_val_score(m, Xtr, ytr, cv=5, scoring="roc_auc").mean() if name != "Baseline (majority)" else 0.5
    m.fit(Xtr, ytr); p = m.predict(Xte)
    auc = roc_auc_score(yte, m.predict_proba(Xte)[:, 1]) if hasattr(m, "predict_proba") else np.nan
    rows.append({"model": name, "CV_AUC": cv, "test_acc": accuracy_score(yte, p), "test_F1": f1_score(yte, p), "test_AUC": auc})
clf = pd.DataFrame(rows).round(3); print(clf.to_string()); clf.to_csv("outputs/classification_results.csv", index=False)

# Sanity check: how much of the skill is just "Heavy traffic"?
heavy_rule = (df.loc[yte.index, "Traffic_Status"] == "Heavy").astype(int)
rule = {"rule_acc": accuracy_score(yte, heavy_rule), "rule_F1": f1_score(yte, heavy_rule)}
nh = df.Traffic_Status != "Heavy"
Xn, yn = X[nh], yc[nh]
Xn_tr, Xn_te, yn_tr, yn_te = train_test_split(Xn, yn, test_size=.2, random_state=RS, stratify=yn)
rfn = RandomForestClassifier(300, random_state=RS, n_jobs=-1).fit(Xn_tr, yn_tr)
rule["non_heavy_only_AUC"] = roc_auc_score(yn_te, rfn.predict_proba(Xn_te)[:, 1])
print("Heavy-traffic rule:", {k: round(v, 3) for k, v in rule.items()})
pd.Series(rule).round(3).to_csv("outputs/sanity_checks.csv")

rf = RandomForestRegressor(200, random_state=RS, n_jobs=-1).fit(X, y)
imp = pd.Series(rf.feature_importances_, X.columns).sort_values().tail(10)
imp.plot.barh(color="#2a6f97", figsize=(7, 4)); plt.title("Top features - waiting time (RF)")
plt.tight_layout(); plt.savefig("outputs/10_feature_importance.png", dpi=130); plt.close()

# ---------- C. Optimization: route allocation minimising expected delays ----------
# Cost coefficient = observed delay rate per traffic status (data-driven).
d = df.groupby("Traffic_Status").Logistics_Delay.mean(); print(d.round(3).to_dict())
shipments = 300                                    # shipments to dispatch this shift
cap = {"Clear": 150, "Detour": 100, "Heavy": 100}  # assumed route capacity (illustrative)
order = list(d.index)
res = linprog(c=[d[k] for k in order], A_eq=[[1]*len(order)], b_eq=[shipments],
              bounds=[(0, cap[k]) for k in order])
alloc = pd.Series(res.x, index=order).round(0)
even = shipments / len(order) * d.sum()            # naive: equal split across routes
print("Optimal allocation:", alloc.to_dict(), "| expected delays", round(res.fun, 1), "vs even split", round(even, 1))
pd.DataFrame({"route": order, "allocated": alloc.values, "delay_rate": d.values}).to_csv("outputs/optimization_allocation.csv", index=False)
pd.Series({"optimized_expected_delays": res.fun, "even_split_expected_delays": even,
           "reduction_pct": (1 - res.fun / even) * 100}).round(2).to_csv("outputs/optimization_saving.csv")
