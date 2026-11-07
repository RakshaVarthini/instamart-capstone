"""Part B: RFM, K-means segments, PCA, lapse prediction."""
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, silhouette_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
import os
os.makedirs("outputs", exist_ok=True)
o = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
c = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
p = pd.read_csv("data/clean/products.csv")
comp = o[o.status == "completed"]

# ---------------- Task 1: RFM ----------------
SNAP = pd.Timestamp("2025-10-01 00:00:00")
g = comp.groupby("customer_id")
rfm = pd.DataFrame({"recency": (SNAP - g["order_ts"].max()).dt.days,
                    "frequency": g["order_id"].nunique(),
                    "monetary": g["order_amount"].sum()}).reset_index()
print("=== RFM ===")
print("customers:", len(rfm))
print("recency median/min/max:", rfm.recency.median(), rfm.recency.min(), rfm.recency.max())
print("frequency mean/max: %.2f %d" % (rfm.frequency.mean(), rfm.frequency.max()))
print("monetary mean/max: %.2f %.2f" % (rfm.monetary.mean(), rfm.monetary.max()))

# ---------------- Task 2: K-means ----------------
X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
print("\n=== K-means K=2..8 ===")
rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
    rows.append((k, km.inertia_, silhouette_score(X, km.labels_)))
    print(f"K={k}: inertia {km.inertia_:.1f} silhouette {rows[-1][2]:.3f}")

# ---------------- Task 3: profile ----------------
K = 4
km = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(X)
rfm["segment_id"] = km.labels_
prof = rfm.groupby("segment_id").agg(size=("customer_id", "count"), recency=("recency", "mean"),
                                     frequency=("frequency", "mean"), monetary=("monetary", "mean")).round(2)
print("\n=== Segment profile (K=4) ===")
print(prof.to_string())
# names assigned from the profile (cluster numbers are arbitrary labels, so names are rule-based)
names = {}
prem = prof.monetary.idxmax(); names[prem] = "Premium Big Spenders"
rest = prof.drop(index=prem)
loyal = rest.frequency.idxmax(); names[loyal] = "Loyal Regulars"
rest = rest.drop(index=loyal)
slip = rest.recency.idxmax(); names[slip] = "Slipping Away"
rest = rest.drop(index=slip)
for i in rest.index: names[i] = "Occasional Shoppers"
rfm["segment_name"] = rfm.segment_id.map(names)
rfm["is_premium"] = rfm.segment_id == prem
print(prof.assign(name=pd.Series(names)).to_string())
pca = PCA(n_components=2).fit(X)
print("PCA explained_variance_ratio_:", pca.explained_variance_ratio_.round(3), "total", round(pca.explained_variance_ratio_.sum(), 3))
rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]].to_csv(
    "outputs/customer_segments.csv", index=False)

# ---------------- Task 4: lapse definition ----------------
CUT = pd.Timestamp("2025-09-01")
pre = o[(o.order_ts < CUT)]                       # only orders before cutoff
pre_c = pre[pre.status == "completed"].merge(p[["product_id", "category"]], on="product_id")
cohort_ids = pre_c.customer_id.unique()
after = o[(o.order_ts >= CUT) & (o.order_ts < "2025-10-01") & (o.status == "completed")]
active = set(after.customer_id)
gg = pre_c.groupby("customer_id")
feat = pd.DataFrame({
    "recency_days": (CUT - gg["order_ts"].max()).dt.days,
    "frequency": gg["order_id"].nunique(),
    "monetary": gg["order_amount"].sum(),
    "avg_discount_pct": gg["discount_pct"].mean(),
    "avg_delivery_minutes": gg["delivery_minutes"].mean(),
    "fav_category": gg["category"].agg(lambda s: s.mode().sort_values().iloc[0]),
})
fin = pre[pre.status.isin(["completed", "cancelled"])]
feat["cancel_rate"] = (fin.assign(x=fin.status == "cancelled").groupby("customer_id")["x"].mean()
                       .reindex(feat.index).fillna(0))
feat = feat.reset_index().merge(c, on="customer_id", how="left")
feat["tenure_days"] = (CUT - feat.signup_date).dt.days
feat["lapsed"] = (~feat.customer_id.isin(active)).astype(int)
NUM = ["recency_days", "frequency", "monetary", "avg_discount_pct", "cancel_rate", "avg_delivery_minutes", "age", "tenure_days"]
CAT = ["city", "gender", "fav_category"]
y = feat["lapsed"]; Xf = feat[NUM + CAT]
Xtr, Xte, ytr, yte = train_test_split(Xf, y, test_size=0.2, stratify=y, random_state=SEED)
print("\n=== Lapse cohort ===")
print(f"cohort {len(feat)}; lapsed {int(y.sum())} ({y.mean()*100:.2f}%); train {len(Xtr)} test {len(Xte)}")
base = np.zeros(len(yte), dtype=int)
print("baseline accuracy %.3f lapsed-F1 %.3f" % (accuracy_score(yte, base), f1_score(yte, base, zero_division=0)))
print("lapsed in train:", int(ytr.sum()), "not lapsed in train:", int((ytr == 0).sum()))

# ---------------- Task 5: preprocessing ----------------
def prep():
    return ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT)])
assert not Xf.isna().any().any(), "missing values in features"
print("\nno missing value in any feature column: True")
pp = prep().fit(Xtr)
print("scaler mean of frequency (train only): %.4f | full-cohort mean: %.4f" % (
    pp.named_transformers_["num"].mean_[NUM.index("frequency")], feat.frequency.mean()))
print("(they differ because the scaler sees only the 924 training rows, never the test rows)")
print("encoded feature count:", len(pp.get_feature_names_out()))

# ---------------- Task 6: tuning ----------------
lr_gs = GridSearchCV(Pipeline([("prep", prep()), ("m", LogisticRegression(max_iter=1000))]),
                     {"m__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(Xtr, ytr)
rf_gs = GridSearchCV(Pipeline([("prep", prep()), ("m", RandomForestClassifier(n_estimators=100, random_state=SEED))]),
                     {"m__max_depth": [3, 5, 8], "m__min_samples_leaf": [5, 10, 20]}, cv=5, scoring="f1").fit(Xtr, ytr)
print("\n=== Tuning ===")
print("LR best", lr_gs.best_params_, "cv F1 %.4f" % lr_gs.best_score_)
print("RF best", rf_gs.best_params_, "cv F1 %.4f" % rf_gs.best_score_)
def ev(name, model, Xt=Xte):
    pr_ = model.predict(Xt)
    print(f"{name}: confusion [[TN FP],[FN TP]] = {confusion_matrix(yte, pr_).tolist()} "
          f"precision {precision_score(yte, pr_, zero_division=0):.3f} recall {recall_score(yte, pr_, zero_division=0):.3f} "
          f"F1 {f1_score(yte, pr_, zero_division=0):.3f}")
    return pr_
print("\n=== Test-set evaluation (lapsed = positive class) ===")
ev("LogisticRegression", lr_gs.best_estimator_)
ev("RandomForest", rf_gs.best_estimator_)
best = rf_gs.best_estimator_ if rf_gs.best_score_ >= lr_gs.best_score_ else lr_gs.best_estimator_
best_name = "RandomForest" if best is rf_gs.best_estimator_ else "LogisticRegression"
print("selected by cross-validated F1 (not by test):", best_name)

# ---------------- Task 7: oversampling ----------------
tr = Xtr.copy(); tr["lapsed"] = ytr
minority = tr[tr.lapsed == 1]; majority = tr[tr.lapsed == 0]
extra = minority.sample(n=len(majority) - len(minority), replace=True, random_state=SEED)
bal = pd.concat([tr, extra])
print("\n=== Oversampling (train only) ===")
print("balanced counts:", bal.lapsed.value_counts().to_dict())
lr_best_params = {k.replace("m__", ""): v for k, v in lr_gs.best_params_.items()}
lr_bal = Pipeline([("prep", prep()), ("m", LogisticRegression(max_iter=1000, **lr_best_params))]).fit(bal[NUM + CAT], bal.lapsed)
print("-- un-balanced LR vs balanced LR on the original test set")
ev("LR un-balanced", lr_gs.best_estimator_)
ev("LR balanced   ", lr_bal)

# ---------------- Task 8: interpret, persist, use ----------------
names_out = rf_gs.best_estimator_.named_steps["prep"].get_feature_names_out()
imp = pd.Series(rf_gs.best_estimator_.named_steps["m"].feature_importances_, index=names_out).sort_values(ascending=False)
print("\n=== Top 5 RF importances ===")
print(imp.head(5).round(4).to_string())
lr_names = lr_gs.best_estimator_.named_steps["prep"].get_feature_names_out()
coef = pd.Series(lr_gs.best_estimator_.named_steps["m"].coef_[0], index=lr_names)
print("\n=== LR coefficients ===")
for f in ["num__recency_days", "num__frequency", "num__monetary", "num__tenure_days", "num__cancel_rate"]:
    print(f, round(coef[f], 4))
joblib.dump(best, "outputs/lapse_model.joblib")
re = joblib.load("outputs/lapse_model.joblib")
print("\nreloaded model reproduces test predictions exactly:", bool((re.predict(Xte) == best.predict(Xte)).all()))
ev("selected model " + best_name, best)
te = feat.loc[Xte.index, ["customer_id"]].copy()
te["lapse_probability"] = best.predict_proba(Xte)[:, 1]
te["actually_lapsed"] = yte.values
prem_ids = set(rfm.loc[rfm.is_premium, "customer_id"])
ret = te[te.customer_id.isin(prem_ids)].sort_values("lapse_probability", ascending=False).head(10)
ret["lapse_probability"] = ret.lapse_probability.round(4)
ret.to_csv("outputs/retention_list.csv", index=False)
print("\nPremium customers in test set:", int(te.customer_id.isin(prem_ids).sum()))
print(ret.to_string(index=False))
print("truly lapsed among list:", int(ret.actually_lapsed.sum()), "of", len(ret))
